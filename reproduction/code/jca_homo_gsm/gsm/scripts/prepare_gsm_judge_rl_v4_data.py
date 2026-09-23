#!/usr/bin/env python3
"""Build step-correctness signed-AWR data from judged GSM rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from jca.gsm.scripts.prepare_gsm_judge_rl_v3_data import (
    AGENT_IDS,
    convert_sft_replay,
    deterministic_holdout_ids,
    parse_response,
    read_jsonl,
    stratified_sample,
    write_jsonl,
)
from jca.gsm.scripts.run_mas import answers_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-coverage GSM signed-AWR v4 data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--a1-late-sft-path", type=Path, required=True)
    parser.add_argument(
        "--expected-rollouts-per-problem",
        type=int,
        default=8,
        help="Required rollout count used to identify all-failed problems.",
    )
    parser.add_argument(
        "--drop-all-failed-problems",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop every row and replay for problems with no correct rollout.",
    )
    parser.add_argument(
        "--a1-late-replay-rows",
        type=int,
        default=0,
        help="Maximum clean A1 turn>0 replay rows; 0 keeps every eligible row.",
    )
    parser.add_argument("--a1-late-replay-weight", type=float, default=1.0)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--sft-replay-ratio", type=float, default=0.3)
    parser.add_argument("--sft-replay-weight", type=float, default=1.0)
    parser.add_argument("--max-rl-rows-per-agent", type=int, default=3000)
    parser.add_argument("--max-holdout-rows-per-agent", type=int, default=600)
    parser.add_argument(
        "--max-judged-rows-per-agent-problem", type=int, default=4
    )
    parser.add_argument("--min-abs-advantage", type=float, default=0.15)
    parser.add_argument("--advantage-min", type=float, default=-1.0)
    parser.add_argument("--advantage-max", type=float, default=1.0)
    parser.add_argument("--min-train-rows-per-agent", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def route_name(response: Dict[str, Any]) -> str:
    if response.get("action") == "confirm_stop":
        return "stop"
    return str(response.get("handoff_target") or "missing")


def turn_bucket(turn: int) -> str:
    return str(turn) if turn < 3 else "3+"


def stable_tiebreak(row: Dict[str, Any], seed: int) -> str:
    identity = ":".join((
        str(seed),
        str(row.get("problem_id")),
        str(row.get("rollout_idx")),
        str(row.get("turn")),
        str(row.get("agent_id")),
        str(row.get("reward_class_v4")),
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def previous_state_label(value: Any) -> str:
    if value is None:
        return "none"
    return "correct" if bool(value) else "wrong"


def validate_source_row(
    source: Dict[str, Any],
    expected_start_agent: str = "A1",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if expected_start_agent not in AGENT_IDS:
        raise ValueError(f"invalid expected start agent: {expected_start_agent}")
    response = parse_response(source)
    judged = source["collaboration_judge_v3"]
    state = source["deterministic_state_v3"]
    if not isinstance(judged, dict) or not isinstance(state, dict):
        raise TypeError("missing collaboration judge dictionaries")
    if not isinstance(state.get("current_answer_correct"), bool):
        raise ValueError("current answer correctness must be boolean")
    if state.get("previous_answer_correct") is not None and not isinstance(
        state.get("previous_answer_correct"), bool
    ):
        raise ValueError("previous answer correctness must be boolean or null")
    agent = str(source.get("agent_id") or "")
    action = str(response.get("action") or "")
    target = response.get("handoff_target")
    turn = int(source.get("turn", -1))
    if agent not in AGENT_IDS:
        raise ValueError("unknown agent")
    if action not in {"handoff", "confirm_stop"}:
        raise ValueError("invalid action")
    if action == "handoff" and target not in AGENT_IDS:
        raise ValueError("invalid handoff target")
    if action == "confirm_stop":
        tentative = str(response.get("tentative_answer") or "").strip()
        confirmed = str(response.get("confirmed_answer") or "").strip()
        if not tentative or not confirmed:
            raise ValueError("confirm_stop missing canonical answer")
        # MATH rows carry a versioned symbolic evaluator.  Use it for the
        # protocol invariant instead of GSM's number-extraction heuristic,
        # which rejects valid forms such as ``0.5`` vs ``\\frac{1}{2}`` and
        # can accept unrelated expressions that happen to share a digit.
        if source.get("math_eval_version", "") == "math_eval_symbolic_v4":
            from jca.src.math_eval import math_answers_equivalent

            answers_are_equal = math_answers_equivalent(tentative, confirmed)
        else:
            answers_are_equal = answers_match(tentative, confirmed)
        if not answers_are_equal:
            raise ValueError("confirm_stop tentative/confirmed mismatch")
    if turn == 0 and (
        agent != expected_start_agent
        or action != "handoff"
        or target not in set(AGENT_IDS) - {expected_start_agent}
    ):
        raise ValueError(f"invalid fixed-{expected_start_agent} proposal")
    return response, judged, state


def step_state_reward(state: Dict[str, Any]) -> Tuple[float, str]:
    """Return a signed reward whose sign is fixed by this turn's answer."""
    previous_correct = state.get("previous_answer_correct")
    current_correct = state["current_answer_correct"]
    if previous_correct is False and current_correct:
        return 1.0, "wrong_to_correct"
    if previous_correct is True and not current_correct:
        return -1.0, "correct_to_wrong"
    if previous_correct is False:
        return -0.6, "wrong_to_wrong"
    if previous_correct is True:
        return 0.6, "correct_to_correct"
    if current_correct:
        return 0.7, "proposal_correct"
    return -0.7, "proposal_wrong"


def build_raw_candidate(source: Dict[str, Any]) -> Dict[str, Any]:
    response, judged, state = validate_source_row(source)
    agent = str(source["agent_id"])
    turn = int(source["turn"])
    base_reward, reward_class = step_state_reward(state)
    judge_score = max(-1.0, min(1.0, float(judged.get("judge_score", 0.0))))
    reward_sign = 1.0 if base_reward > 0 else -1.0
    judge_multiplier = 1.0 + 0.1 * reward_sign * judge_score
    process_reward = base_reward * judge_multiplier

    try:
        terminal_em = float(source["em"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing terminal EM") from exc
    if terminal_em not in {0.0, 1.0}:
        raise ValueError("terminal EM must be binary")
    raw_reward = max(-1.0, min(1.0, process_reward))

    output = dict(source)
    output["response"] = json.dumps(response, ensure_ascii=False)
    output["action"] = str(response["action"])
    output["route_v4"] = route_name(response)
    output["answer_state_v4"] = str(state.get("answer_state") or "unknown")
    output["reward_class_v4"] = reward_class
    output["step_em_v4"] = int(state["current_answer_correct"])
    output["step_state_reward_v4"] = base_reward
    output["judge_multiplier_v4"] = round(judge_multiplier, 6)
    output["process_reward_v4"] = round(process_reward, 6)
    output["terminal_em_v4"] = terminal_em
    output["raw_reward_v4"] = round(raw_reward, 6)
    output["previous_state_v4"] = previous_state_label(
        state.get("previous_answer_correct")
    )
    output["sampling_stratum_v4"] = "|".join((
        agent,
        turn_bucket(turn),
        reward_class,
        str(response["action"]),
        route_name(response),
    ))
    return output


def all_failed_problem_ids(
    rows: Sequence[Dict[str, Any]],
    *,
    expected_rollouts_per_problem: int,
) -> set[str]:
    """Find problems whose complete rollout set has binary terminal EM = 0."""
    outcomes: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        problem_id = str(row.get("problem_id") or "")
        rollout_idx = int(row.get("rollout_idx", -1))
        if not problem_id or rollout_idx < 0:
            raise ValueError("invalid problem or rollout identity")
        try:
            terminal_em = float(row["em"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing terminal EM") from exc
        if terminal_em not in {0.0, 1.0}:
            raise ValueError("terminal EM must be binary")
        previous = outcomes[problem_id].get(rollout_idx)
        if previous is not None and previous != terminal_em:
            raise ValueError(
                f"inconsistent terminal EM for {problem_id} rollout {rollout_idx}"
            )
        outcomes[problem_id][rollout_idx] = terminal_em
    incomplete = {
        problem_id: len(rollouts)
        for problem_id, rollouts in outcomes.items()
        if len(rollouts) != expected_rollouts_per_problem
    }
    if incomplete:
        sample = sorted(incomplete.items())[:5]
        raise ValueError(
            "unexpected rollout count while filtering all-failed problems: "
            f"{sample}"
        )
    return {
        problem_id
        for problem_id, rollouts in outcomes.items()
        if not any(rollouts.values())
    }


def build_raw_candidates(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    candidates: List[Dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for source in rows:
        try:
            candidates.append(build_raw_candidate(source))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected[str(exc) or type(exc).__name__] += 1
    return candidates, rejected


def validate_fixed_a1_source(
    rows: Sequence[Dict[str, Any]],
    expected_start_agent: str = "A1",
) -> int:
    if expected_start_agent not in AGENT_IDS:
        raise ValueError(f"invalid expected start agent: {expected_start_agent}")
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        declared_start = row.get("start_agent")
        if declared_start is not None and str(declared_start) != expected_start_agent:
            raise ValueError(
                "non-A1 declared start-agent mismatch: "
                f"expected {expected_start_agent}, found {declared_start}"
            )
        key = (
            str(row.get("problem_id") or ""),
            int(row.get("rollout_idx", -1)),
        )
        grouped[key].append(row)
    for key, group in grouped.items():
        first = min(group, key=lambda row: int(row.get("turn", -1)))
        if (
            int(first.get("turn", -1)) != 0
            or str(first.get("agent_id")) != expected_start_agent
        ):
            raise ValueError(
                f"rollout group does not start with {expected_start_agent}: {key}"
            )
    return len(grouped)


def verifier_baselines(
    candidates: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in candidates:
        agent = str(row["agent_id"])
        if agent == "A1":
            continue
        grouped[(agent, str(row["previous_state_v4"]))].append(
            float(row["raw_reward_v4"])
        )
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def attach_advantages(
    candidates: Sequence[Dict[str, Any]],
    baselines: Dict[Tuple[str, str], float],
    *,
    advantage_min: float,
    advantage_max: float,
    min_abs_advantage: float,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    a1_problem_rewards: Dict[str, List[float]] = defaultdict(list)
    for row in candidates:
        if str(row["agent_id"]) == "A1":
            a1_problem_rewards[str(row["problem_id"])].append(
                float(row["raw_reward_v4"])
            )
    a1_means = {
        problem_id: statistics.fmean(values)
        for problem_id, values in a1_problem_rewards.items()
    }

    selected: List[Dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for source in candidates:
        row = dict(source)
        raw_reward = float(row["raw_reward_v4"])
        agent = str(row["agent_id"])
        if agent == "A1":
            relative = raw_reward - a1_means[str(row["problem_id"])]
            advantage = 0.65 * raw_reward + 0.35 * relative
        else:
            key = (agent, str(row["previous_state_v4"]))
            baseline = baselines.get(key, 0.0)
            advantage = 0.75 * raw_reward + 0.25 * (raw_reward - baseline)
        advantage = max(advantage_min, min(advantage_max, advantage))
        if raw_reward * advantage < 0:
            advantage = math.copysign(abs(advantage), raw_reward)
        if abs(advantage) < min_abs_advantage:
            dropped["near_zero_advantage"] += 1
            continue
        row["train_weight"] = round(advantage, 6)
        row["reward"] = round(advantage, 6)
        row["sample_source_v4"] = "judged_signed_awr"
        row["loss_mask_mode_v4"] = (
            "decision" if advantage < 0 and agent in {"A2", "A3"} else "full"
        )
        if row["loss_mask_mode_v4"] == "decision":
            reward_class = str(row["reward_class_v4"])
            field_map = {
                "proposal_wrong": ("tentative_answer",),
                "wrong_to_wrong": ("tentative_answer",),
                "correct_to_wrong": (
                    "tentative_answer",
                    "confirmed_answer",
                ),
            }
            fields = field_map.get(reward_class)
            if not fields:
                raise ValueError(
                    f"missing negative decision fields for {reward_class}"
                )
            row["decision_fields_v4"] = list(fields)
        selected.append(row)
    return selected, dropped


def cap_per_agent_problem(
    rows: Sequence[Dict[str, Any]],
    *,
    cap: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agent = str(row["agent_id"])
        phase = (
            "late"
            if agent == "A1" and int(row.get("turn", -1)) > 0
            else "start_or_verifier"
        )
        grouped[(agent, str(row["problem_id"]), phase)].append(row)
    kept: List[Dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for (agent, _problem_id, _phase), group in grouped.items():
        positive = [row for row in group if float(row["train_weight"]) > 0]
        negative = [row for row in group if float(row["train_weight"]) < 0]
        for sign_rows in (positive, negative):
            sign_rows.sort(
                key=lambda row: (
                    -abs(float(row["train_weight"])),
                    stable_tiebreak(row, seed),
                )
            )
        per_sign = cap // 2
        chosen = positive[:per_sign] + negative[:per_sign]
        remaining = positive[per_sign:] + negative[per_sign:]
        remaining.sort(
            key=lambda row: (
                -abs(float(row["train_weight"])),
                stable_tiebreak(row, seed + 1),
            )
        )
        chosen.extend(remaining[: max(0, cap - len(chosen))])
        kept.extend(chosen)
        dropped[f"problem_cap:{agent}"] += len(group) - len(chosen)
    return kept, dropped


def round_robin_sample(
    rows: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sampling_stratum_v4"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    strata = sorted(groups)
    rng.shuffle(strata)
    selected: List[Dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if groups[stratum]:
                selected.append(groups[stratum].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def sample_a1_sign(
    rows: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    late_rows = [row for row in rows if int(row.get("turn", -1)) > 0]
    selected_late = round_robin_sample(late_rows, min(count, len(late_rows)), seed=seed)
    remaining = count - len(selected_late)
    remaining -= remaining % 2
    start_rows = [row for row in rows if int(row.get("turn", -1)) == 0]
    by_route = {
        route: [row for row in start_rows if str(row["route_v4"]) == route]
        for route in ("A2", "A3")
    }
    per_route = min(remaining // 2, *(len(group) for group in by_route.values()))
    selected: List[Dict[str, Any]] = list(selected_late)
    for route_index, route in enumerate(("A2", "A3")):
        selected.extend(round_robin_sample(
            by_route[route],
            per_route,
            seed=seed + route_index,
        ))
    return selected


def select_signed_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_rows_per_agent: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    selected: List[Dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for agent_index, agent in enumerate(AGENT_IDS):
        agent_rows = [row for row in rows if str(row["agent_id"]) == agent]
        positive = [row for row in agent_rows if float(row["train_weight"]) > 0]
        negative = [row for row in agent_rows if float(row["train_weight"]) < 0]
        per_sign = min(max_rows_per_agent // 2, len(positive), len(negative))
        if agent == "A1":
            chosen_positive = sample_a1_sign(
                positive, per_sign, seed=seed + 101 * agent_index
            )
            chosen_negative = sample_a1_sign(
                negative, per_sign, seed=seed + 211 * agent_index
            )
            balanced_per_sign = min(len(chosen_positive), len(chosen_negative))
            chosen_positive = sample_a1_sign(
                positive,
                balanced_per_sign,
                seed=seed + 307 * agent_index,
            )
            chosen_negative = sample_a1_sign(
                negative,
                balanced_per_sign,
                seed=seed + 401 * agent_index,
            )
        else:
            chosen_positive = round_robin_sample(
                positive, per_sign, seed=seed + 101 * agent_index
            )
            chosen_negative = round_robin_sample(
                negative, per_sign, seed=seed + 211 * agent_index
            )
        chosen = chosen_positive + chosen_negative
        selected.extend(chosen)
        dropped[f"sign_balance:{agent}:positive"] += len(positive) - len(chosen_positive)
        dropped[f"sign_balance:{agent}:negative"] += len(negative) - len(chosen_negative)
    random.Random(seed).shuffle(selected)
    return selected, dropped


def build_replay_rows(
    selected_rl: Sequence[Dict[str, Any]],
    sft_rows_by_agent: Dict[str, Sequence[Dict[str, Any]]],
    holdout_ids: set[str],
    *,
    replay_ratio: float,
    replay_weight: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rl_counts = Counter(str(row["agent_id"]) for row in selected_rl)
    replay: List[Dict[str, Any]] = []
    for agent_index, agent in enumerate(AGENT_IDS):
        desired = round(rl_counts[agent] * replay_ratio / (1.0 - replay_ratio))
        eligible = [
            row
            for row in sft_rows_by_agent[agent]
            if str(row.get("problem_id") or "") not in holdout_ids
        ]
        if agent == "A1":
            desired -= desired % 2
            by_route: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in eligible:
                response = parse_response(row, "label")
                if int(row.get("turn", -1)) == 0:
                    by_route[route_name(response)].append(row)
            sampled: List[Dict[str, Any]] = []
            for route_index, route in enumerate(("A2", "A3")):
                sampled.extend(stratified_sample(
                    by_route[route],
                    desired // 2,
                    seed=seed + 1009 * (agent_index + 1) + route_index,
                ))
        else:
            sampled = stratified_sample(
                eligible,
                desired,
                seed=seed + 1009 * (agent_index + 1),
            )
        if len(sampled) < desired:
            raise ValueError(
                f"not enough non-holdout SFT replay for {agent}: "
                f"need {desired}, found {len(sampled)}"
            )
        for source in sampled:
            row = convert_sft_replay(source, replay_weight)
            row["sample_source_v4"] = "sft_protocol_replay"
            row["loss_mask_mode_v4"] = "full"
            row["reward_class_v4"] = "sft_replay"
            row["raw_reward_v4"] = replay_weight
            row["decision_fields_v4"] = []
            row["route_v4"] = row.pop("route_v3")
            row["sampling_stratum_v4"] = row.pop("sampling_stratum_v3")
            replay.append(row)
    return replay


def build_a1_late_replay_rows(
    source_rows: Sequence[Dict[str, Any]],
    holdout_ids: set[str],
    *,
    max_rows: int,
    replay_weight: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    eligible: List[Dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for source in source_rows:
        try:
            if str(source.get("agent_id") or "") != "A1":
                raise ValueError("non-A1 row")
            if int(source.get("turn", -1)) <= 0:
                raise ValueError("turn-zero row")
            if str(source.get("problem_id") or "") in holdout_ids:
                raise ValueError("holdout problem")
            response = parse_response(source, "label")
            action = str(response.get("action") or "")
            if action == "handoff":
                if response.get("handoff_target") not in {"A2", "A3"}:
                    raise ValueError("invalid A1 late handoff target")
                if response.get("confirmed_answer") is not None:
                    raise ValueError("A1 late handoff has confirmed answer")
            elif action == "confirm_stop":
                tentative = str(response.get("tentative_answer") or "").strip()
                confirmed = str(response.get("confirmed_answer") or "").strip()
                if not tentative or not confirmed:
                    raise ValueError("A1 late stop missing canonical answer")
                if not answers_match(tentative, confirmed):
                    raise ValueError("A1 late stop answer mismatch")
            else:
                raise ValueError("invalid A1 late action")
            eligible.append(source)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            excluded[str(exc) or type(exc).__name__] += 1

    desired = len(eligible) if max_rows == 0 else max_rows
    if len(eligible) < desired:
        raise ValueError(
            "not enough clean A1 late-state replay: "
            f"need {desired}, found {len(eligible)}"
        )
    sampled = stratified_sample(eligible, desired, seed=seed + 4001)
    replay: List[Dict[str, Any]] = []
    for source in sampled:
        row = convert_sft_replay(source, replay_weight)
        row["sample_source_v4"] = "a1_late_sft_replay"
        row["loss_mask_mode_v4"] = "full"
        row["reward_class_v4"] = "a1_late_sft_replay"
        row["raw_reward_v4"] = replay_weight
        row["decision_fields_v4"] = []
        row["route_v4"] = row.pop("route_v3")
        row["sampling_stratum_v4"] = row.pop("sampling_stratum_v3")
        replay.append(row)
    return replay, excluded


def counter(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field)) for row in rows))


def signed_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(
        "positive" if float(row["train_weight"]) > 0 else "negative"
        for row in rows
    ))


def advantage_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float | None]:
    values = [float(row["train_weight"]) for row in rows]
    return {
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def action_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for agent in AGENT_IDS:
        agent_rows = [row for row in rows if str(row.get("agent_id")) == agent]
        actions = Counter(str(row.get("action")) for row in agent_rows)
        output[agent] = {
            "actions": dict(actions),
            "confirm_stop_fraction": (
                actions["confirm_stop"] / len(agent_rows) if agent_rows else None
            ),
            "reward_signs": signed_counts(agent_rows),
        }
    return output


def build_data(
    source_rows: Sequence[Dict[str, Any]],
    sft_rows_by_agent: Dict[str, Sequence[Dict[str, Any]]],
    a1_late_sft_rows: Sequence[Dict[str, Any]],
    *,
    expected_rollouts_per_problem: int,
    drop_all_failed_problems: bool,
    holdout_fraction: float,
    replay_ratio: float,
    replay_weight: float,
    a1_late_replay_rows: int,
    a1_late_replay_weight: float,
    max_rl_rows_per_agent: int,
    max_holdout_rows_per_agent: int,
    max_judged_rows_per_agent_problem: int,
    min_abs_advantage: float,
    advantage_min: float,
    advantage_max: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_group_count = validate_fixed_a1_source(source_rows)
    all_failed_ids = all_failed_problem_ids(
        source_rows,
        expected_rollouts_per_problem=expected_rollouts_per_problem,
    )
    eligible_source = [
        row
        for row in source_rows
        if not drop_all_failed_problems
        or str(row.get("problem_id") or "") not in all_failed_ids
    ]
    holdout_ids = deterministic_holdout_ids(
        (str(row.get("problem_id") or "") for row in eligible_source),
        holdout_fraction,
        seed,
    )
    train_source = [
        row
        for row in eligible_source
        if str(row.get("problem_id") or "") not in holdout_ids
    ]
    holdout_source = [
        row
        for row in eligible_source
        if str(row.get("problem_id") or "") in holdout_ids
    ]
    train_raw, rejected_train = build_raw_candidates(train_source)
    holdout_raw, rejected_holdout = build_raw_candidates(holdout_source)
    baselines = verifier_baselines(train_raw)
    train_advantaged, advantage_dropped_train = attach_advantages(
        train_raw,
        baselines,
        advantage_min=advantage_min,
        advantage_max=advantage_max,
        min_abs_advantage=min_abs_advantage,
    )
    holdout_advantaged, advantage_dropped_holdout = attach_advantages(
        holdout_raw,
        baselines,
        advantage_min=advantage_min,
        advantage_max=advantage_max,
        min_abs_advantage=min_abs_advantage,
    )
    capped_train, cap_dropped_train = cap_per_agent_problem(
        train_advantaged,
        cap=max_judged_rows_per_agent_problem,
        seed=seed,
    )
    capped_holdout, cap_dropped_holdout = cap_per_agent_problem(
        holdout_advantaged,
        cap=max_judged_rows_per_agent_problem,
        seed=seed + 1,
    )
    selected_rl, balance_dropped_train = select_signed_rows(
        capped_train,
        max_rows_per_agent=max_rl_rows_per_agent,
        seed=seed,
    )
    holdout_rows, balance_dropped_holdout = select_signed_rows(
        capped_holdout,
        max_rows_per_agent=max_holdout_rows_per_agent,
        seed=seed + 1,
    )
    replay_excluded_ids = set(holdout_ids)
    if drop_all_failed_problems:
        replay_excluded_ids.update(all_failed_ids)
    replay_rows = build_replay_rows(
        selected_rl,
        sft_rows_by_agent,
        replay_excluded_ids,
        replay_ratio=replay_ratio,
        replay_weight=replay_weight,
        seed=seed,
    )
    late_replay_rows, late_replay_excluded = build_a1_late_replay_rows(
        a1_late_sft_rows,
        replay_excluded_ids,
        max_rows=a1_late_replay_rows,
        replay_weight=a1_late_replay_weight,
        seed=seed,
    )
    train_rows = list(selected_rl) + replay_rows + late_replay_rows
    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(holdout_rows)

    dropped = (
        advantage_dropped_train
        + cap_dropped_train
        + balance_dropped_train
    )
    holdout_dropped = (
        advantage_dropped_holdout
        + cap_dropped_holdout
        + balance_dropped_holdout
    )
    stats = {
        "version": "full_gsm_v4_step_em_transition_awr",
        "source_rows": len(source_rows),
        "source_rollout_groups": source_group_count,
        "source_problem_count": len({str(row.get("problem_id")) for row in source_rows}),
        "all_failed_problem_count": len(all_failed_ids),
        "all_failed_problem_ids": sorted(all_failed_ids),
        "all_failed_source_rows": len(source_rows) - len(eligible_source),
        "drop_all_failed_problems": drop_all_failed_problems,
        "eligible_source_rows": len(eligible_source),
        "eligible_problem_count": len({
            str(row.get("problem_id")) for row in eligible_source
        }),
        "raw_candidates": len(train_raw),
        "judged_signed_rows": len(selected_rl),
        "sft_replay_rows": len(replay_rows),
        "a1_late_sft_source_rows": len(a1_late_sft_rows),
        "a1_late_sft_replay_rows": len(late_replay_rows),
        "a1_late_sft_excluded": dict(late_replay_excluded),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_problem_ids": sorted(holdout_ids),
        "actual_sft_replay_ratio": (
            (len(replay_rows) + len(late_replay_rows)) / len(train_rows)
        ),
        "train_agents": counter(train_rows, "agent_id"),
        "train_sources": counter(train_rows, "sample_source_v4"),
        "train_reward_signs": signed_counts(train_rows),
        "judged_reward_signs": signed_counts(selected_rl),
        "train_reward_classes": counter(train_rows, "reward_class_v4"),
        "train_loss_masks": counter(train_rows, "loss_mask_mode_v4"),
        "negative_decision_field_sets": dict(Counter(
            "+".join(row.get("decision_fields_v4", []))
            for row in train_rows
            if float(row["train_weight"]) < 0
            and str(row["agent_id"]) in {"A2", "A3"}
        )),
        "train_advantage": advantage_stats(train_rows),
        "holdout_advantage": advantage_stats(holdout_rows),
        "train_actions_by_agent": action_stats(train_rows),
        "judged_actions_by_agent": action_stats(selected_rl),
        "a1_train_turns": counter(
            [row for row in train_rows if str(row.get("agent_id")) == "A1"],
            "turn",
        ),
        "step_terminal_sign_conflicts": sum(
            (
                float(row["process_reward_v4"]) > 0
                and float(row["terminal_em_v4"]) == 0
            )
            or (
                float(row["process_reward_v4"]) < 0
                and float(row["terminal_em_v4"]) == 1
            )
            for row in selected_rl
        ),
        "verifier_baselines": {
            "|".join(key): round(value, 6) for key, value in baselines.items()
        },
        "source_rejected": dict(rejected_train),
        "holdout_source_rejected": dict(rejected_holdout),
        "selection_dropped": {key: value for key, value in dropped.items() if value},
        "holdout_selection_dropped": {
            key: value for key, value in holdout_dropped.items() if value
        },
        "max_rl_rows_per_agent": max_rl_rows_per_agent,
        "max_judged_rows_per_agent_problem": max_judged_rows_per_agent_problem,
        "min_abs_advantage": min_abs_advantage,
        "advantage_clip": [advantage_min, advantage_max],
        "verifier_confirm_quota": "statistics_only_not_enforced",
        "training_contract": {
            "algorithm": "conservative_signed_awr",
            "fixed_start_agent": "A1",
            "a1_start_only": True,
            "a3_start_control": False,
            "all_agents_trained": True,
            "a1_negative_mask": "full",
            "verifier_negative_mask": "decision",
            "negative_objective": "reference_margin",
            "terminal_em_controls_reward_sign": False,
            "current_answer_correct_controls_reward_sign": True,
            "wrong_to_correct_reward": 1.0,
            "correct_to_wrong_reward": -1.0,
            "all_failed_problems_dropped": drop_all_failed_problems,
            "expected_rollouts_per_problem": expected_rollouts_per_problem,
            "canonical_stop_answers_required": True,
            "a1_late_state_replay": True,
            "a1_late_judged_rollout": True,
            "sft_replay_positive": True,
            "signed_negative_examples": sum(
                float(row["train_weight"]) < 0 for row in train_rows
            ),
            "reward_is_state_conditioned": True,
            "global_confirm_quota_enforced": False,
        },
    }
    return train_rows, holdout_rows, stats


def validate_outputs(
    train_rows: Sequence[Dict[str, Any]],
    holdout_rows: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
    *,
    min_train_rows_per_agent: int,
    advantage_min: float,
    advantage_max: float,
) -> None:
    if stats.get("drop_all_failed_problems"):
        all_failed_ids = set(stats.get("all_failed_problem_ids", []))
        leaked = {
            str(row.get("problem_id") or "")
            for row in [*train_rows, *holdout_rows]
            if str(row.get("problem_id") or "") in all_failed_ids
        }
        if leaked:
            raise ValueError(f"all-failed problems leaked into output: {sorted(leaked)[:5]}")
    for agent in AGENT_IDS:
        agent_rows = [row for row in train_rows if str(row["agent_id"]) == agent]
        if len(agent_rows) < min_train_rows_per_agent:
            raise ValueError(
                f"too few v4 training rows for {agent}: "
                f"{len(agent_rows)} < {min_train_rows_per_agent}"
            )
        if not any(float(row["train_weight"]) < 0 for row in agent_rows):
            raise ValueError(f"{agent} has no negative advantages")
        if not any(float(row["train_weight"]) > 0 for row in agent_rows):
            raise ValueError(f"{agent} has no positive advantages")
    for row in [*train_rows, *holdout_rows]:
        weight = float(row["train_weight"])
        if not math.isfinite(weight) or not advantage_min <= weight <= advantage_max:
            raise ValueError(f"invalid clipped advantage: {weight}")
        agent = str(row["agent_id"])
        mask_mode = str(row["loss_mask_mode_v4"])
        if weight < 0 and agent in {"A2", "A3"} and mask_mode != "decision":
            raise ValueError("verifier negative is not decision-masked")
        if weight < 0 and agent in {"A2", "A3"}:
            fields = row.get("decision_fields_v4")
            if not isinstance(fields, list) or not fields:
                raise ValueError("verifier negative has no decision fields")
        if weight < 0 and agent == "A1" and mask_mode != "full":
            raise ValueError("A1 negative is not full-response masked")
        if weight > 0 and mask_mode != "full":
            raise ValueError("positive sample is not full-response trained")
        if row.get("sample_source_v4") == "judged_signed_awr":
            current_correct = row["deterministic_state_v3"][
                "current_answer_correct"
            ]
            if current_correct != (weight > 0):
                raise ValueError("current answer correctness and reward sign disagree")
    a1_routes = Counter(
        str(row["route_v4"])
        for row in train_rows
        if row.get("agent_id") == "A1" and int(row.get("turn", -1)) == 0
    )
    if a1_routes.get("A2", 0) != a1_routes.get("A3", 0):
        raise ValueError(f"A1 first routes are not balanced: {dict(a1_routes)}")
    a1_late_rows = [
        row
        for row in train_rows
        if row.get("agent_id") == "A1" and int(row.get("turn", -1)) > 0
    ]
    if not a1_late_rows:
        raise ValueError("A1 has no late-state training rows")
    judged_a1_late = [
        row
        for row in a1_late_rows
        if row.get("sample_source_v4") == "judged_signed_awr"
    ]
    if not judged_a1_late:
        raise ValueError("A1 has no judged late-state training rows")
    if not any(row.get("previous_state_v4") == "wrong" for row in judged_a1_late):
        raise ValueError("A1 has no judged training rows after a wrong prior state")
    if stats["training_contract"]["signed_negative_examples"] <= 0:
        raise ValueError("signed training contract contains no negative examples")


def main() -> None:
    args = parse_args()
    if not 0 <= args.holdout_fraction < 1:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    if not 0 < args.sft_replay_ratio < 1:
        raise SystemExit("--sft-replay-ratio must be in (0, 1)")
    if args.max_rl_rows_per_agent <= 0 or args.max_holdout_rows_per_agent <= 0:
        raise SystemExit("row caps must be positive")
    if args.max_judged_rows_per_agent_problem < 2:
        raise SystemExit("problem cap must be at least 2")
    if args.expected_rollouts_per_problem <= 0:
        raise SystemExit("--expected-rollouts-per-problem must be positive")
    if args.a1_late_replay_rows < 0:
        raise SystemExit("--a1-late-replay-rows must be non-negative")
    if args.a1_late_replay_weight <= 0:
        raise SystemExit("--a1-late-replay-weight must be positive")
    if not args.advantage_min < 0 < args.advantage_max:
        raise SystemExit("advantage clip must span zero")
    if not 0 <= args.min_abs_advantage < args.advantage_max:
        raise SystemExit("invalid minimum absolute advantage")
    for path in (args.train_output, args.holdout_output, args.stats_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {path}; use --overwrite or a new path")

    source_rows = read_jsonl(args.input)
    sft_rows_by_agent = {
        agent: read_jsonl(args.sft_data_dir / f"{agent}.jsonl")
        for agent in AGENT_IDS
    }
    a1_late_sft_rows = read_jsonl(args.a1_late_sft_path)
    train_rows, holdout_rows, stats = build_data(
        source_rows,
        sft_rows_by_agent,
        a1_late_sft_rows,
        expected_rollouts_per_problem=args.expected_rollouts_per_problem,
        drop_all_failed_problems=args.drop_all_failed_problems,
        holdout_fraction=args.holdout_fraction,
        replay_ratio=args.sft_replay_ratio,
        replay_weight=args.sft_replay_weight,
        a1_late_replay_rows=args.a1_late_replay_rows,
        a1_late_replay_weight=args.a1_late_replay_weight,
        max_rl_rows_per_agent=args.max_rl_rows_per_agent,
        max_holdout_rows_per_agent=args.max_holdout_rows_per_agent,
        max_judged_rows_per_agent_problem=args.max_judged_rows_per_agent_problem,
        min_abs_advantage=args.min_abs_advantage,
        advantage_min=args.advantage_min,
        advantage_max=args.advantage_max,
        seed=args.seed,
    )
    validate_outputs(
        train_rows,
        holdout_rows,
        stats,
        min_train_rows_per_agent=args.min_train_rows_per_agent,
        advantage_min=args.advantage_min,
        advantage_max=args.advantage_max,
    )
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.holdout_output, holdout_rows)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("GSM signed-AWR full-data v4 gate: PASS")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
