#!/usr/bin/env python3
"""Build stable, collaboration-balanced GSM offline-RL v3 training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


AGENT_IDS = ("A1", "A2", "A3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare collaboration-balanced, positive-only GSM RL v3 data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--sft-replay-ratio", type=float, default=0.3)
    parser.add_argument("--sft-replay-weight", type=float, default=1.0)
    parser.add_argument("--min-reasoning-score", type=float, default=0.2)
    parser.add_argument("--min-verification-score", type=float, default=0.2)
    parser.add_argument("--min-action-score", type=float, default=0.0)
    parser.add_argument("--min-route-score", type=float, default=0.0)
    parser.add_argument("--min-quality-score", type=float, default=0.2)
    parser.add_argument("--route-cap-multiplier", type=float, default=1.5)
    parser.add_argument("--min-train-rows-per-agent", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def parse_response(row: Dict[str, Any], field: str = "response") -> Dict[str, Any]:
    payload = row.get(field)
    if isinstance(payload, dict):
        return dict(payload)
    if not isinstance(payload, str):
        raise ValueError(f"{field} is not a JSON string")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} JSON is not an object")
    return parsed


def turn_bucket(turn: int) -> str:
    return str(turn) if turn < 3 else "3+"


def route_name(response: Dict[str, Any]) -> str:
    if response.get("action") == "confirm_stop":
        return "stop"
    return str(response.get("handoff_target") or "missing")


def deterministic_holdout_ids(
    problem_ids: Iterable[str],
    fraction: float,
    seed: int,
) -> set[str]:
    unique = sorted(set(problem_ids))
    count = max(1, round(len(unique) * fraction)) if unique and fraction > 0 else 0
    ranked = sorted(
        unique,
        key=lambda problem_id: hashlib.sha256(
            f"{seed}:{problem_id}".encode("utf-8")
        ).hexdigest(),
    )
    return set(ranked[:count])


def quality_score(turn: int, judged: Dict[str, Any]) -> float:
    reasoning = float(judged["reasoning_score"])
    action = float(judged["action_score"])
    route = float(judged["route_score"])
    if turn == 0:
        return 0.50 * reasoning + 0.30 * action + 0.20 * route
    verification = float(judged["verification_score"])
    return (
        0.35 * reasoning
        + 0.30 * verification
        + 0.20 * action
        + 0.15 * route
    )


def _reject_reason(
    row: Dict[str, Any],
    response: Dict[str, Any],
    judged: Dict[str, Any],
    state: Dict[str, Any],
    *,
    min_reasoning_score: float,
    min_verification_score: float,
    min_action_score: float,
    min_route_score: float,
    min_quality_score: float,
) -> str | None:
    turn = int(row.get("turn", -1))
    agent = str(row.get("agent_id", ""))
    action = str(response.get("action") or "")
    target = response.get("handoff_target")
    changed = bool(state.get("answer_changed"))
    previous_correct = state.get("previous_answer_correct")
    answer_state = str(state.get("answer_state") or "")

    if agent not in AGENT_IDS:
        return "unknown_agent"
    if not bool(state.get("current_answer_correct")):
        return "current_answer_wrong"
    if action not in {"handoff", "confirm_stop"}:
        return "invalid_action"
    if action == "handoff" and target not in AGENT_IDS:
        return "invalid_handoff_target"
    if turn == 0 and (agent != "A1" or action != "handoff" or target not in {"A2", "A3"}):
        return "invalid_fixed_a1_proposal"
    if turn > 0 and not bool(judged.get("independent_verification")):
        return "not_independent_verification"
    if changed and action != "handoff":
        return "changed_answer_without_handoff"
    if action == "confirm_stop" and previous_correct is not True:
        return "confirm_without_correct_consensus"
    if action == "confirm_stop" and changed:
        return "correction_confirmed_immediately"
    if float(judged.get("reasoning_score", -1.0)) < min_reasoning_score:
        return "low_reasoning_score"
    if turn > 0 and float(judged.get("verification_score", -1.0)) < min_verification_score:
        return "low_verification_score"
    if float(judged.get("action_score", -1.0)) < min_action_score:
        return "low_action_score"
    if float(judged.get("route_score", -1.0)) < min_route_score:
        return "low_route_score"
    if quality_score(turn, judged) < min_quality_score:
        return "low_quality_score"
    recommended = str(judged.get("recommended_action") or "")
    if recommended not in {"handoff", "confirm_stop"}:
        return "invalid_recommended_action"
    if action == "confirm_stop" and recommended != action:
        return "judge_recommends_handoff"
    if action == "handoff" and recommended == "confirm_stop" and float(
        judged.get("route_score", -1.0)
    ) < 0.3:
        return "redundant_handoff"
    if answer_state == "correct_to_correct" and action == "handoff" and float(
        judged.get("route_score", -1.0)
    ) < 0.2:
        return "established_answer_loop"
    return None


def select_positive_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    min_reasoning_score: float = 0.2,
    min_verification_score: float = 0.2,
    min_action_score: float = 0.0,
    min_route_score: float = 0.0,
    min_quality_score: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    selected: List[Dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for source in rows:
        try:
            response = parse_response(source)
            judged = source["collaboration_judge_v3"]
            state = source["deterministic_state_v3"]
            if not isinstance(judged, dict) or not isinstance(state, dict):
                raise TypeError("missing v3 dictionaries")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            rejected["invalid_v3_record"] += 1
            continue
        reason = _reject_reason(
            source,
            response,
            judged,
            state,
            min_reasoning_score=min_reasoning_score,
            min_verification_score=min_verification_score,
            min_action_score=min_action_score,
            min_route_score=min_route_score,
            min_quality_score=min_quality_score,
        )
        if reason:
            rejected[reason] += 1
            continue
        turn = int(source["turn"])
        score = quality_score(turn, judged)
        weight = 0.4 + 0.6 * max(0.0, min(1.0, (score - 0.2) / 0.8))
        output = dict(source)
        route = route_name(response)
        answer_state = str(state["answer_state"])
        stratum = "|".join((
            str(source["agent_id"]),
            turn_bucket(turn),
            str(response["action"]),
            route,
            answer_state,
        ))
        output["response"] = json.dumps(response, ensure_ascii=False)
        output["train_weight"] = round(weight, 6)
        if "reward" in output:
            output["source_reward_v1"] = output["reward"]
        output["reward"] = round(weight, 6)
        output["sample_source_v3"] = "judged_positive"
        output["quality_score_v3"] = round(score, 6)
        output["route_v3"] = route
        output["sampling_stratum_v3"] = stratum
        selected.append(output)
    return selected, rejected


def _sample_cap(
    rows: List[Dict[str, Any]],
    cap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(rows) <= cap:
        return list(rows)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:cap]


def balance_collaboration_routes(
    rows: Sequence[Dict[str, Any]],
    *,
    route_cap_multiplier: float = 1.5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    dropped: Counter[str] = Counter()
    a1_first: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    remaining: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("agent_id") == "A1" and int(row.get("turn", -1)) == 0:
            a1_first[str(row.get("route_v3"))].append(row)
        else:
            remaining.append(row)

    balanced: List[Dict[str, Any]] = []
    if a1_first:
        if set(a1_first) != {"A2", "A3"}:
            raise ValueError(f"A1 first-turn routes are not A2/A3: {sorted(a1_first)}")
        target_count = min(len(a1_first["A2"]), len(a1_first["A3"]))
        for route in ("A2", "A3"):
            kept = _sample_cap(a1_first[route], target_count, rng)
            balanced.extend(kept)
            dropped[f"A1_turn0_route_{route}"] += len(a1_first[route]) - len(kept)

    base_groups: Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in remaining:
        response = parse_response(row)
        state = row["deterministic_state_v3"]
        base = (
            str(row["agent_id"]),
            turn_bucket(int(row["turn"])),
            str(response["action"]),
            str(state["answer_state"]),
        )
        base_groups[base][str(row["route_v3"])].append(row)

    for base, route_groups in sorted(base_groups.items()):
        nonempty_counts = [len(group) for group in route_groups.values() if group]
        if len(nonempty_counts) <= 1:
            cap = nonempty_counts[0]
        else:
            cap = max(1, math.ceil(min(nonempty_counts) * route_cap_multiplier))
        for route, group in sorted(route_groups.items()):
            kept = _sample_cap(group, cap, rng)
            balanced.extend(kept)
            dropped["|".join((*base, route))] += len(group) - len(kept)

    rng.shuffle(balanced)
    return balanced, dropped


def replay_stratum(row: Dict[str, Any], response: Dict[str, Any]) -> str:
    return "|".join((
        turn_bucket(int(row.get("turn", 0))),
        str(response.get("action") or ""),
        route_name(response),
        str(row.get("sample_type") or "unknown"),
        str(row.get("error_type") or "none"),
    ))


def convert_sft_replay(row: Dict[str, Any], weight: float) -> Dict[str, Any]:
    response = parse_response(row, "label")
    if response.get("action") not in {"handoff", "confirm_stop"}:
        raise ValueError("SFT replay has invalid action")
    agent = str(row.get("agent_id") or "")
    if agent not in AGENT_IDS:
        raise ValueError("SFT replay has invalid agent")
    stratum = replay_stratum(row, response)
    return {
        "problem_id": str(row.get("problem_id") or ""),
        "rollout_idx": int(row.get("rollout_idx", -1)),
        "turn": int(row.get("turn", 0)),
        "agent_id": agent,
        "messages": row["messages"],
        "response": json.dumps(response, ensure_ascii=False),
        "action": response["action"],
        "train_weight": round(weight, 6),
        "reward": round(weight, 6),
        "sample_source_v3": "sft_protocol_replay",
        "quality_score_v3": 1.0,
        "route_v3": route_name(response),
        "sampling_stratum_v3": f"replay|{agent}|{stratum}",
        "sft_replay_metadata_v3": {
            "sample_type": row.get("sample_type"),
            "error_type": row.get("error_type"),
            "route_signature": row.get("route_signature"),
        },
    }


def stratified_sample(
    rows: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        response = parse_response(row, "label")
        groups[replay_stratum(row, response)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: List[Dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


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
                parsed = parse_response(row, "label")
                if int(row.get("turn", -1)) == 0:
                    by_route[route_name(parsed)].append(row)
            per_route = desired // 2
            sampled = []
            for route_index, route in enumerate(("A2", "A3")):
                sampled.extend(stratified_sample(
                    by_route[route],
                    per_route,
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
        replay.extend(convert_sft_replay(row, replay_weight) for row in sampled)
    return replay


def _counter(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field)) for row in rows))


def summarize(
    source_rows: Sequence[Dict[str, Any]],
    selected_before_balance: Sequence[Dict[str, Any]],
    selected_rl: Sequence[Dict[str, Any]],
    replay_rows: Sequence[Dict[str, Any]],
    train_rows: Sequence[Dict[str, Any]],
    holdout_rows: Sequence[Dict[str, Any]],
    rejected: Counter[str],
    balance_dropped: Counter[str],
    holdout_ids: set[str],
) -> Dict[str, Any]:
    weights = [float(row["train_weight"]) for row in train_rows]
    return {
        "source_rows": len(source_rows),
        "positive_rows_before_balance": len(selected_before_balance),
        "judged_rows_after_balance": len(selected_rl),
        "sft_replay_rows": len(replay_rows),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_problem_ids": sorted(holdout_ids),
        "actual_sft_replay_ratio": (
            len(replay_rows) / len(train_rows) if train_rows else 0.0
        ),
        "train_agents": _counter(train_rows, "agent_id"),
        "train_sources": _counter(train_rows, "sample_source_v3"),
        "train_actions": _counter(train_rows, "action"),
        "train_routes": _counter(train_rows, "route_v3"),
        "train_strata": _counter(train_rows, "sampling_stratum_v3"),
        "holdout_agents": _counter(holdout_rows, "agent_id"),
        "rejected": dict(rejected),
        "balance_dropped": {
            key: value for key, value in balance_dropped.items() if value
        },
        "train_weight": {
            "min": min(weights) if weights else None,
            "mean": statistics.fmean(weights) if weights else None,
            "max": max(weights) if weights else None,
        },
        "training_contract": {
            "signed_negative_examples": 0,
            "all_weights_non_negative": all(weight >= 0 for weight in weights),
            "fixed_start_agent": "A1",
            "a1_first_handoff_targets_balanced": True,
            "correction_requires_later_verification": True,
            "all_agents_trained": True,
        },
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_data(
    source_rows: Sequence[Dict[str, Any]],
    sft_rows_by_agent: Dict[str, Sequence[Dict[str, Any]]],
    *,
    holdout_fraction: float = 0.1,
    replay_ratio: float = 0.3,
    replay_weight: float = 1.0,
    route_cap_multiplier: float = 1.5,
    seed: int = 42,
    min_reasoning_score: float = 0.2,
    min_verification_score: float = 0.2,
    min_action_score: float = 0.0,
    min_route_score: float = 0.0,
    min_quality_score: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    holdout_ids = deterministic_holdout_ids(
        (str(row.get("problem_id") or "") for row in source_rows),
        holdout_fraction,
        seed,
    )
    train_source = [
        row for row in source_rows if str(row.get("problem_id") or "") not in holdout_ids
    ]
    holdout_source = [
        row for row in source_rows if str(row.get("problem_id") or "") in holdout_ids
    ]
    selected_before_balance, rejected_train = select_positive_candidates(
        train_source,
        min_reasoning_score=min_reasoning_score,
        min_verification_score=min_verification_score,
        min_action_score=min_action_score,
        min_route_score=min_route_score,
        min_quality_score=min_quality_score,
    )
    selected_rl, balance_dropped = balance_collaboration_routes(
        selected_before_balance,
        route_cap_multiplier=route_cap_multiplier,
        seed=seed,
    )
    holdout_rows, rejected_holdout = select_positive_candidates(
        holdout_source,
        min_reasoning_score=min_reasoning_score,
        min_verification_score=min_verification_score,
        min_action_score=min_action_score,
        min_route_score=min_route_score,
        min_quality_score=min_quality_score,
    )
    replay_rows = build_replay_rows(
        selected_rl,
        sft_rows_by_agent,
        holdout_ids,
        replay_ratio=replay_ratio,
        replay_weight=replay_weight,
        seed=seed,
    )
    train_rows = list(selected_rl) + replay_rows
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(holdout_rows)
    rejected = rejected_train + Counter({
        f"holdout:{key}": value for key, value in rejected_holdout.items()
    })
    stats = summarize(
        source_rows,
        selected_before_balance,
        selected_rl,
        replay_rows,
        train_rows,
        holdout_rows,
        rejected,
        balance_dropped,
        holdout_ids,
    )
    return train_rows, holdout_rows, stats


def main() -> None:
    args = parse_args()
    if not 0 <= args.holdout_fraction < 1:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    if not 0 < args.sft_replay_ratio < 1:
        raise SystemExit("--sft-replay-ratio must be in (0, 1)")
    if args.sft_replay_weight < 0:
        raise SystemExit("--sft-replay-weight must be non-negative")
    if args.route_cap_multiplier < 1:
        raise SystemExit("--route-cap-multiplier must be >= 1")
    for path in (args.train_output, args.holdout_output, args.stats_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {path}; use --overwrite or a new path")

    source_rows = read_jsonl(args.input)
    sft_rows_by_agent = {
        agent: read_jsonl(args.sft_data_dir / f"{agent}.jsonl")
        for agent in AGENT_IDS
    }
    train_rows, holdout_rows, stats = build_data(
        source_rows,
        sft_rows_by_agent,
        holdout_fraction=args.holdout_fraction,
        replay_ratio=args.sft_replay_ratio,
        replay_weight=args.sft_replay_weight,
        route_cap_multiplier=args.route_cap_multiplier,
        seed=args.seed,
        min_reasoning_score=args.min_reasoning_score,
        min_verification_score=args.min_verification_score,
        min_action_score=args.min_action_score,
        min_route_score=args.min_route_score,
        min_quality_score=args.min_quality_score,
    )
    agent_counts = Counter(str(row["agent_id"]) for row in train_rows)
    for agent in AGENT_IDS:
        if agent_counts[agent] < args.min_train_rows_per_agent:
            raise SystemExit(
                f"too few v3 training rows for {agent}: {agent_counts[agent]} "
                f"< {args.min_train_rows_per_agent}"
            )
    a1_routes = Counter(
        str(row["route_v3"])
        for row in train_rows
        if row.get("agent_id") == "A1"
        and int(row.get("turn", -1)) == 0
    )
    if a1_routes.get("A2", 0) != a1_routes.get("A3", 0):
        raise SystemExit(f"A1 first handoff is not balanced: {dict(a1_routes)}")
    if any(float(row["train_weight"]) < 0 for row in train_rows):
        raise SystemExit("negative train_weight found; v3 forbids signed suppression")

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.holdout_output, holdout_rows)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_stats = args.stats_output.with_suffix(args.stats_output.suffix + ".tmp")
    temporary_stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_stats.replace(args.stats_output)
    print("GSM judge RL v3 data gate: PASS")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
