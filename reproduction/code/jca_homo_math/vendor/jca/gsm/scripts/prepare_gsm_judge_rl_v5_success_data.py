#!/usr/bin/env python3
"""Build the v5-success signed-RWR data from fixed-start GSM rollouts.

The trajectory mix is selected before row-level filtering so the experiment
controls the useful unit (a complete A1-start rollout), while each retained
turn is trained independently by the plain signed-RWR trainer.  Deterministic
step correctness fixes the reward sign; the GPT judge only supplies magnitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from jca.gsm.scripts.prepare_gsm_judge_rl_v3_data import (
    AGENT_IDS,
    deterministic_holdout_ids,
    read_jsonl,
    write_jsonl,
)
from jca.gsm.scripts.prepare_gsm_judge_rl_v4_data import (
    all_failed_problem_ids,
    validate_fixed_a1_source,
    validate_source_row,
)


TRAJECTORY_CLASSES = (
    "a1_correct",
    "no_correction",
    "correction_success",
    "correction_failed",
)
PROVENANCE_FIELDS = (
    "data_origin",
    "generation_source",
    "label_source",
    "reward_source",
    "sample_source",
    "sample_source_v4",
    "source",
)

_HISTORY_AGENT_RE = re.compile(r"^\s*\[(A[123])\]\s*$", re.MULTILINE)
_HISTORY_HANDOFF_RE = re.compile(
    r"^\s*(?:\u2192|->)\s*handoff\s+to\s+(A[123])(?:\s*:|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _response_payload(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    response = row.get("response")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    return response if isinstance(response, Mapping) else None


def _assistant_message_has_self_handoff(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    agent_match = _HISTORY_AGENT_RE.search(content)
    handoff_match = _HISTORY_HANDOFF_RE.search(content)
    return bool(
        agent_match
        and handoff_match
        and agent_match.group(1).upper() == handoff_match.group(1).upper()
    )


def self_handoff_location(row: Mapping[str, Any]) -> Optional[str]:
    """Identify a self-handoff in the supervised response or its history."""
    agent_id = str(row.get("agent_id") or "")
    response = _response_payload(row)
    if (
        response is not None
        and response.get("action") == "handoff"
        and response.get("handoff_target") == agent_id
    ):
        return "response"
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if _assistant_message_has_self_handoff(message.get("content")):
                return "history"
    return None


def truncate_trajectories_at_self_handoff(
    groups: Mapping[Tuple[str, int], Sequence[Dict[str, Any]]],
) -> Tuple[Dict[Tuple[str, int], List[Dict[str, Any]]], Dict[str, Any]]:
    """Keep only the valid prefix before each trajectory's first self-handoff."""
    cleaned: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    locations: Counter[str] = Counter()
    first_turns: Counter[str] = Counter()
    rows_removed = 0
    trajectories_truncated = 0
    trajectories_removed = 0

    for key, source_group in groups.items():
        group = list(source_group)
        cutoff = len(group)
        location: Optional[str] = None
        for index, row in enumerate(group):
            detected = self_handoff_location(row)
            if detected is not None:
                cutoff = index
                location = detected
                break
        if location is not None:
            trajectories_truncated += 1
            locations[location] += 1
            first_turns[str(int(group[cutoff].get("turn", cutoff)))] += 1
            rows_removed += len(group) - cutoff
        prefix = group[:cutoff]
        if prefix:
            cleaned[key] = prefix
        else:
            trajectories_removed += 1

    return cleaned, {
        "policy": "keep_prefix_before_first_self_handoff",
        "trajectories_truncated": trajectories_truncated,
        "trajectories_removed": trajectories_removed,
        "rows_removed": rows_removed,
        "detections_by_location": dict(locations),
        "first_self_handoff_turns": dict(sorted(first_turns.items(), key=lambda item: int(item[0]))),
        "remaining_trajectories": len(cleaned),
        "remaining_rows": sum(len(group) for group in cleaned.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v5-success, teacher-free GSM signed-RWR data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--expected-rollouts-per-problem", type=int, default=8)
    parser.add_argument(
        "--expected-start-agent",
        choices=AGENT_IDS,
        default="A1",
        help="Required first role for every source trajectory.",
    )
    parser.add_argument("--drop-all-failed-problems", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--a1-correct-ratio", type=float, default=0.30)
    parser.add_argument("--no-correction-ratio", type=float, default=0.20)
    parser.add_argument("--correction-success-ratio", type=float, default=0.35)
    parser.add_argument("--correction-failed-ratio", type=float, default=0.15)
    parser.add_argument(
        "--max-train-trajectories", type=int, default=0,
        help="Maximum selected training trajectories; 0 uses the largest exact mix.",
    )
    parser.add_argument(
        "--max-holdout-trajectories", type=int, default=0,
        help="Maximum selected holdout trajectories; 0 uses the largest exact mix.",
    )
    parser.add_argument("--base-reward-weight", type=float, default=0.35)
    parser.add_argument("--judge-reward-weight", type=float, default=0.65)
    parser.add_argument("--transition-boost", type=float, default=1.25)
    parser.add_argument("--min-aligned-judge", type=float, default=0.0)
    parser.add_argument(
        "--correct-to-correct-coef", type=float, default=1.0,
        help="Causal-credit multiplier applied only to correct_to_correct rows.",
    )
    parser.add_argument(
        "--correct-to-correct-coef-a1", type=float, default=None,
        help="Optional A1-only override for correct_to_correct credit.",
    )
    parser.add_argument(
        "--correct-to-correct-coef-a2", type=float, default=None,
        help="Optional A2-only override for correct_to_correct credit.",
    )
    parser.add_argument(
        "--correct-to-correct-coef-a3", type=float, default=None,
        help="Optional A3-only override for correct_to_correct credit.",
    )
    parser.add_argument(
        "--correct-to-correct-handoff-coef", type=float, default=1.0,
        help="Additional credit multiplier for correct_to_correct handoffs.",
    )
    parser.add_argument(
        "--wrong-to-correct-multiplier", type=float, default=None,
        help=(
            "Transition multiplier for wrong_to_correct; omitted falls back "
            "to --transition-boost for backward compatibility."
        ),
    )
    parser.add_argument(
        "--correct-to-wrong-multiplier", type=float, default=None,
        help=(
            "Transition multiplier for correct_to_wrong; omitted falls back "
            "to --transition-boost for backward compatibility."
        ),
    )
    parser.add_argument(
        "--drop-wrong-to-wrong-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop wrong_to_wrong rows whose action is confirm_stop.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _stable_key(rows: Sequence[Dict[str, Any]], seed: int) -> str:
    first = rows[0]
    identity = f"{seed}:{first.get('problem_id')}:{first.get('rollout_idx')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def group_trajectories(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        problem_id = str(row.get("problem_id") or "")
        rollout_idx = int(row.get("rollout_idx", -1))
        if not problem_id or rollout_idx < 0:
            raise ValueError("invalid problem_id/rollout_idx")
        grouped[(problem_id, rollout_idx)].append(row)
    for key, group in grouped.items():
        group.sort(key=lambda row: int(row.get("turn", -1)))
        turns = [int(row.get("turn", -1)) for row in group]
        if turns != list(range(len(group))):
            raise ValueError(f"trajectory has non-contiguous turns: {key}: {turns}")
    return dict(grouped)


def classify_trajectory(rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Classify one complete fixed-A1 rollout and expose anomaly counters."""
    if not rows:
        raise ValueError("empty trajectory")
    first = rows[0]
    first_state = first["deterministic_state_v3"]
    initial_correct = bool(first_state["current_answer_correct"])
    terminal_values = {float(row["em"]) for row in rows}
    if terminal_values != {0.0} and terminal_values != {1.0}:
        raise ValueError("terminal EM must be binary and consistent within trajectory")
    terminal_em = int(next(iter(terminal_values)))
    changed_after_start = any(
        bool(row["deterministic_state_v3"].get("answer_changed"))
        for row in rows[1:]
    )
    if initial_correct:
        kind = "a1_correct"
    elif changed_after_start:
        kind = "correction_success" if terminal_em else "correction_failed"
    else:
        kind = "no_correction"
    anomalies = {
        "initial_wrong_final_correct_without_change": int(
            not initial_correct and terminal_em == 1 and not changed_after_start
        ),
        "initial_correct_final_wrong": int(initial_correct and terminal_em == 0),
        "rows": len(rows),
        "terminal_em": terminal_em,
        "initial_correct": initial_correct,
        "changed_after_start": changed_after_start,
    }
    return kind, anomalies


def validate_mix_ratios(ratios: Dict[str, float]) -> None:
    if any(value < 0 for value in ratios.values()):
        raise ValueError("trajectory ratios must be non-negative")
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"trajectory ratios must sum to 1, got {total}")
    if not any(ratios.values()):
        raise ValueError("at least one trajectory ratio must be positive")


def select_exact_mix(
    groups_by_class: Dict[str, List[Tuple[Tuple[str, int], List[Dict[str, Any]]]]],
    ratios: Dict[str, float],
    *,
    max_trajectories: int,
    seed: int,
) -> Tuple[List[Tuple[Tuple[str, int], List[Dict[str, Any]]]], Dict[str, int]]:
    """Select the largest feasible integer mix without replacement.

    For a requested total N, each class gets floor(N * ratio), with leftover
    slots assigned by largest fractional remainder.  N is reduced until every
    requested class has enough trajectories, so no class is silently replaced.
    """
    available = {kind: len(groups_by_class.get(kind, [])) for kind in TRAJECTORY_CLASSES}
    positive = [kind for kind in TRAJECTORY_CLASSES if ratios[kind] > 0]
    if not positive:
        raise ValueError("no positive trajectory ratios")
    upper = min(
        available[kind] / ratios[kind]
        for kind in positive
    )
    total = int(math.floor(upper + 1e-9))
    if max_trajectories > 0:
        total = min(total, max_trajectories)
    while total > 0:
        counts = {kind: int(math.floor(total * ratios[kind] + 1e-9)) for kind in TRAJECTORY_CLASSES}
        remainder = total - sum(counts.values())
        ranked = sorted(
            TRAJECTORY_CLASSES,
            key=lambda kind: (-(total * ratios[kind] - counts[kind]), kind),
        )
        for kind in ranked:
            if remainder <= 0:
                break
            if ratios[kind] > 0:
                counts[kind] += 1
                remainder -= 1
        if all(counts[kind] <= available[kind] for kind in TRAJECTORY_CLASSES):
            break
        total -= 1
    if total <= 0:
        raise ValueError(f"trajectory mix is infeasible: available={available}")
    selected: List[Tuple[Tuple[str, int], List[Dict[str, Any]]]] = []
    for index, kind in enumerate(TRAJECTORY_CLASSES):
        pool = sorted(
            groups_by_class.get(kind, []),
            key=lambda item: _stable_key(item[1], seed + 1009 * index),
        )
        selected.extend(pool[:counts[kind]])
    random.Random(seed).shuffle(selected)
    return selected, counts


def _aligned_judge(row: Dict[str, Any]) -> float:
    state = row["deterministic_state_v3"]
    judge = row["collaboration_judge_v3"]
    sign = 1.0 if bool(state["current_answer_correct"]) else -1.0
    score = float(judge.get("judge_score", 0.0))
    if not math.isfinite(score):
        raise ValueError("non-finite judge score")
    return sign * max(-1.0, min(1.0, score))


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _effective_c2c_coef(
    agent_id: str,
    global_coef: float,
    per_agent: Optional[Mapping[str, Optional[float]]],
) -> float:
    """Resolve an optional role-specific c2c override without changing defaults."""

    _validate_nonnegative_finite(global_coef, "correct_to_correct_coef")
    override = (per_agent or {}).get(str(agent_id))
    if override is None:
        return global_coef
    _validate_nonnegative_finite(float(override), f"correct_to_correct_coef_{agent_id.lower()}")
    return float(override)


def _transition_multiplier(
    transition: str,
    legacy_boost: float,
    wrong_to_correct: Optional[float],
    correct_to_wrong: Optional[float],
) -> float:
    """Resolve asymmetric transition credit while preserving legacy behavior."""

    _validate_nonnegative_finite(legacy_boost, "transition_boost")
    if wrong_to_correct is not None:
        _validate_nonnegative_finite(float(wrong_to_correct), "wrong_to_correct_multiplier")
    if correct_to_wrong is not None:
        _validate_nonnegative_finite(float(correct_to_wrong), "correct_to_wrong_multiplier")
    if transition == "wrong_to_correct":
        return float(legacy_boost if wrong_to_correct is None else wrong_to_correct)
    if transition == "correct_to_wrong":
        return float(legacy_boost if correct_to_wrong is None else correct_to_wrong)
    return 1.0


def reward_row(
    source: Dict[str, Any],
    trajectory_class: str,
    *,
    base_reward_weight: float,
    judge_reward_weight: float,
    transition_boost: float,
    min_aligned_judge: float,
    correct_to_correct_coef: float = 1.0,
    drop_wrong_to_wrong_stop: bool = False,
    correct_to_correct_coef_by_agent: Optional[Mapping[str, Optional[float]]] = None,
    correct_to_correct_handoff_coef: float = 1.0,
    wrong_to_correct_multiplier: Optional[float] = None,
    correct_to_wrong_multiplier: Optional[float] = None,
    expected_start_agent: str = "A1",
) -> Tuple[Dict[str, Any] | None, str | None]:
    """Return a teacher-free signed-RWR row, or a pollution rejection reason."""
    self_handoff = self_handoff_location(source)
    if self_handoff == "response":
        return None, "self_handoff_filtered"
    if self_handoff == "history":
        return None, "historical_self_handoff_filtered"
    response, _judged, state = validate_source_row(
        source, expected_start_agent=expected_start_agent,
    )
    agent_id = str(source.get("agent_id") or "")
    if not math.isfinite(correct_to_correct_coef) or correct_to_correct_coef < 0:
        raise ValueError("correct_to_correct_coef must be finite and non-negative")
    _validate_nonnegative_finite(
        correct_to_correct_handoff_coef, "correct_to_correct_handoff_coef",
    )
    transition = str(state.get("answer_state") or "")
    if (
        drop_wrong_to_wrong_stop
        and transition == "wrong_to_wrong"
        and response.get("action") == "confirm_stop"
    ):
        return None, "wrong_to_wrong_confirm_stop_filtered"
    aligned = _aligned_judge(source)
    if aligned < min_aligned_judge:
        return None, "gpt_correctness_conflict"
    if not math.isclose(base_reward_weight + judge_reward_weight, 1.0, abs_tol=1e-6):
        raise ValueError("base and judge reward weights must sum to 1")
    sign = 1.0 if bool(state["current_answer_correct"]) else -1.0
    magnitude = base_reward_weight + judge_reward_weight * aligned
    magnitude = max(base_reward_weight, min(1.0, magnitude))
    transition_multiplier = _transition_multiplier(
        transition,
        transition_boost,
        wrong_to_correct_multiplier,
        correct_to_wrong_multiplier,
    )
    transition_magnitude = magnitude * transition_multiplier
    transition_capped = transition_magnitude > 1.0
    magnitude = min(1.0, transition_magnitude)
    agent_c2c_coef = _effective_c2c_coef(
        agent_id,
        correct_to_correct_coef,
        correct_to_correct_coef_by_agent,
    )
    causal_credit_coef = agent_c2c_coef if transition == "correct_to_correct" else 1.0
    action = str(response.get("action") or "")
    handoff_credit_coef = (
        correct_to_correct_handoff_coef
        if transition == "correct_to_correct" and action == "handoff"
        else 1.0
    )
    effective_c2c_coef = causal_credit_coef * handoff_credit_coef
    magnitude *= effective_c2c_coef
    reward = sign * magnitude
    # Keep the public reward in [-1, 1] for validators.  For an explicitly
    # asymmetric transition multiplier, expose the uncapped signed weight to
    # RWR as train_weight so a saturated -1 reward can still receive stronger
    # suppression.  With legacy/default flags this is exactly the old value.
    train_magnitude = (
        transition_magnitude * effective_c2c_coef
        if (
            (transition == "wrong_to_correct" and wrong_to_correct_multiplier is not None)
            or (transition == "correct_to_wrong" and correct_to_wrong_multiplier is not None)
        )
        else magnitude
    )
    train_weight = sign * train_magnitude
    output = dict(source)
    output["response"] = json.dumps(response, ensure_ascii=False)
    output["reward"] = round(reward, 6)
    output["train_weight"] = round(train_weight, 6)
    output["sample_source"] = "gsm_judge_rl_v5_success_signed_rwr"
    output["trajectory_class"] = trajectory_class
    output["aligned_judge"] = round(aligned, 6)
    output["reward_magnitude"] = round(magnitude, 6)
    output["reward_sign"] = int(sign)
    output["reward_base_weight"] = base_reward_weight
    output["reward_judge_weight"] = judge_reward_weight
    output["transition_boost"] = transition_boost
    output["effective_transition_multiplier"] = transition_multiplier
    output["transition_reward_capped"] = transition_capped
    output["effective_c2c_agent_coef"] = agent_c2c_coef
    output["effective_c2c_handoff_coef"] = handoff_credit_coef
    output["causal_credit_coef"] = causal_credit_coef
    output["loss_weight_unclipped"] = bool(train_weight != reward)
    output["loss_mask_mode"] = "full_response"
    output["teacher_provenance"] = False
    output["route"] = (
        "stop" if response.get("action") == "confirm_stop"
        else str(response.get("handoff_target") or "missing")
    )
    return output, None


def _has_teacher_provenance(row: Dict[str, Any]) -> bool:
    for field in PROVENANCE_FIELDS:
        value = row.get(field)
        if value is not None and "teacher" in str(value).lower():
            return True
    return False


def _group_stats(
    selected: Sequence[Tuple[Tuple[str, int], List[Dict[str, Any]]]],
    rows: Sequence[Dict[str, Any]],
    *,
    rejected: Counter[str] | None = None,
) -> Dict[str, Any]:
    trajectories = Counter()
    for _, group in selected:
        kind, _ = classify_trajectory(group)
        trajectories[kind] += 1
    output: Dict[str, Any] = {
        "trajectories": dict(trajectories),
        "rows": len(rows),
        "rows_by_agent": dict(Counter(str(row.get("agent_id")) for row in rows)),
        "rows_by_class": dict(Counter(str(row.get("trajectory_class")) for row in rows)),
        "reward_signs_by_agent": {},
        "reward_mass_by_agent": {},
        "signed_reward_sum_by_agent": {},
        "reward_mean_by_agent": {},
        "reward_by_agent_transition_action": {},
        "gpt_conflict_rows": int((rejected or {}).get("gpt_correctness_conflict", 0)),
        "filtered_wrong_to_wrong_confirm_stop": int(
            (rejected or {}).get("wrong_to_wrong_confirm_stop_filtered", 0)
        ),
        "filtered_self_handoff": int(
            (rejected or {}).get("self_handoff_filtered", 0)
        ),
        "filtered_historical_self_handoff": int(
            (rejected or {}).get("historical_self_handoff_filtered", 0)
        ),
        "wrong_to_wrong_confirm_stop": 0,
        "a1_late_adoption_rows": 0,
    }
    wrong_trajectories = (
        trajectories["no_correction"]
        + trajectories["correction_success"]
        + trajectories["correction_failed"]
    )
    correction_trajectories = (
        trajectories["correction_success"] + trajectories["correction_failed"]
    )
    output["correction_metrics"] = {
        "initial_wrong_trajectories": wrong_trajectories,
        "correction_trajectories": correction_trajectories,
        "correction_coverage": (
            correction_trajectories / wrong_trajectories
            if wrong_trajectories else None
        ),
        "correction_final_em": (
            trajectories["correction_success"] / correction_trajectories
            if correction_trajectories else None
        ),
    }
    for agent in AGENT_IDS:
        agent_rows = [row for row in rows if str(row.get("agent_id")) == agent]
        output["reward_signs_by_agent"][agent] = dict(Counter(
            "positive" if float(row["reward"]) > 0
            else ("negative" if float(row["reward"]) < 0 else "zero")
            for row in agent_rows
        ))
        output["reward_mass_by_agent"][agent] = round(
            sum(abs(float(row["reward"])) for row in agent_rows), 6
        )
        signed_reward_sum = sum(float(row["reward"]) for row in agent_rows)
        output["signed_reward_sum_by_agent"][agent] = round(signed_reward_sum, 6)
        output["reward_mean_by_agent"][agent] = (
            round(signed_reward_sum / len(agent_rows), 6) if agent_rows else None
        )
    transition_buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        state = row.get("deterministic_state_v3") or {}
        transition = str(state.get("answer_state") or "")
        try:
            parsed_response = json.loads(row.get("response", "{}"))
        except (TypeError, json.JSONDecodeError):
            parsed_response = {}
        action = str(parsed_response.get("action") or row.get("action") or "")
        key = (str(row.get("agent_id") or ""), transition, action)
        bucket = transition_buckets.setdefault(
            key,
            {
                "rows": 0,
                "absolute_reward_mass": 0.0,
                "signed_reward_sum": 0.0,
                "absolute_train_weight_mass": 0.0,
                "signed_train_weight_sum": 0.0,
                "transition_capped_rows": 0,
            },
        )
        reward = float(row.get("reward", 0.0))
        train_weight = float(row.get("train_weight", reward))
        bucket["rows"] += 1
        bucket["absolute_reward_mass"] += abs(reward)
        bucket["signed_reward_sum"] += reward
        bucket["absolute_train_weight_mass"] += abs(train_weight)
        bucket["signed_train_weight_sum"] += train_weight
        bucket["transition_capped_rows"] += int(bool(row.get("transition_reward_capped")))
    for (agent, transition, action), bucket in sorted(transition_buckets.items()):
        bucket["absolute_reward_mass"] = round(bucket["absolute_reward_mass"], 6)
        bucket["signed_reward_sum"] = round(bucket["signed_reward_sum"], 6)
        bucket["absolute_train_weight_mass"] = round(
            bucket["absolute_train_weight_mass"], 6,
        )
        bucket["signed_train_weight_sum"] = round(
            bucket["signed_train_weight_sum"], 6,
        )
        bucket["mean_reward"] = round(
            bucket["signed_reward_sum"] / bucket["rows"], 6,
        ) if bucket["rows"] else None
        bucket["mean_train_weight"] = round(
            bucket["signed_train_weight_sum"] / bucket["rows"], 6,
        ) if bucket["rows"] else None
        output["reward_by_agent_transition_action"].setdefault(agent, {}).setdefault(
            transition, {}
        )[action] = bucket
    for row in rows:
        state = row["deterministic_state_v3"]
        if str(state.get("answer_state")) == "wrong_to_wrong" and row.get("route") == "stop":
            output["wrong_to_wrong_confirm_stop"] += 1
        if str(row.get("agent_id")) == "A1" and int(row.get("turn", -1)) > 0 and str(state.get("answer_state")) == "wrong_to_correct":
            output["a1_late_adoption_rows"] += 1
    return output


def build_data(
    source_rows: Sequence[Dict[str, Any]],
    *,
    expected_rollouts_per_problem: Optional[int],
    drop_all_failed_problems: bool,
    holdout_fraction: float,
    ratios: Dict[str, float],
    max_train_trajectories: int,
    max_holdout_trajectories: int,
    base_reward_weight: float,
    judge_reward_weight: float,
    transition_boost: float,
    min_aligned_judge: float,
    seed: int,
    correct_to_correct_coef: float = 1.0,
    drop_wrong_to_wrong_stop: bool = False,
    correct_to_correct_coef_by_agent: Optional[Mapping[str, Optional[float]]] = None,
    correct_to_correct_handoff_coef: float = 1.0,
    wrong_to_correct_multiplier: Optional[float] = None,
    correct_to_wrong_multiplier: Optional[float] = None,
    expected_start_agent: str = "A1",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    validate_mix_ratios(ratios)
    source_groups = group_trajectories(source_rows)
    validate_fixed_a1_source(
        source_rows, expected_start_agent=expected_start_agent,
    )
    all_failed = all_failed_problem_ids(
        source_rows, expected_rollouts_per_problem=expected_rollouts_per_problem,
    )
    clean_source_groups, self_handoff_truncation = (
        truncate_trajectories_at_self_handoff(source_groups)
    )
    eligible = {
        key: group for key, group in clean_source_groups.items()
        if not drop_all_failed_problems or key[0] not in all_failed
    }
    holdout_ids = deterministic_holdout_ids(
        (problem_id for problem_id, _ in eligible), holdout_fraction, seed,
    )
    split_groups = {
        "train": {key: group for key, group in eligible.items() if key[0] not in holdout_ids},
        "holdout": {key: group for key, group in eligible.items() if key[0] in holdout_ids},
    }
    selected: Dict[str, List[Tuple[Tuple[str, int], List[Dict[str, Any]]]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for split, groups in split_groups.items():
        by_class: Dict[str, List[Tuple[Tuple[str, int], List[Dict[str, Any]]]]] = defaultdict(list)
        for key, group in groups.items():
            kind, _ = classify_trajectory(group)
            by_class[kind].append((key, group))
        selected[split], counts[split] = select_exact_mix(
            by_class, ratios,
            max_trajectories=(max_train_trajectories if split == "train" else max_holdout_trajectories),
            seed=seed + (0 if split == "train" else 1),
        )

    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "holdout": []}
    rejection_counts: Dict[str, Counter[str]] = {"train": Counter(), "holdout": Counter()}
    anomaly_counts: Counter[str] = Counter()
    teacher_rows = 0
    for split in ("train", "holdout"):
        for _, group in selected[split]:
            kind, anomalies = classify_trajectory(group)
            for name, value in anomalies.items():
                if name in {
                    "initial_wrong_final_correct_without_change",
                    "initial_correct_final_wrong",
                } and value:
                    anomaly_counts[name] += int(value)
            for source in group:
                if _has_teacher_provenance(source):
                    teacher_rows += 1
                    rejection_counts[split]["teacher_provenance"] += 1
                    continue
                try:
                    row, reason = reward_row(
                        source, kind,
                        base_reward_weight=base_reward_weight,
                        judge_reward_weight=judge_reward_weight,
                        transition_boost=transition_boost,
                        min_aligned_judge=min_aligned_judge,
                        correct_to_correct_coef=correct_to_correct_coef,
                        drop_wrong_to_wrong_stop=drop_wrong_to_wrong_stop,
                        correct_to_correct_coef_by_agent=correct_to_correct_coef_by_agent,
                        correct_to_correct_handoff_coef=correct_to_correct_handoff_coef,
                        wrong_to_correct_multiplier=wrong_to_correct_multiplier,
                        correct_to_wrong_multiplier=correct_to_wrong_multiplier,
                        expected_start_agent=expected_start_agent,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    rejection_counts[split][str(exc) or type(exc).__name__] += 1
                    continue
                if row is None:
                    rejection_counts[split][reason or "rejected"] += 1
                    continue
                outputs[split].append(row)
        random.Random(seed + (0 if split == "train" else 1)).shuffle(outputs[split])

    stats = {
        "version": "gsm_judge_rl_v5_success",
        "source_rows": len(source_rows),
        "source_trajectories": len(source_groups),
        "source_problems": len({key[0] for key in source_groups}),
        "self_handoff_truncation": self_handoff_truncation,
        "all_failed_problem_count": len(all_failed),
        "all_failed_source_rows": sum(len(group) for key, group in source_groups.items() if key[0] in all_failed),
        "drop_all_failed_problems": drop_all_failed_problems,
        "eligible_rows": sum(len(group) for group in eligible.values()),
        "eligible_trajectories": len(eligible),
        "holdout_problem_count": len(holdout_ids),
        "holdout_problem_ids": sorted(holdout_ids),
        "target_ratios": ratios,
        "selected_trajectory_counts": counts,
        "eligible_trajectory_classes": dict(Counter(
            classify_trajectory(group)[0] for group in eligible.values()
        )),
        "reward": {
            "base_weight": base_reward_weight,
            "judge_weight": judge_reward_weight,
            "transition_boost": transition_boost,
            "wrong_to_correct_multiplier": wrong_to_correct_multiplier,
            "correct_to_wrong_multiplier": correct_to_wrong_multiplier,
            "min_aligned_judge": min_aligned_judge,
            "correct_to_correct_coef": correct_to_correct_coef,
            "correct_to_correct_coef_by_agent": dict(correct_to_correct_coef_by_agent or {}),
            "effective_correct_to_correct_coef_by_agent": {
                agent: _effective_c2c_coef(
                    agent, correct_to_correct_coef, correct_to_correct_coef_by_agent,
                )
                for agent in AGENT_IDS
            },
            "correct_to_correct_handoff_coef": correct_to_correct_handoff_coef,
            "drop_wrong_to_wrong_stop": drop_wrong_to_wrong_stop,
            "sign_source": "deterministic_state_v3.current_answer_correct",
            "magnitude_source": "base + judge * aligned_judge",
            "public_reward_clip": [-1.0, 1.0],
            "train_weight_field": "train_weight",
            "custom_transition_multiplier_uses_unclipped_train_weight": True,
        },
        "train": _group_stats(
            selected["train"], outputs["train"], rejected=rejection_counts["train"],
        ),
        "holdout": _group_stats(
            selected["holdout"], outputs["holdout"], rejected=rejection_counts["holdout"],
        ),
        "rejections": {split: dict(values) for split, values in rejection_counts.items()},
        "anomalies": dict(anomaly_counts),
        "teacher_provenance_rows": teacher_rows,
        "teacher_provenance_assertion": teacher_rows == 0,
        "objective": {
            "algorithm": "plain_signed_rwr",
            "loss": "-reward * assistant_response_log_probability + kl_to_own_sft",
            "loss_mask": "full_response",
            "same_for_agents": True,
            "sft_replay": False,
            "teacher": False,
            "a3_start": expected_start_agent == "A3",
            "fixed_start_agent": expected_start_agent,
        },
    }
    if teacher_rows:
        raise ValueError(f"teacher provenance found in {teacher_rows} selected source rows")
    if not outputs["train"] or not outputs["holdout"]:
        raise ValueError("data construction produced an empty split")
    return outputs["train"], outputs["holdout"], stats


def main() -> None:
    args = parse_args()
    ratios = {
        "a1_correct": args.a1_correct_ratio,
        "no_correction": args.no_correction_ratio,
        "correction_success": args.correction_success_ratio,
        "correction_failed": args.correction_failed_ratio,
    }
    if args.base_reward_weight < 0 or args.judge_reward_weight < 0:
        raise SystemExit("reward weights must be non-negative")
    if (
        not math.isfinite(args.correct_to_correct_coef)
        or args.correct_to_correct_coef < 0
    ):
        raise SystemExit("correct-to-correct coefficient must be finite and non-negative")
    for name in (
        "correct_to_correct_coef_a1",
        "correct_to_correct_coef_a2",
        "correct_to_correct_coef_a3",
        "correct_to_correct_handoff_coef",
        "wrong_to_correct_multiplier",
        "correct_to_wrong_multiplier",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise SystemExit(f"{name.replace('_', '-')} must be finite and non-negative")
    c2c_by_agent = {
        agent: value
        for agent, value in (
            ("A1", args.correct_to_correct_coef_a1),
            ("A2", args.correct_to_correct_coef_a2),
            ("A3", args.correct_to_correct_coef_a3),
        )
        if value is not None
    }
    source = read_jsonl(args.input)
    train, holdout, stats = build_data(
        source,
        expected_rollouts_per_problem=args.expected_rollouts_per_problem,
        drop_all_failed_problems=args.drop_all_failed_problems,
        holdout_fraction=args.holdout_fraction,
        ratios=ratios,
        max_train_trajectories=args.max_train_trajectories,
        max_holdout_trajectories=args.max_holdout_trajectories,
        base_reward_weight=args.base_reward_weight,
        judge_reward_weight=args.judge_reward_weight,
        transition_boost=args.transition_boost,
        min_aligned_judge=args.min_aligned_judge,
        seed=args.seed,
        correct_to_correct_coef=args.correct_to_correct_coef,
        drop_wrong_to_wrong_stop=args.drop_wrong_to_wrong_stop,
        correct_to_correct_coef_by_agent=c2c_by_agent,
        correct_to_correct_handoff_coef=args.correct_to_correct_handoff_coef,
        wrong_to_correct_multiplier=args.wrong_to_correct_multiplier,
        correct_to_wrong_multiplier=args.correct_to_wrong_multiplier,
        expected_start_agent=args.expected_start_agent,
    )
    outputs = (args.train_output, args.holdout_output, args.stats_output)
    if not args.overwrite and any(path.exists() for path in outputs):
        raise SystemExit("output exists; pass --overwrite or choose a new tag")
    write_jsonl(args.train_output, train)
    write_jsonl(args.holdout_output, holdout)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
