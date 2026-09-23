#!/usr/bin/env python3
"""Create the MATH process-reward ablation from corrected prepared rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ANSWER_STATE_CORRECTNESS = {
    "proposal_correct": (None, True),
    "proposal_wrong": (None, False),
    "correct_to_correct": (True, True),
    "correct_to_wrong": (True, False),
    "wrong_to_correct": (False, True),
    "wrong_to_wrong": (False, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute MATH rewards without the Qwen process/judge score."
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--holdout-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--transition-boost", type=float, default=1.25)
    parser.add_argument("--correct-to-correct-coef-a1", type=float, default=1.0)
    parser.add_argument("--correct-to-correct-coef-a2", type=float, default=0.5)
    parser.add_argument("--correct-to-correct-coef-a3", type=float, default=1.0)
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    row = json.loads(
        line,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(row, dict):
        raise ValueError(f"expected object at {path}:{line_number}")
    return row


def row_identity(row: dict[str, Any]) -> tuple[str, int, int, str]:
    identity = (
        str(row.get("problem_id") or ""),
        int(row.get("rollout_idx", -1)),
        int(row.get("turn", -1)),
        str(row.get("agent_id") or ""),
    )
    if not identity[0] or identity[1] < 0 or identity[2] < 0:
        raise ValueError(f"invalid row identity: {identity}")
    if identity[3] not in {"A1", "A2", "A3"}:
        raise ValueError(f"invalid agent in row identity: {identity}")
    return identity


def expected_reward(row: dict[str, Any], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    state = row.get("deterministic_state_v3")
    if not isinstance(state, dict):
        raise ValueError(f"missing deterministic state at {row_identity(row)}")
    if not isinstance(state.get("current_answer_correct"), bool):
        raise ValueError(f"missing correctness state at {row_identity(row)}")
    transition = str(state.get("answer_state") or "")
    if transition not in ANSWER_STATE_CORRECTNESS:
        raise ValueError(f"invalid answer transition at {row_identity(row)}: {transition!r}")
    previous_correct = state.get("previous_answer_correct")
    if previous_correct is not None and not isinstance(previous_correct, bool):
        raise ValueError(f"invalid previous correctness state at {row_identity(row)}")
    observed_correctness = (previous_correct, state["current_answer_correct"])
    if observed_correctness != ANSWER_STATE_CORRECTNESS[transition]:
        raise ValueError(
            f"answer transition state mismatch at {row_identity(row)}: "
            f"{transition!r} does not match {observed_correctness!r}"
        )
    sign = 1.0 if state["current_answer_correct"] else -1.0
    transition_multiplier = args.transition_boost if transition in {
        "correct_to_wrong",
        "wrong_to_correct",
    } else 1.0
    transition_magnitude = min(1.0, transition_multiplier)
    agent = str(row["agent_id"])
    c2c_coefficients = {
        "A1": args.correct_to_correct_coef_a1,
        "A2": args.correct_to_correct_coef_a2,
        "A3": args.correct_to_correct_coef_a3,
    }
    c2c_coef = c2c_coefficients[agent] if transition == "correct_to_correct" else 1.0
    magnitude = transition_magnitude * c2c_coef
    reward = round(sign * magnitude, 6)
    details = {
        "reward": reward,
        "train_weight": reward,
        "aligned_judge": -1.0 if sign < 0 else 1.0,
        "reward_magnitude": round(abs(reward), 6),
        "reward_sign": int(sign),
        "reward_base_weight": 1.0,
        "reward_judge_weight": 0.0,
        "transition_boost": args.transition_boost,
        "effective_transition_multiplier": transition_multiplier,
        "transition_reward_capped": transition_magnitude < transition_multiplier,
        "effective_c2c_agent_coef": c2c_coef,
        "effective_c2c_handoff_coef": 1.0,
        "causal_credit_coef": c2c_coef if transition == "correct_to_correct" else 1.0,
        "loss_weight_unclipped": False,
    }
    return reward, details


def transformed_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output = dict(row)
    reward, details = expected_reward(row, args)
    output.update(details)
    output["reward"] = reward
    output["train_weight"] = reward
    output["sample_source"] = "math_v13_nonthinking_wo_process_reward"
    output["reward_source"] = "math_exact_state_only_no_process_reward"
    output["ablation"] = "wo_process_reward"
    return output


def transform_file(source: Path, destination: Path, args: argparse.Namespace) -> tuple[int, str, Counter[str]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    count = 0
    output_digest = hashlib.sha256()
    agents: Counter[str] = Counter()
    seen: set[tuple[str, int, int, str]] = set()
    try:
        with source.open(encoding="utf-8") as source_handle, temporary.open("w", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                row = read_row(line, source, line_number)
                identity = row_identity(row)
                if identity in seen:
                    raise ValueError(f"duplicate row identity: {identity}")
                seen.add(identity)
                converted = transformed_row(row, args)
                encoded = (json.dumps(converted, ensure_ascii=False) + "\n").encode("utf-8")
                output_handle.buffer.write(encoded)
                output_digest.update(encoded)
                count += 1
                agents[identity[3]] += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count, output_digest.hexdigest(), agents


def validate_output(path: Path, args: argparse.Namespace) -> tuple[int, Counter[str]]:
    count = 0
    agents: Counter[str] = Counter()
    seen: set[tuple[str, int, int, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = read_row(line, path, line_number)
            identity = row_identity(row)
            if identity in seen:
                raise ValueError(f"duplicate output identity: {identity}")
            seen.add(identity)
            expected, details = expected_reward(row, args)
            if float(row.get("reward")) != expected or float(row.get("train_weight")) != expected:
                raise ValueError(f"reward mismatch at {identity}")
            if float(row.get("reward_base_weight")) != 1.0 or float(row.get("reward_judge_weight")) != 0.0:
                raise ValueError(f"reward weights mismatch at {identity}")
            for key, value in details.items():
                if key in {"reward", "train_weight"}:
                    continue
                if row.get(key) != value:
                    raise ValueError(f"{key} mismatch at {identity}")
            if row.get("ablation") != "wo_process_reward":
                raise ValueError(f"ablation marker missing at {identity}")
            count += 1
            agents[identity[3]] += 1
    if count == 0:
        raise ValueError(f"empty output: {path}")
    return count, agents


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.transition_boost) or args.transition_boost < 0:
        raise SystemExit("transition boost must be finite and non-negative")
    for value in (args.correct_to_correct_coef_a1, args.correct_to_correct_coef_a2, args.correct_to_correct_coef_a3):
        if not math.isfinite(value) or value < 0:
            raise SystemExit("c2c coefficients must be finite and non-negative")
    inputs = (args.train_input, args.holdout_input)
    outputs = (args.train_output, args.holdout_output, args.stats_output)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    if args.validate_existing:
        for path in outputs:
            if not path.is_file():
                raise SystemExit(f"existing output not found: {path}")
        stats = json.loads(args.stats_output.read_text(encoding="utf-8"))
        if stats.get("source_sha256") != {"train": digest(args.train_input), "holdout": digest(args.holdout_input)}:
            raise SystemExit("source fingerprint mismatch in existing ablation data")
        train_count, train_agents = validate_output(args.train_output, args)
        holdout_count, holdout_agents = validate_output(args.holdout_output, args)
        if stats.get("train_rows") != train_count or stats.get("holdout_rows") != holdout_count:
            raise SystemExit("existing row counts disagree with stats")
        print(json.dumps({"status": "valid", "train_rows": train_count, "holdout_rows": holdout_count, "train_agents": dict(train_agents), "holdout_agents": dict(holdout_agents)}, ensure_ascii=False))
        return
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"output exists; use --validate-existing or --overwrite: {existing[0]}")
    train_rows, train_digest, train_agents = transform_file(args.train_input, args.train_output, args)
    holdout_rows, holdout_digest, holdout_agents = transform_file(args.holdout_input, args.holdout_output, args)
    stats = {
        "schema_version": 1,
        "ablation": "wo_process_reward",
        "source": {
            "train": str(args.train_input.resolve()),
            "holdout": str(args.holdout_input.resolve()),
        },
        "source_sha256": {"train": digest(args.train_input), "holdout": digest(args.holdout_input)},
        "output_sha256": {"train": train_digest, "holdout": holdout_digest},
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "train_rows_by_agent": dict(train_agents),
        "holdout_rows_by_agent": dict(holdout_agents),
        "reward": {
            "base_weight": 1.0,
            "judge_weight": 0.0,
            "transition_boost": args.transition_boost,
            "correct_to_correct_coef_by_agent": {"A1": args.correct_to_correct_coef_a1, "A2": args.correct_to_correct_coef_a2, "A3": args.correct_to_correct_coef_a3},
            "sign_source": "deterministic_state_v3.current_answer_correct",
        },
    }
    temporary = args.stats_output.with_name(f".{args.stats_output.name}.tmp.{os.getpid()}")
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.stats_output)
    print(json.dumps({"status": "built", "train_rows": train_rows, "holdout_rows": holdout_rows, "train_agents": dict(train_agents), "holdout_agents": dict(holdout_agents)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
