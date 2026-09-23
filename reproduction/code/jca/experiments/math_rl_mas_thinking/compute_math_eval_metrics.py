#!/usr/bin/env python3
"""Compute symbolic EM and GSM-style numeric soft F1 for MATH eval JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from jca.src.math_eval import (
    MATH_EVAL_VERSION,
    MATH_SOFT_F1_VERSION,
    finite_real_scalar,
    gsm_numeric_ratio,
    math_answers_equivalent,
    math_soft_f1,
)


METRICS_VERSION = "math_eval_metrics_v1"


def _row_fields(
    row: dict[str, Any], path: Path, line_number: int
) -> tuple[tuple[str, int], str, str]:
    trajectory = row.get("trajectory")
    problem = row.get("problem")
    if isinstance(trajectory, dict) and isinstance(problem, dict):
        problem_id = trajectory.get("problem_id")
        rollout_idx = trajectory.get("rollout_idx", 0)
        gold_answer = problem.get("gold_answer")
    else:
        problem_id = row.get("problem_id")
        rollout_idx = row.get("rollout_idx", 0)
        gold_answer = row.get("gold_answer")

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError(f"invalid problem_id in {path}:{line_number}")
    if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int):
        raise ValueError(f"invalid rollout_idx in {path}:{line_number}")
    if not isinstance(gold_answer, str):
        raise ValueError(f"missing gold_answer in {path}:{line_number}")

    prediction = row.get("final_answer")
    if "final_answer" not in row:
        prediction = row.get("prediction")
    if prediction is not None and not isinstance(prediction, str):
        raise ValueError(f"invalid prediction in {path}:{line_number}")
    return (problem_id, rollout_idx), str(prediction or "").strip(), gold_answer.strip()


def compute_metrics(
    input_path: Path,
    *,
    expected_count: int = 0,
    recorded_em_policy: str = "require",
) -> dict[str, Any]:
    if recorded_em_policy not in {"require", "report"}:
        raise ValueError(
            "recorded_em_policy must be either 'require' or 'report'"
        )
    identities: set[tuple[str, int]] = set()
    source_hash = hashlib.sha256()
    count = 0
    symbolic_correct = 0
    recorded_correct = 0.0
    recorded_em_mismatch_count = 0
    recorded_em_mismatch_examples: list[dict[str, Any]] = []
    soft_total = 0.0
    numeric_total = 0.0
    numeric_count = 0
    termination_counts: Counter[str] = Counter()
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"blank line in {input_path}:{line_number}")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid JSON in {input_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row in {input_path}:{line_number}")
            identity, prediction, gold_answer = _row_fields(
                row, input_path, line_number
            )
            if identity in identities:
                raise ValueError(
                    f"duplicate identity {identity} in {input_path}:{line_number}"
                )
            identities.add(identity)

            equivalent = math_answers_equivalent(prediction, gold_answer)
            symbolic_correct += int(equivalent)
            recorded_em = row.get("em")
            if (
                isinstance(recorded_em, bool)
                or not isinstance(recorded_em, (int, float))
            ):
                raise ValueError(f"invalid recorded EM at row {line_number}")
            recorded_em_value = float(recorded_em)
            recorded_correct += recorded_em_value
            if recorded_em_value != float(equivalent):
                if recorded_em_policy == "require":
                    raise ValueError(f"recorded EM disagrees at row {line_number}")
                recorded_em_mismatch_count += 1
                if len(recorded_em_mismatch_examples) < 10:
                    recorded_em_mismatch_examples.append(
                        {
                            "line": line_number,
                            "problem_id": identity[0],
                            "rollout_idx": identity[1],
                            "recorded_em": recorded_em_value,
                            "recomputed_em": float(equivalent),
                        }
                    )
            soft_score, numeric_eligible, numeric_score = math_soft_f1(
                prediction, gold_answer
            )
            soft_total += soft_score
            if numeric_eligible:
                numeric_count += 1
                numeric_total += float(numeric_score or 0.0)
            termination_counts[str(row.get("terminated_by") or "unknown")] += 1
            count += 1

    if expected_count and count != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {count}")
    metrics = {
        "metrics_version": METRICS_VERSION,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_soft_f1_version": MATH_SOFT_F1_VERSION,
        "source": str(input_path.resolve()),
        "source_sha256": source_hash.hexdigest(),
        "count": count,
        "symbolic_correct": symbolic_correct,
        "symbolic_em": symbolic_correct / count if count else 0.0,
        "math_soft_f1": soft_total / count if count else 0.0,
        "numeric_soft_f1": numeric_total / numeric_count if numeric_count else 0.0,
        "numeric_coverage": {
            "count": numeric_count,
            "rate": numeric_count / count if count else 0.0,
        },
        "nonnumeric_gold_count": count - numeric_count,
        "termination_counts": dict(sorted(termination_counts.items())),
    }
    if recorded_em_policy == "report":
        metrics.update(
            {
                "recorded_correct": recorded_correct,
                "recorded_em": recorded_correct / count if count else 0.0,
                "recorded_em_mismatch_count": recorded_em_mismatch_count,
                "recorded_em_mismatch_examples": recorded_em_mismatch_examples,
            }
        )
    return metrics


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument(
        "--recorded-em-policy",
        choices=("require", "report"),
        default="require",
        help="Require stored EM to match the current evaluator or report differences",
    )
    args = parser.parse_args()
    if args.expected_count < 0:
        parser.error("--expected-count must be nonnegative")
    metrics = compute_metrics(
        args.input,
        expected_count=args.expected_count,
        recorded_em_policy=args.recorded_em_policy,
    )
    write_metrics(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
