#!/usr/bin/env python3
"""Normalize reused MATH rollouts against the bundled evaluator contract."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from jca.experiments.math_rl_mas_thinking import score_math_rollouts as scorer
from jca.experiments.math_rl_mas_thinking.math_artifact_validator import (
    EVALUATOR_FINGERPRINT,
    MATH_EVAL_FINGERPRINT,
)
from jca.src.math_eval import MATH_EVAL_VERSION, compute_math_em, load_math_problems


SCHEMA = "math_reused_raw_normalization_v1"


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_load(raw: bytes, path: Path, line_no: int) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=reject_json_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
    return value


def group_key(row: dict[str, Any], path: Path, line_no: int) -> tuple[str, int]:
    problem_id = row.get("problem_id")
    rollout_idx = row.get("rollout_idx")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError(f"invalid problem_id at {path}:{line_no}")
    if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx < 0:
        raise ValueError(f"invalid rollout_idx at {path}:{line_no}")
    return problem_id, rollout_idx


def groups(
    path: Path,
    digest: Any | None = None,
) -> Iterator[tuple[tuple[str, int], list[dict[str, Any]]]]:
    current_key: tuple[str, int] | None = None
    current_rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if digest is not None:
                digest.update(raw)
            if not raw.strip():
                continue
            row = strict_load(raw, path, line_no)
            key = group_key(row, path, line_no)
            if current_key is not None and key != current_key:
                yield current_key, current_rows
                current_rows = []
            current_key = key
            current_rows.append(row)
    if current_key is not None:
        yield current_key, current_rows


def source_mode(row: dict[str, Any]) -> bool:
    values = []
    for field in ("thinking_enabled", "enable_thinking"):
        value = row.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be present and boolean")
        values.append(value)
    if values[0] != values[1]:
        raise ValueError("thinking aliases disagree")
    return values[0]


def normalize_group(
    rows: list[dict[str, Any]],
    problem: Any,
) -> tuple[list[dict[str, Any]], bool, dict[str, set[str]]]:
    if not rows:
        raise ValueError("empty rollout group")
    terminal = rows[-1].get("final_answer")
    if terminal is None:
        terminal = ""
    if not isinstance(terminal, str):
        raise ValueError("terminal final_answer must be a string or null")
    terminal_em = float(compute_math_em(terminal.strip(), problem.gold_answer))
    fingerprints = {
        "math_eval_fingerprint": set(),
        "evaluator_fingerprint": set(),
        "math_eval_version": set(),
    }
    normalized = []
    policy_mode: bool | None = None
    for source in rows:
        mode = source_mode(source)
        if policy_mode is not None and policy_mode != mode:
            raise ValueError("rollout group mixes policy thinking modes")
        policy_mode = mode
        for field in fingerprints:
            fingerprints[field].add(str(source.get(field)))
        row = dict(source)
        row["thinking_enabled"] = mode
        row["enable_thinking"] = mode
        row["math_eval_version"] = MATH_EVAL_VERSION
        row["math_eval_fingerprint"] = MATH_EVAL_FINGERPRINT
        row["evaluator_fingerprint"] = EVALUATOR_FINGERPRINT
        row["em"] = terminal_em
        normalized.append(row)
    scorer.validate_source_group(normalized)
    states = scorer.deterministic_states(normalized, problem)
    if set(states) != {int(row["turn"]) for row in normalized}:
        raise ValueError("deterministic state coverage is incomplete")
    assert policy_mode is not None
    return normalized, policy_mode, fingerprints


def serialize(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inspect_normalized(
    path: Path,
    problem_map: dict[str, Any],
    expected_rollouts: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    seen_groups: set[tuple[str, int]] = set()
    rollouts: dict[str, set[int]] = collections.defaultdict(set)
    modes: set[bool] = set()
    row_count = 0
    for key, rows in groups(path, digest):
        if key in seen_groups:
            raise ValueError(f"non-contiguous duplicate group: {key}")
        seen_groups.add(key)
        problem = problem_map.get(key[0])
        if problem is None:
            raise ValueError(f"unknown MATH problem: {key[0]}")
        normalized, mode, _ = normalize_group(rows, problem)
        if normalized != rows:
            raise ValueError(f"normalized output has stale evaluator fields: {key}")
        modes.add(mode)
        rollouts[key[0]].add(key[1])
        row_count += len(rows)
    validate_coverage(rollouts, modes, expected_rollouts)
    return {
        "sha256": digest.hexdigest(),
        "rows": row_count,
        "groups": len(seen_groups),
        "problems": len(rollouts),
        "policy_thinking_enabled": next(iter(modes)),
    }


def validate_coverage(
    rollouts: dict[str, set[int]],
    modes: set[bool],
    expected_rollouts: int,
) -> None:
    if not rollouts:
        raise ValueError("rollout input is empty")
    expected = set(range(expected_rollouts))
    invalid = [problem_id for problem_id, values in rollouts.items() if values != expected]
    if invalid:
        raise ValueError(
            f"{len(invalid)} problems do not contain rollout ids 0..{expected_rollouts - 1}; "
            f"first={invalid[0]}"
        )
    if len(modes) != 1:
        raise ValueError("rollout input mixes policy thinking modes")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must differ")
    if args.output.exists() or args.stats_output.exists():
        raise ValueError("normalization output already exists")
    problem_map = {
        problem.problem_id: problem
        for problem in load_math_problems(args.data_root, split="train")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    seen_groups: set[tuple[str, int]] = set()
    rollouts: dict[str, set[int]] = collections.defaultdict(set)
    modes: set[bool] = set()
    source_contracts: dict[str, set[str]] = collections.defaultdict(set)
    row_count = 0
    try:
        with temporary.open("wb") as handle:
            for key, rows in groups(args.input, source_digest):
                if key in seen_groups:
                    raise ValueError(f"non-contiguous duplicate group: {key}")
                seen_groups.add(key)
                problem = problem_map.get(key[0])
                if problem is None:
                    raise ValueError(f"unknown MATH problem: {key[0]}")
                normalized, mode, fingerprints = normalize_group(rows, problem)
                modes.add(mode)
                rollouts[key[0]].add(key[1])
                for field, values in fingerprints.items():
                    source_contracts[field].update(values)
                for row in normalized:
                    payload = serialize(row)
                    handle.write(payload)
                    output_digest.update(payload)
                    row_count += 1
            validate_coverage(rollouts, modes, args.expected_rollouts)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    payload = {
        "schema": SCHEMA,
        "dataset": "MATH",
        "source_path": str(args.input.resolve()),
        "source_sha256": source_digest.hexdigest(),
        "source_size_bytes": args.input.stat().st_size,
        "source_contracts": {
            key: sorted(values) for key, values in sorted(source_contracts.items())
        },
        "sampling_policy_lineage": args.sampling_policy_lineage,
        "reused_legacy_sampled_data": True,
        "output_path": str(args.output.resolve()),
        "output_sha256": output_digest.hexdigest(),
        "rows": row_count,
        "groups": len(seen_groups),
        "problems": len(rollouts),
        "expected_rollouts_per_problem": args.expected_rollouts,
        "policy_thinking_enabled": next(iter(modes)),
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "judge_scores_reused": False,
    }
    atomic_json(args.stats_output, payload)
    return payload


def validate_existing(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output.is_file() or not args.stats_output.is_file():
        raise ValueError("normalized output or stats are missing")
    stats = json.loads(args.stats_output.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "source_path": str(args.input.resolve()),
        "sampling_policy_lineage": args.sampling_policy_lineage,
        "reused_legacy_sampled_data": True,
        "output_path": str(args.output.resolve()),
        "expected_rollouts_per_problem": args.expected_rollouts,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "judge_scores_reused": False,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"normalization stats field {key} disagrees")
    if hash_file(args.input) != stats.get("source_sha256"):
        raise ValueError("reused raw source hash disagrees with normalization stats")
    problem_map = {
        problem.problem_id: problem
        for problem in load_math_problems(args.data_root, split="train")
    }
    observed = inspect_normalized(args.output, problem_map, args.expected_rollouts)
    for observed_key, stats_key in (
        ("sha256", "output_sha256"),
        ("rows", "rows"),
        ("groups", "groups"),
        ("problems", "problems"),
        ("policy_thinking_enabled", "policy_thinking_enabled"),
    ):
        if observed[observed_key] != stats.get(stats_key):
            raise ValueError(f"normalized output {observed_key} disagrees with stats")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=8)
    parser.add_argument("--sampling-policy-lineage", required=True)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_rollouts <= 0:
        raise SystemExit("--expected-rollouts must be positive")
    for path in (args.input, args.data_root):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    try:
        payload = validate_existing(args) if args.validate_existing else build(args)
    except Exception as exc:
        raise SystemExit(f"[normalize-reused] {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
