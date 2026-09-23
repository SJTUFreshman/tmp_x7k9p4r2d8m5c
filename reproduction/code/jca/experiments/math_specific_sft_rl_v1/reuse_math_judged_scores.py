#!/usr/bin/env python3
"""Migrate reusable MATH process scores onto a current normalized rollout."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from jca.experiments.math_rl_mas_thinking import score_math_rollouts as scorer
from jca.experiments.math_rl_mas_thinking.math_artifact_validator import (
    EVALUATOR_FINGERPRINT,
    MATH_EVAL_FINGERPRINT,
    validate_scored_output,
)
from jca.src.math_eval import MATH_EVAL_VERSION, load_math_problems


REUSE_SCHEMA = "math_judge_score_reuse_v1"
JUDGE_FIELDS = (
    "collaboration_judge_v3",
    "collaboration_judge_v4",
    "collaboration_judge_model_v3",
    "collaboration_judge_model_v4",
    "collaboration_judge_status_v3",
    "collaboration_judge_status_v4",
    "collaboration_judge_schema_v4",
    "judge_thinking_enabled",
    "judge_state_disagreements_v3",
)
STATE_SEMANTIC_FIELDS = (
    "previous_answer",
    "current_answer",
    "previous_answer_correct",
    "current_answer_correct",
    "answer_changed",
    "answer_state",
)


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
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
    seen: set[tuple[str, int]] = set()
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if digest is not None:
                digest.update(raw)
            if not raw.strip():
                continue
            row = strict_load(raw, path, line_no)
            key = group_key(row, path, line_no)
            if current_key is not None and key != current_key:
                if current_key in seen:
                    raise ValueError(f"non-contiguous duplicate group: {current_key}")
                seen.add(current_key)
                yield current_key, current_rows
                current_rows = []
            current_key = key
            current_rows.append(row)
    if current_key is not None:
        if current_key in seen:
            raise ValueError(f"non-contiguous duplicate group: {current_key}")
        yield current_key, current_rows


def state_projection(state: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(state.get(field) for field in STATE_SEMANTIC_FIELDS)


def serialize(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def patch_row(
    source: dict[str, Any],
    legacy: dict[str, Any],
    current_state: dict[str, Any],
    expected_judge_model: str,
) -> dict[str, Any]:
    if scorer._source_projection(legacy) != scorer._source_projection(source):
        raise ValueError("legacy judged row does not match the immutable source projection")
    if legacy.get("em") != source.get("em"):
        raise ValueError("legacy and current terminal EM disagree")
    legacy_state = legacy.get("deterministic_state_v3")
    if not isinstance(legacy_state, dict):
        raise ValueError("legacy judged row is missing deterministic state")
    if state_projection(legacy_state) != state_projection(current_state):
        raise ValueError("legacy and current deterministic state disagree")
    if legacy.get("judge_thinking_enabled") is not True:
        raise ValueError("legacy judge was not thinking-enabled")
    if legacy.get("collaboration_judge_schema_v4") != scorer.JUDGE_SCHEMA:
        raise ValueError("legacy judge schema differs")
    for field in ("collaboration_judge_status_v3", "collaboration_judge_status_v4"):
        if legacy.get(field) != scorer.JUDGE_STATUS:
            raise ValueError(f"legacy judged row has invalid {field}")
    for field in ("collaboration_judge_model_v3", "collaboration_judge_model_v4"):
        if legacy.get(field) != expected_judge_model:
            raise ValueError(f"legacy judged row has unexpected {field}")

    merged = dict(source)
    for field in JUDGE_FIELDS:
        if field in legacy:
            merged[field] = legacy[field]
    for field in ("collaboration_judge_v3", "collaboration_judge_v4"):
        payload = dict(merged.get(field) or {})
        payload["schema"] = scorer.JUDGE_SCHEMA
        payload["math_eval_version"] = MATH_EVAL_VERSION
        payload["math_eval_fingerprint"] = MATH_EVAL_FINGERPRINT
        payload["evaluator_fingerprint"] = EVALUATOR_FINGERPRINT
        if "process_score" in payload:
            payload.setdefault("judge_score", payload["process_score"])
            payload.setdefault("score_scale", scorer.DISCRETE_SCALE)
        merged[field] = payload
    merged["deterministic_state_v3"] = dict(current_state)
    merged["math_eval_version"] = MATH_EVAL_VERSION
    merged["math_eval_fingerprint"] = MATH_EVAL_FINGERPRINT
    merged["evaluator_fingerprint"] = EVALUATOR_FINGERPRINT
    merged["thinking_enabled"] = bool(source["thinking_enabled"])
    merged["enable_thinking"] = bool(source["enable_thinking"])
    merged["em"] = source["em"]
    if not scorer._row_contract_current(merged, strict=True):
        raise ValueError("migrated judged row fails the current score contract")
    if scorer._source_projection(merged) != scorer._source_projection(source):
        raise ValueError("score migration mutated an immutable source field")
    return merged


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.stats_output.exists() or args.reuse_stats_output.exists():
        raise ValueError("score-reuse output already exists")
    problem_map = {
        problem.problem_id: problem
        for problem in load_math_problems(args.data_root, split="train")
    }
    source_digest = hashlib.sha256()
    legacy_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    source_groups = groups(args.source, source_digest)
    legacy_groups = groups(args.legacy_judged, legacy_digest)
    rollouts: dict[str, set[int]] = collections.defaultdict(set)
    agents: collections.Counter[str] = collections.Counter()
    score_counts: collections.Counter[str] = collections.Counter()
    modes: set[bool] = set()
    legacy_math_fingerprints: set[str] = set()
    legacy_runtime_fingerprints: set[str] = set()
    row_count = 0
    group_count = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pairs = itertools.zip_longest(source_groups, legacy_groups)
            for source_item, legacy_item in pairs:
                if source_item is None or legacy_item is None:
                    raise ValueError("source and legacy judged group counts differ")
                source_key, source_rows = source_item
                legacy_key, legacy_rows = legacy_item
                if source_key != legacy_key:
                    raise ValueError(
                        f"source and legacy judged group order differs: {source_key} != {legacy_key}"
                    )
                if len(source_rows) != len(legacy_rows):
                    raise ValueError(f"group row count differs: {source_key}")
                turns = [int(row.get("turn", -1)) for row in source_rows]
                legacy_turns = [int(row.get("turn", -1)) for row in legacy_rows]
                if turns != list(range(len(source_rows))) or legacy_turns != turns:
                    raise ValueError(f"group turns differ or are non-contiguous: {source_key}")
                scorer.validate_source_group(source_rows)
                problem = problem_map.get(source_key[0])
                if problem is None:
                    raise ValueError(f"unknown MATH train problem: {source_key[0]}")
                current_states = scorer.deterministic_states(source_rows, problem)
                rollouts[source_key[0]].add(source_key[1])
                group_count += 1
                for source, legacy in zip(source_rows, legacy_rows):
                    turn = int(source["turn"])
                    migrated = patch_row(
                        source,
                        legacy,
                        current_states[turn],
                        args.expected_judge_model,
                    )
                    payload = serialize(migrated)
                    handle.write(payload)
                    output_digest.update(payload)
                    row_count += 1
                    agents[str(migrated["agent_id"])] += 1
                    score_counts[str(migrated["collaboration_judge_v4"]["judge_score"])] += 1
                    mode = scorer.policy_thinking_mode(migrated)
                    if mode is None:
                        raise ValueError("migrated row lacks a policy thinking mode")
                    modes.add(mode)
                    legacy_math_fingerprints.add(str(legacy.get("math_eval_fingerprint")))
                    legacy_runtime_fingerprints.add(str(legacy.get("evaluator_fingerprint")))
            expected = set(range(args.expected_rollouts))
            invalid = [problem_id for problem_id, values in rollouts.items() if values != expected]
            if invalid:
                raise ValueError(
                    f"{len(invalid)} problems have incomplete rollout coverage; first={invalid[0]}"
                )
            if len(modes) != 1:
                raise ValueError("migrated output mixes policy thinking modes")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    standard_stats = {
        "groups": group_count,
        "rows": row_count,
        "problems": len(rollouts),
        "agents": dict(agents),
        "score_counts": dict(score_counts),
        "judge_model": args.expected_judge_model,
        "judge_thinking_enabled": True,
        "policy_thinking_enabled": next(iter(modes)),
        "policy_thinking_retained": False,
        "schema": scorer.JUDGE_SCHEMA,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
    }
    reuse_stats = {
        **standard_stats,
        "schema": REUSE_SCHEMA,
        "judge_schema": scorer.JUDGE_SCHEMA,
        "source_path": str(args.source.resolve()),
        "source_sha256": source_digest.hexdigest(),
        "legacy_judged_path": str(args.legacy_judged.resolve()),
        "legacy_judged_sha256": legacy_digest.hexdigest(),
        "output_path": str(args.output.resolve()),
        "output_sha256": output_digest.hexdigest(),
        "judge_scores_reused": True,
        "immutable_source_projection_mismatches": 0,
        "deterministic_state_mismatches": 0,
        "legacy_math_eval_fingerprints": sorted(legacy_math_fingerprints),
        "legacy_evaluator_fingerprints": sorted(legacy_runtime_fingerprints),
    }
    atomic_json(args.stats_output, standard_stats)
    atomic_json(args.reuse_stats_output, reuse_stats)
    return reuse_stats


def validate_existing(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.output, args.stats_output, args.reuse_stats_output):
        if not path.is_file():
            raise ValueError(f"required score-reuse output is missing: {path}")
    stats = json.loads(args.reuse_stats_output.read_text(encoding="utf-8"))
    expected = {
        "schema": REUSE_SCHEMA,
        "judge_schema": scorer.JUDGE_SCHEMA,
        "source_path": str(args.source.resolve()),
        "legacy_judged_path": str(args.legacy_judged.resolve()),
        "output_path": str(args.output.resolve()),
        "judge_scores_reused": True,
        "judge_model": args.expected_judge_model,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "immutable_source_projection_mismatches": 0,
        "deterministic_state_mismatches": 0,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"score-reuse stats field {key} disagrees")
    for path, key in (
        (args.source, "source_sha256"),
        (args.legacy_judged, "legacy_judged_sha256"),
        (args.output, "output_sha256"),
    ):
        if hash_file(path) != stats.get(key):
            raise ValueError(f"score-reuse hash disagrees: {path}")
    observed = validate_scored_output(
        args.source,
        args.output,
        args.stats_output,
        args.data_root,
        args.expected_rollouts,
    )
    for key in ("groups", "rows", "problems"):
        if observed[key] != stats.get(key):
            raise ValueError(f"validated score count {key} disagrees")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--legacy-judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--reuse-stats-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=8)
    parser.add_argument("--expected-judge-model", default="qwen14b_gsm_judge")
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_rollouts <= 0:
        raise SystemExit("--expected-rollouts must be positive")
    for path in (args.source, args.legacy_judged, args.data_root):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    try:
        payload = validate_existing(args) if args.validate_existing else build(args)
    except Exception as exc:
        raise SystemExit(f"[reuse-judged] {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
