#!/usr/bin/env python3
"""Build the legacy-v13 data with every judge-score effect removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PROJECT_ROOT = Path("/data/wangyuheng/jca")
DEFAULT_SOURCE = DEFAULT_PROJECT_ROOT / "rl_data/gsm/judge_rl/v13_14b/source_rejudged.jsonl"
DEFAULT_ORIGINAL_TRAIN = (
    DEFAULT_PROJECT_ROOT
    / "rl_data/gsm/judge_rl/gsm_judge_rl_v13_14b_role_c2c_1_05_1/train.jsonl"
)

EXPECTED_SOURCE_SHA256 = "f530cb6a697ddf0ebe669cebde78a3e18d776dd2eca049257b5cb1ec757f4228"
EXPECTED_ORIGINAL_TRAIN_SHA256 = "54edcb7ca00716199e2feb3b79ca11452cfb8cd4e3a15b43e5de7ad3ab97a2a0"
EXPECTED_SOURCE_ROWS = 26920
EXPECTED_ORIGINAL_TRAIN_ROWS = 9995
EXPECTED_TRAIN_ROWS = 10184
EXPECTED_HOLDOUT_ROWS = 1226
EXPECTED_ADDED_TRAIN_ROWS = 189
EXPECTED_TRAIN_SHA256 = "47ec3f9e9629002d5ef9b893ec52d8789952b1fa7b39687650706fac27389b86"
EXPECTED_HOLDOUT_SHA256 = "e213004b36e3805bb8884b5b85de1f1564531406615ac3027e09a37c7a01c670"
EXPECTED_TRAIN_ROWS_BY_AGENT = {"A1": 3814, "A2": 3568, "A3": 2802}
EXPECTED_SELECTED_TRAIN_TRAJECTORIES = {
    "a1_correct": 1054,
    "no_correction": 703,
    "correction_success": 1229,
    "correction_failed": 527,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the strict v13 without-process-reward ablation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--original-train", type=Path, default=DEFAULT_ORIGINAL_TRAIN)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing outputs without rebuilding or writing them.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build and validate in memory without writing output files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def row_identity(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    identity = (
        str(row.get("problem_id") or ""),
        int(row.get("rollout_idx", -1)),
        int(row.get("turn", -1)),
        str(row.get("agent_id") or ""),
    )
    if not identity[0] or identity[1] < 0 or identity[2] < 0:
        raise ValueError(f"invalid row identity: {identity}")
    if identity[3] not in EXPECTED_TRAIN_ROWS_BY_AGENT:
        raise ValueError(f"invalid agent in row identity: {identity}")
    return identity


def identities(rows: Iterable[Mapping[str, Any]], label: str) -> set[tuple[str, int, int, str]]:
    row_ids = [row_identity(row) for row in rows]
    unique_ids = set(row_ids)
    if len(unique_ids) != len(row_ids):
        raise ValueError(f"{label} contains duplicate row identities")
    return unique_ids


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_dataset(
    train: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    original_train: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require_equal(len(original_train), EXPECTED_ORIGINAL_TRAIN_ROWS, "original v13 train rows")
    require_equal(len(train), EXPECTED_TRAIN_ROWS, "ablation train rows")
    require_equal(len(holdout), EXPECTED_HOLDOUT_ROWS, "ablation holdout rows")

    original_ids = identities(original_train, "original v13 train")
    train_ids = identities(train, "ablation train")
    holdout_ids = identities(holdout, "ablation holdout")
    if train_ids & holdout_ids:
        raise ValueError("ablation train and holdout row identities overlap")
    missing_original = original_ids - train_ids
    if missing_original:
        raise ValueError(
            f"ablation train is missing {len(missing_original)} original v13 row identities"
        )
    require_equal(
        len(train_ids - original_ids),
        EXPECTED_ADDED_TRAIN_ROWS,
        "new rows after disabling judge filtering",
    )

    rows_by_agent = dict(Counter(str(row.get("agent_id")) for row in train))
    require_equal(rows_by_agent, EXPECTED_TRAIN_ROWS_BY_AGENT, "train rows by agent")

    for index, row in enumerate(train):
        require_equal(float(row.get("reward_base_weight", -1)), 1.0, f"train[{index}] base weight")
        require_equal(float(row.get("reward_judge_weight", -1)), 0.0, f"train[{index}] judge weight")

    reward_stats = stats.get("reward") or {}
    require_equal(float(reward_stats.get("base_weight", -1)), 1.0, "stats base weight")
    require_equal(float(reward_stats.get("judge_weight", -1)), 0.0, "stats judge weight")
    require_equal(float(reward_stats.get("min_aligned_judge", 0)), -1.0, "stats judge threshold")
    require_equal(float(reward_stats.get("transition_boost", -1)), 1.25, "stats transition boost")
    require_equal(
        reward_stats.get("effective_correct_to_correct_coef_by_agent"),
        {"A1": 1.0, "A2": 0.5, "A3": 1.0},
        "stats role-specific c2c coefficients",
    )
    require_equal(stats.get("source_rows"), EXPECTED_SOURCE_ROWS, "stats source rows")
    require_equal(
        (stats.get("selected_trajectory_counts") or {}).get("train"),
        EXPECTED_SELECTED_TRAIN_TRAJECTORIES,
        "selected training trajectories",
    )
    require_equal((stats.get("train") or {}).get("rows"), EXPECTED_TRAIN_ROWS, "stats train rows")
    require_equal((stats.get("holdout") or {}).get("rows"), EXPECTED_HOLDOUT_ROWS, "stats holdout rows")
    train_rejections = (stats.get("rejections") or {}).get("train") or {}
    require_equal(int(train_rejections.get("gpt_correctness_conflict", 0)), 0, "judge-conflict rejections")

    train_sha256 = jsonl_sha256(train)
    holdout_sha256 = jsonl_sha256(holdout)
    require_equal(train_sha256, EXPECTED_TRAIN_SHA256, "ablation train SHA256")
    require_equal(holdout_sha256, EXPECTED_HOLDOUT_SHA256, "ablation holdout SHA256")

    return {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "original_train_sha256": EXPECTED_ORIGINAL_TRAIN_SHA256,
        "original_train_rows": EXPECTED_ORIGINAL_TRAIN_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "holdout_rows": EXPECTED_HOLDOUT_ROWS,
        "train_sha256": train_sha256,
        "holdout_sha256": holdout_sha256,
        "train_rows_by_agent": EXPECTED_TRAIN_ROWS_BY_AGENT,
        "original_train_identities_preserved": len(original_ids),
        "rows_added_after_disabling_judge_filter": len(train_ids - original_ids),
        "reward": {
            "base_weight": 1.0,
            "judge_weight": 0.0,
            "min_aligned_judge": -1.0,
            "transition_boost": 1.25,
            "correct_to_correct_coef_by_agent": {"A1": 1.0, "A2": 0.5, "A3": 1.0},
        },
        "seed": 42,
        "self_handoff_compatibility": "temporary module-level detector override",
    }


def validate_hashes(source: Path, original_train: Path) -> None:
    for path in (source, original_train):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_equal(sha256_file(source), EXPECTED_SOURCE_SHA256, "v13 source SHA256")
    require_equal(
        sha256_file(original_train),
        EXPECTED_ORIGINAL_TRAIN_SHA256,
        "original v13 train SHA256",
    )


def write_outputs(
    builder: Any,
    train_output: Path,
    holdout_output: Path,
    stats_output: Path,
    train: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    outputs = (train_output, holdout_output, stats_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary = [path.with_name(f".{path.name}.tmp.{os.getpid()}") for path in outputs]
    try:
        builder.write_jsonl(temporary[0], train)
        builder.write_jsonl(temporary[1], holdout)
        temporary[2].write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for source, destination in zip(temporary, outputs):
            os.replace(source, destination)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()
    if args.validate_existing and args.check_only:
        raise SystemExit("--validate-existing and --check-only are mutually exclusive")
    if args.overwrite and (args.validate_existing or args.check_only):
        raise SystemExit("--overwrite cannot be combined with a read-only mode")
    validate_hashes(args.source, args.original_train)
    original_train = read_jsonl(args.original_train)

    if args.validate_existing:
        train = read_jsonl(args.train_output)
        holdout = read_jsonl(args.holdout_output)
        stats = json.loads(args.stats_output.read_text(encoding="utf-8"))
        contract = validate_dataset(train, holdout, stats, original_train)
        require_equal(stats.get("ablation_contract"), contract, "saved ablation contract")
        print(json.dumps({"status": "valid", **contract}, ensure_ascii=False, indent=2))
        return

    project_parent = str(args.project_root.resolve().parent)
    if project_parent not in sys.path:
        sys.path.insert(0, project_parent)
    from jca.gsm.scripts import prepare_gsm_judge_rl_v5_success_data as builder

    source_rows = builder.read_jsonl(args.source)
    original_detector = builder.self_handoff_location
    builder.self_handoff_location = lambda _row: None
    try:
        train, holdout, stats = builder.build_data(
            source_rows,
            expected_rollouts_per_problem=8,
            drop_all_failed_problems=True,
            holdout_fraction=0.1,
            ratios={
                "a1_correct": 0.30,
                "no_correction": 0.20,
                "correction_success": 0.35,
                "correction_failed": 0.15,
            },
            max_train_trajectories=0,
            max_holdout_trajectories=0,
            base_reward_weight=1.0,
            judge_reward_weight=0.0,
            transition_boost=1.25,
            min_aligned_judge=-1.0,
            seed=42,
            correct_to_correct_coef=1.0,
            drop_wrong_to_wrong_stop=False,
            correct_to_correct_coef_by_agent={"A1": 1.0, "A2": 0.5, "A3": 1.0},
            correct_to_correct_handoff_coef=1.0,
            wrong_to_correct_multiplier=None,
            correct_to_wrong_multiplier=None,
            expected_start_agent="A1",
        )
    finally:
        builder.self_handoff_location = original_detector

    contract = validate_dataset(train, holdout, stats, original_train)
    if args.check_only:
        print(json.dumps({"status": "valid_in_memory", **contract}, ensure_ascii=False, indent=2))
        return
    stats = dict(stats)
    stats["ablation_contract"] = contract
    write_outputs(
        builder,
        args.train_output,
        args.holdout_output,
        args.stats_output,
        train,
        holdout,
        stats,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": "built", **contract}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
