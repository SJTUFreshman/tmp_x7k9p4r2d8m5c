#!/usr/bin/env python3
"""Write and validate the base-data manifest for wo_process_reward v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "math_wo_process_reward_base_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-holdout", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def expected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    if stats.get("ablation") != "wo_process_reward":
        raise ValueError("reward transformation stats lack the ablation marker")
    reward = stats.get("reward") or {}
    if reward.get("base_weight") != 1.0 or reward.get("judge_weight") != 0.0:
        raise ValueError("reward transformation did not remove the process/judge component")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": "MATH",
        "ablation": "wo_process_reward",
        "invariant": (
            "same selected trajectories and policy lineage as the paired main run; "
            "only process/judge reward is removed"
        ),
        "source": {
            "prepare_manifest": file_meta(args.source_manifest),
            "train": file_meta(args.source_train),
            "holdout": file_meta(args.source_holdout),
            "prepare_contract": source_manifest,
        },
        "outputs": {
            "train": file_meta(args.train),
            "holdout": file_meta(args.holdout),
            "stats": file_meta(args.stats),
        },
        "reward": reward,
        "training_contract": {
            "algorithm": "conservative_signed_awr_v4_protocol_floor_v1",
            "initialization": "paired MATH-specific SFT adapters",
            "reference": "frozen identical paired MATH-specific SFT adapters",
            "negative_mask": "role-aware semantic decision fields",
            "protocol_floor": "advantage-independent one-sided SFT log-prob floor",
        },
    }


def main() -> None:
    args = parse_args()
    expected = expected_manifest(args)
    if args.validate_existing:
        if not args.output.is_file():
            raise SystemExit(f"manifest missing: {args.output}")
        actual = json.loads(args.output.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("wo_process_reward base manifest no longer matches its inputs")
        print(json.dumps({"status": "valid", "manifest": str(args.output)}))
        return
    if args.output.exists():
        raise SystemExit("manifest exists; validate it or choose a new ablation root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"status": "built", "manifest": str(args.output)}))


if __name__ == "__main__":
    main()
