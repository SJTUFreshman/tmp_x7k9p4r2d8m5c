#!/usr/bin/env python3
"""Write or validate one role's immutable MATH RL training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=("A1", "A2", "A3"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--prepare-manifest", type=Path, required=True)
    parser.add_argument("--parameters", required=True, help="Canonical JSON object")
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


def adapter_meta(path: Path) -> dict[str, Any]:
    config = file_meta(path / "adapter_config.json")
    weights = file_meta(path / "adapter_model.safetensors")
    digest = hashlib.sha256()
    for name, value in (("adapter_config.json", config), ("adapter_model.safetensors", weights)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value["sha256"].encode("ascii"))
        digest.update(b"\0")
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "files": [config, weights],
    }


def expected(args: argparse.Namespace) -> dict[str, Any]:
    parameters = json.loads(args.parameters)
    if not isinstance(parameters, dict):
        raise ValueError("--parameters must be a JSON object")
    adapter = adapter_meta(args.sft_adapter)
    return {
        "schema_version": 1,
        "dataset": "MATH",
        "source_split": "train",
        "agent": args.agent,
        "model": {
            "path": str(args.model.resolve()),
            "config": file_meta(args.model / "config.json"),
        },
        "policy_initialization": adapter,
        "frozen_reference": adapter,
        "policy_reference_identical_at_step_zero": True,
        "train": file_meta(args.train),
        "holdout": file_meta(args.holdout),
        "prepare_manifest": file_meta(args.prepare_manifest),
        "parameters": parameters,
    }


def main() -> None:
    args = parse_args()
    value = expected(args)
    if args.validate_existing:
        actual = json.loads(args.output.read_text(encoding="utf-8"))
        if actual != value:
            raise SystemExit("training contract changed; use a new TAG")
        print(json.dumps({"status": "valid", "agent": args.agent}))
        return
    if args.output.exists():
        raise SystemExit(f"training contract already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"status": "written", "agent": args.agent}))


if __name__ == "__main__":
    main()
