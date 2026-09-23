#!/usr/bin/env python3
"""Build SAS execution outcomes from MultiPL-E result files."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for completion_path in sorted(args.completions_dir.rglob("*.json.gz")):
        if completion_path.name.endswith(".results.json.gz"):
            continue
        result_path = completion_path.with_name(completion_path.name[:-8] + ".results.json.gz")
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        completion = load(completion_path)
        result = load(result_path)
        results = result.get("results", [])
        passed = bool(results) and all(
            item.get("status") == "OK" and item.get("exit_code") == 0
            for item in results[:1]
        )
        rows.append({
            "dataset": completion["dataset"],
            "name": completion["name"],
            "language": completion["language"],
            "passed": passed,
            "result_file": str(result_path),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"outcomes={len(rows)} passed={sum(row['passed'] for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
