#!/usr/bin/env python3
"""Validate that a SAS rollout exactly covers an input JSONL prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def identity(row: dict, *, rollout: bool) -> tuple[object, object, object]:
    problem_key = "problem_id" if rollout else "name"
    return row.get("dataset"), row.get(problem_key), row.get("language")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    inputs = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    records = [json.loads(line) for line in args.rollout.open(encoding="utf-8") if line.strip()]
    expected = args.expected_count if args.expected_count is not None else len(inputs)
    if expected < 1 or expected > len(inputs):
        raise ValueError(f"invalid expected count {expected}; input has {len(inputs)} rows")
    if len(records) != expected:
        raise ValueError(f"rollout has {len(records)} records; expected {expected}")
    for index, (row, record) in enumerate(zip(inputs[:expected], records), 1):
        if identity(row, rollout=False) != identity(record, rollout=True):
            raise ValueError(
                f"identity mismatch at record {index}: "
                f"input={identity(row, rollout=False)} rollout={identity(record, rollout=True)}"
            )
        if not isinstance(record.get("raw_responses"), list) or not record["raw_responses"]:
            raise ValueError(f"missing raw_responses at record {index}")
        if record.get("enable_thinking") is not False:
            raise ValueError(f"enable_thinking is not false at record {index}")
    print(f"validated_rollout_records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
