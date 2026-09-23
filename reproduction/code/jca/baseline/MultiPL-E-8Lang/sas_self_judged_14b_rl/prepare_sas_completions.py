#!/usr/bin/env python3
"""Convert SAS rollout records into standard MultiPL-E completion files."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MULTIPL_E_SCRIPTS = PROJECT_ROOT / "Code" / "MultiPL-E" / "scripts"
sys.path.insert(0, str(MULTIPL_E_SCRIPTS))

from multipl_e_completion_adapter import normalize_completion


def normalized_completion(record: dict) -> tuple[str, str]:
    row = dict(record.get("row", {}))
    language = str(record.get("language", ""))
    raw_completion = str(record.get("final_completion", ""))
    completion = normalize_completion(
        raw_completion,
        str(row.get("prompt", "")),
        str(row.get("tests", "")),
        list(row.get("stop_tokens") or []),
        language,
    )
    return raw_completion, completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.rollout.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = record.get("dataset")
            language = record.get("language")
            problem_id = record.get("problem_id")
            if not all(isinstance(value, str) and value for value in (dataset, language, problem_id)):
                raise ValueError(f"missing task identity at line {line_no}")
            row = dict(record.get("row", {}))
            raw_completion, completion = normalized_completion(record)
            row.update({"dataset": dataset, "name": problem_id, "language": language})
            row["completions"] = [completion]
            row["raw_completions"] = [raw_completion]
            row["completion_adapter"] = "qwen_code_continuation_v4"
            path = args.output_dir / dataset / language / f"{problem_id}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(row, handle, ensure_ascii=False)
            count += 1
    print(f"prepared_completions={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
