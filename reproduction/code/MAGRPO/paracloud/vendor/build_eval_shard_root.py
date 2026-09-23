#!/usr/bin/env python3
"""Materialize one MATH JSONL shard as a loader-compatible test root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ID_PATTERN = re.compile(r"^(?P<subject>.+)_(?P<index>[0-9]{5})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shard-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("problem_id") or "")
            if not ID_PATTERN.fullmatch(problem_id):
                raise ValueError(f"invalid problem_id at {path}:{line_number}")
            ids.append(problem_id)
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError(f"shard must contain 500 unique problem IDs, got {len(ids)}")
    return ids


def load_selected_rows(source_root: Path, ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    selected: dict[str, set[int]] = defaultdict(set)
    for problem_id in ids:
        match = ID_PATTERN.fullmatch(problem_id)
        assert match is not None
        selected[match.group("subject")].add(int(match.group("index")))
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for subject, indexes in sorted(selected.items()):
        source = source_root / subject / "test-00000-of-00001.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_parquet(source)
        rows: list[dict[str, Any]] = []
        for index in sorted(indexes):
            if index >= len(frame):
                raise ValueError(f"missing source row {subject}_{index:05d}")
            row = frame.iloc[index]
            rows.append({column: row[column] for column in frame.columns})
        rows_by_subject[subject] = rows
    if sum(len(rows) for rows in rows_by_subject.values()) != len(ids):
        raise ValueError("selected source rows do not match shard size")
    return rows_by_subject


def expected_manifest(args: argparse.Namespace, ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "shard_file": str(args.shard_file.resolve()),
        "shard_sha256": sha256(args.shard_file),
        "problem_ids": ids,
        "count": len(ids),
    }


def main() -> None:
    args = parse_args()
    for path in (args.source_root, args.shard_file):
        if not path.exists():
            raise SystemExit(f"path not found: {path}")
    ids = shard_ids(args.shard_file)
    manifest = expected_manifest(args, ids)
    if args.validate_existing:
        if not args.manifest.is_file():
            raise SystemExit(f"manifest not found: {args.manifest}")
        actual = json.loads(args.manifest.read_text(encoding="utf-8"))
        if actual != manifest:
            raise SystemExit("evaluation shard manifest does not match source")
        for subject in sorted({ID_PATTERN.fullmatch(value).group("subject") for value in ids}):
            parquet = args.output_root / subject / "test-00000-of-00001.parquet"
            if not parquet.is_file():
                raise SystemExit(f"evaluation shard parquet missing: {parquet}")
        print(json.dumps({"status": "valid", "count": len(ids)}, ensure_ascii=False))
        return
    if args.output_root.exists() or args.manifest.exists():
        raise SystemExit("evaluation shard output exists; use --validate-existing")
    rows_by_subject = load_selected_rows(args.source_root, ids)
    import pandas as pd

    for subject, rows in rows_by_subject.items():
        destination = args.output_root / subject / "test-00000-of-00001.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_name(f".{args.manifest.name}.tmp.{os.getpid()}")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, args.manifest)
    print(json.dumps({"status": "built", "count": len(ids), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
