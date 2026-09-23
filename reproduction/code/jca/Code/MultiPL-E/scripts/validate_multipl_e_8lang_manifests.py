#!/usr/bin/env python3
"""Validate the fixed eight-language benchmark split manifests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


EXPECTED_LANGUAGES = ("py", "cpp", "java", "php", "ts", "cs", "sh", "js")
EXPECTED_TEST_COUNTS = {"humaneval": 49, "mbpp": 120}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def problem_ids(items: object, label: str) -> list[str]:
    if not isinstance(items, list):
        raise ValueError(f"{label} must be a list")
    ids = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("problem_id"), str):
            raise ValueError(f"{label}[{index}] has no string problem_id")
        ids.append(item["problem_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate problem IDs")
    return ids


def jsonl_problem_ids(path: Path) -> list[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    ids = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            problem_id = row.get("name")
            if not isinstance(problem_id, str):
                raise ValueError(f"{path}:{line_number} has no string name")
            ids.append(problem_id)
    return ids


def validate_manifest(path: Path, seed: int) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"manifest does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    root_dataset = data.get("root_dataset")
    if root_dataset not in EXPECTED_TEST_COUNTS:
        raise ValueError(f"{path}: unsupported root_dataset {root_dataset!r}")
    if data.get("schema_version") != 2:
        raise ValueError(f"{path}: schema_version must be 2")
    if data.get("seed") != seed:
        raise ValueError(f"{path}: seed is {data.get('seed')!r}, expected {seed}")
    train_ratio = data.get("train_ratio")
    test_ratio = data.get("test_ratio")
    if (
        not isinstance(train_ratio, (int, float))
        or not isinstance(test_ratio, (int, float))
        or not math.isclose(train_ratio, 0.7)
        or not math.isclose(test_ratio, 0.3)
    ):
        raise ValueError(f"{path}: split ratio must be train=0.7/test=0.3")

    languages = data.get("languages")
    if not isinstance(languages, dict):
        raise ValueError(f"{path}: languages must be an object")
    if set(languages) != set(EXPECTED_LANGUAGES):
        raise ValueError(
            f"{path}: languages are {sorted(languages)}, "
            f"expected {list(EXPECTED_LANGUAGES)}"
        )

    common = data.get("common_split")
    if not isinstance(common, dict):
        raise ValueError(f"{path}: common_split must be an object")
    common_train = common.get("train_problem_ids")
    common_test = common.get("test_problem_ids")
    if not isinstance(common_train, list) or not isinstance(common_test, list):
        raise ValueError(f"{path}: common split ID lists are missing")
    if len(common_train) != len(set(common_train)):
        raise ValueError(f"{path}: common train IDs contain duplicates")
    if len(common_test) != len(set(common_test)):
        raise ValueError(f"{path}: common test IDs contain duplicates")
    if set(common_train) & set(common_test):
        raise ValueError(f"{path}: train/test problem IDs overlap")
    expected_test_count = EXPECTED_TEST_COUNTS[root_dataset]
    if len(common_test) != expected_test_count:
        raise ValueError(
            f"{path}: test count is {len(common_test)}, expected {expected_test_count}"
        )

    split_root = path.parent
    for language in EXPECTED_LANGUAGES:
        details = languages[language]
        train_ids = problem_ids(details.get("train"), f"{root_dataset}/{language}/train")
        test_ids = problem_ids(details.get("test"), f"{root_dataset}/{language}/test")
        if train_ids != common_train or test_ids != common_test:
            raise ValueError(
                f"{path}: {language} split IDs differ from common_split or ordering"
            )
        if details.get("train_count") != len(common_train):
            raise ValueError(f"{path}: invalid {language} train_count")
        if details.get("test_count") != len(common_test):
            raise ValueError(f"{path}: invalid {language} test_count")
        for split_name, expected_ids in (
            ("train", common_train),
            ("test", common_test),
        ):
            relative = details.get(f"{split_name}_file")
            split_path = split_root / relative if isinstance(relative, str) else None
            if split_path is None or not split_path.is_file():
                raise ValueError(
                    f"{path}: missing {language} {split_name} file: {relative!r}"
                )
            expected_sha256 = details.get(f"{split_name}_sha256")
            actual_sha256 = sha256(split_path)
            if expected_sha256 != actual_sha256:
                raise ValueError(
                    f"{path}: {language} {split_name} SHA256 mismatch"
                )
            if jsonl_problem_ids(split_path) != expected_ids:
                raise ValueError(
                    f"{path}: {language} {split_name} file IDs or ordering differ"
                )

    return {
        "root_dataset": root_dataset,
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "seed": seed,
        "train_count": len(common_train),
        "test_count": len(common_test),
        "languages": list(EXPECTED_LANGUAGES),
    }


def main() -> int:
    args = parse_args()
    reports = [validate_manifest(path, args.seed) for path in args.manifest]
    roots = [report["root_dataset"] for report in reports]
    if len(roots) != len(set(roots)):
        raise SystemExit("duplicate root_dataset manifests were provided")
    print(json.dumps({"status": "ok", "manifests": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
