#!/usr/bin/env python3
"""Summarize SAS MultiPL-E results selected by fixed split manifests."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-language and aggregate SAS pass@1 scores."
    )
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--completions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--evaluation", default="single-agent")
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize only result files that exist (for smoke-test subsets).",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Optional human-readable score summary written alongside the JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def pass_at_k(num_completions: int, num_correct: int, k: int) -> float:
    if num_completions <= 0 or num_correct <= 0:
        return 0.0
    if num_completions - num_correct < k:
        return 1.0
    return 1.0 - (
        math.comb(num_completions - num_correct, k)
        / math.comb(num_completions, k)
    )


def render_text_summary(summary: dict) -> str:
    lines = [
        "===== Final evaluation scores =====",
        f"Model: {summary['model_name']}",
        f"Evaluation: {summary['evaluation']}",
        f"Overall test problems: {summary['overall']['test_count']}",
        f"Overall correct: {summary['overall']['correct_count']}",
        f"Overall weighted pass@1: {summary['overall']['weighted_pass_at_1']:.6f}",
        f"Overall macro-language pass@1: "
        f"{summary['overall']['macro_language_pass_at_1']:.6f}",
        "",
        "root_dataset\tlanguage\ttest_count\tcorrect_count\tpass@1",
    ]
    lines.extend(
        f"{record['root_dataset']}\t{record['language']}\t"
        f"{record['test_count']}\t{record['correct_count']}\t"
        f"{record['pass_at_1']:.6f}"
        for record in summary["languages"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    completions_dir = args.completions_dir.resolve()
    records = []
    manifest_info = []

    for manifest_arg in args.split_manifest:
        manifest_path = manifest_arg.resolve()
        manifest = load_json(manifest_path)
        root_dataset = manifest["root_dataset"]
        manifest_info.append(
            {
                "root_dataset": root_dataset,
                "path": str(manifest_path),
                "seed": manifest["seed"],
            }
        )
        split_root = manifest_path.parent

        for language, details in sorted(manifest["languages"].items()):
            if args.languages and language not in args.languages:
                continue
            problem_scores = []
            correct_total = 0
            completion_total = 0
            for item in details["test"]:
                problem_id = item["problem_id"]
                result_path = (
                    completions_dir
                    / root_dataset
                    / language
                    / f"{problem_id}.results.json.gz"
                )
                if not result_path.exists():
                    if args.allow_partial:
                        continue
                    raise SystemExit(f"Missing result: {result_path}")
                result_data = load_json(result_path)
                results = result_data.get("results", [])
                if not results:
                    raise SystemExit(f"Empty result list: {result_path}")
                correct = sum(
                    result.get("status") == "OK" and result.get("exit_code") == 0
                    for result in results
                )
                problem_scores.append(pass_at_k(len(results), correct, 1))
                correct_total += correct
                completion_total += len(results)

            test_count = len(problem_scores)
            if test_count == 0 and args.allow_partial:
                continue
            records.append(
                {
                    "root_dataset": root_dataset,
                    "language": language,
                    "evaluator_language": details.get("evaluator_language"),
                    "test_count": test_count,
                    "completion_count": completion_total,
                    "correct_count": correct_total,
                    "pass_at_1": statistics.fmean(problem_scores),
                }
            )

    datasets = {}
    for root_dataset in sorted({record["root_dataset"] for record in records}):
        subset = [
            record for record in records if record["root_dataset"] == root_dataset
        ]
        datasets[root_dataset] = {
            "languages": len(subset),
            "test_count": sum(record["test_count"] for record in subset),
            "completion_count": sum(
                record["completion_count"] for record in subset
            ),
            "correct_count": sum(record["correct_count"] for record in subset),
            "weighted_pass_at_1": statistics.fmean(
                [
                    score
                    for record in subset
                    for score in [record["pass_at_1"]]
                    for _ in range(record["test_count"])
                ]
            ),
            "macro_language_pass_at_1": statistics.fmean(
                [record["pass_at_1"] for record in subset]
            ),
        }

    all_records = records
    summary = {
        "schema_version": 1,
        "evaluation": args.evaluation,
        "model_name": args.model_name,
        "manifests": manifest_info,
        "languages": records,
        "datasets": datasets,
        "overall": {
            "languages": len(all_records),
            "test_count": sum(record["test_count"] for record in all_records),
            "completion_count": sum(
                record["completion_count"] for record in all_records
            ),
            "correct_count": sum(record["correct_count"] for record in all_records),
            "macro_language_pass_at_1": statistics.fmean(
                [record["pass_at_1"] for record in all_records]
            ),
            "weighted_pass_at_1": statistics.fmean(
                [
                    score
                    for record in all_records
                    for score in [record["pass_at_1"]]
                    for _ in range(record["test_count"])
                ]
            ),
        },
    }
    if args.model_path is not None:
        summary["model_path"] = str(args.model_path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    text_summary = render_text_summary(summary)
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(text_summary)

    print(text_summary, end="")
    print(
        f"overall\tall\t{summary['overall']['test_count']}\t"
        f"{summary['overall']['correct_count']}\t"
        f"{summary['overall']['weighted_pass_at_1']:.6f}"
    )
    print(f"Summary: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
