#!/usr/bin/env python3
"""Score completed MATH baselines on fixed shards and select a minimax shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


PREFERRED_BASELINE_ORDER = ("mad", "agentverse", "gptswarm", "aflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute each MATH baseline's EM on every manifest shard and select "
            "argmin_shard(max_baseline(EM))."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME=JSONL",
        help="Explicit baseline result; repeat for multiple baselines. Auto-discovers run-root otherwise.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_shards(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Manifest has no shards: {manifest_path}")

    normalized = []
    all_ids: list[str] = []
    for item in shards:
        shard_id = int(item["shard_id"])
        problem_ids = [str(value) for value in item["problem_ids"]]
        expected = int(item["count"])
        if len(problem_ids) != expected or len(problem_ids) != len(set(problem_ids)):
            raise ValueError(f"Invalid problem IDs for shard {shard_id}")
        normalized.append(
            {
                "shard_id": shard_id,
                "file": str(item["file"]),
                "count": expected,
                "problem_ids": problem_ids,
            }
        )
        all_ids.extend(problem_ids)

    normalized.sort(key=lambda item: item["shard_id"])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Manifest shards contain duplicate problem IDs")
    if int(manifest.get("total_count", len(all_ids))) != len(all_ids):
        raise ValueError("Manifest total_count does not match shard problem IDs")
    return manifest, normalized, set(all_ids)


def discover_baselines(run_root: Path) -> list[tuple[str, Path]]:
    candidates = {
        path.parent.name: path
        for path in run_root.glob("*/trajectories.jsonl")
        if path.is_file()
    }
    if not candidates:
        raise ValueError(f"No */trajectories.jsonl files found under {run_root}")
    ordered_names = [name for name in PREFERRED_BASELINE_ORDER if name in candidates]
    ordered_names.extend(sorted(set(candidates) - set(ordered_names)))
    return [(name, candidates[name]) for name in ordered_names]


def parse_baseline_specs(specs: list[str], run_root: Path) -> list[tuple[str, Path]]:
    if not specs:
        return discover_baselines(run_root)
    parsed = []
    seen = set()
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"Invalid --baseline value {spec!r}; expected NAME=JSONL")
        if name in seen:
            raise ValueError(f"Duplicate baseline name: {name}")
        path = Path(raw_path.strip()).expanduser().resolve()
        parsed.append((name, path))
        seen.add(name)
    return parsed


def load_baseline(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing baseline JSONL: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                problem_id = str(item["problem_id"])
                em = float(item["em"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid row at {path}:{line_number}: {exc}") from exc
            if problem_id in rows:
                raise ValueError(f"Duplicate problem_id {problem_id!r} in {path}")
            if not 0.0 <= em <= 1.0:
                raise ValueError(f"Invalid EM {em} for {problem_id!r} in {path}")
            rows[problem_id] = {"em": em, "error": bool(item.get("error"))}

    actual_ids = set(rows)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Problem ID coverage mismatch for {path}: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return rows


def summarize(
    manifest_path: Path,
    baseline_specs: list[tuple[str, Path]],
) -> dict[str, Any]:
    manifest, shards, expected_ids = load_shards(manifest_path)
    if not baseline_specs:
        raise ValueError("At least one baseline is required")

    baseline_rows = {
        name: load_baseline(path, expected_ids)
        for name, path in baseline_specs
    }
    baseline_summaries = []
    for name, path in baseline_specs:
        rows = baseline_rows[name]
        correct = sum(item["em"] for item in rows.values())
        baseline_summaries.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "count": len(rows),
                "correct": int(correct) if correct.is_integer() else correct,
                "em": correct / len(rows),
                "errors": sum(item["error"] for item in rows.values()),
            }
        )

    shard_summaries = []
    for shard in shards:
        scores = {}
        for name, _ in baseline_specs:
            rows = baseline_rows[name]
            selected = [rows[problem_id] for problem_id in shard["problem_ids"]]
            correct = sum(item["em"] for item in selected)
            scores[name] = {
                "correct": int(correct) if correct.is_integer() else correct,
                "em": correct / shard["count"],
                "errors": sum(item["error"] for item in selected),
            }
        maximum = max(score["em"] for score in scores.values())
        maximizers = [name for name, score in scores.items() if score["em"] == maximum]
        shard_summaries.append(
            {
                "shard_id": shard["shard_id"],
                "file": shard["file"],
                "count": shard["count"],
                "scores": scores,
                "max_baseline_em": maximum,
                "max_baselines": maximizers,
                "mean_baseline_em": mean(score["em"] for score in scores.values()),
            }
        )

    minimax_value = min(shard["max_baseline_em"] for shard in shard_summaries)
    minimax_candidates = [
        shard["shard_id"]
        for shard in shard_summaries
        if shard["max_baseline_em"] == minimax_value
    ]
    selected = min(
        (shard for shard in shard_summaries if shard["shard_id"] in minimax_candidates),
        key=lambda shard: (shard["mean_baseline_em"], shard["shard_id"]),
    )

    return {
        "schema_version": 1,
        "selection_rule": "min(max_baseline_em), then min(mean_baseline_em), then min(shard_id)",
        "manifest": str(manifest_path.resolve()),
        "split_seed": manifest.get("seed"),
        "num_shards": len(shards),
        "shard_size": shards[0]["count"],
        "baselines": baseline_summaries,
        "shards": shard_summaries,
        "minimax_value": minimax_value,
        "minimax_candidates": minimax_candidates,
        "selected_shard_id": selected["shard_id"],
        "selected_shard_file": selected["file"],
    }


def print_report(summary: dict[str, Any]) -> None:
    names = [item["name"] for item in summary["baselines"]]
    print("Full-test baseline scores:")
    for item in summary["baselines"]:
        print(
            f"  {item['name']}: {item['correct']}/{item['count']} "
            f"EM={item['em']:.4f} errors={item['errors']}"
        )

    headers = ["shard", *names, "max", "mean", "max_by"]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for shard in summary["shards"]:
        values = [f"{100 * shard['scores'][name]['em']:.2f}%" for name in names]
        print(
            " | ".join(
                [
                    f"{shard['shard_id']:02d}",
                    *values,
                    f"{100 * shard['max_baseline_em']:.2f}%",
                    f"{100 * shard['mean_baseline_em']:.2f}%",
                    ",".join(shard["max_baselines"]),
                ]
            )
        )
    print(
        f"\nSelected shard_{summary['selected_shard_id']:02d}: "
        f"minimax={100 * summary['minimax_value']:.2f}% "
        f"candidates={summary['minimax_candidates']}"
    )


def main() -> None:
    args = parse_args()
    baseline_specs = parse_baseline_specs(args.baseline, args.run_root.resolve())
    summary = summarize(args.manifest.resolve(), baseline_specs)
    print_report(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output.chmod(0o644)
        print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
