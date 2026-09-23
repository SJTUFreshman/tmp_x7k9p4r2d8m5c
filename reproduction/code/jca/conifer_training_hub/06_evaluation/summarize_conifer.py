#!/usr/bin/env python3
"""Summarize Conifer trajectories with quality, routing, and cost metrics."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 5) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trajectories": 0}
    turns = [row.get("trajectory", {}).get("steps", []) for row in rows]
    final_checks = [row.get("final_checks") or {} for row in rows]
    hard = [float(row.get("hard_score", checks.get("hard_score", 0.0)) or 0.0) for row, checks in zip(rows, final_checks)]
    explicit = [float(checks.get("explicit_score", 0.0) or 0.0) for checks in final_checks]
    coverage = [float(checks.get("requirement_coverage", 0.0) or 0.0) for checks in final_checks]
    # check_constraints already computes this per problem; it was simply never
    # rolled up.  It is the only rule metric that does not saturate, so it is
    # the one that can still tell two methods apart.
    lexical = [
        float(checks["reference_lexical_f1"]) for checks in final_checks
        if isinstance(checks.get("reference_lexical_f1"), (int, float))
    ]
    stop = [row.get("trajectory", {}).get("terminated_by") == "stop" for row in rows]
    handoffs = [sum(step.get("action") == "handoff" for step in seq) for seq in turns]
    active = [len({step.get("active_agent") for step in seq if step.get("active_agent")}) for seq in turns]
    judge_quality = [float(row.get("judge", {}).get("final_quality")) for row in rows if row.get("judge", {}).get("status") == "llm_scored"]
    source_type = defaultdict(list)
    difficulty = defaultdict(list)
    for row, score in zip(rows, hard):
        problem = row.get("problem") or {}
        source_type[str(problem.get("source_type", "unknown"))].append(score)
        difficulty[str(problem.get("difficulty", "unknown"))].append(score)
    return {
        "trajectories": len(rows),
        "start_agent_counts": dict(Counter(
            str(row.get("start_agent") or row.get("trajectory", {}).get("start_agent") or "unknown")
            for row in rows
        )),
        "stopped": sum(stop),
        "stop_rate": round(sum(stop) / len(rows), 5),
        "mean_turns": mean([len(seq) for seq in turns]),
        "mean_handoffs": mean(handoffs),
        "mean_active_agents": mean(active),
        "mean_final_hard_score": mean(hard),
        "mean_final_explicit_score": mean(explicit),
        "mean_requirement_coverage": mean(coverage),
        "mean_reference_lexical_f1": mean(lexical),
        "lexical_scored_trajectories": len(lexical),
        "all_explicit_pass_rate": round(sum(bool(checks.get("all_explicit_passed")) for checks in final_checks) / len(rows), 5),
        "mean_judge_final_quality": mean(judge_quality),
        "judge_scored_trajectories": len(judge_quality),
        "source_type_hard_score": {key: mean(value) for key, value in sorted(source_type.items())},
        "difficulty_hard_score": {key: mean(value) for key, value in sorted(difficulty.items())},
        "action_counts": dict(Counter(step.get("action") for seq in turns for step in seq)),
        "agent_turn_counts": dict(Counter(step.get("active_agent") for seq in turns for step in seq)),
        "terminated_by": dict(Counter(row.get("trajectory", {}).get("terminated_by") for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports: dict[str, Any] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be LABEL=PATH, got {spec!r}")
        label, value = spec.split("=", 1)
        path = Path(value)
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
        loaded[label] = read(path)
        reports[label] = summarize(loaded[label])
    # Paired final-score deltas make ablations interpretable when keys overlap.
    labels = list(loaded)
    if len(labels) >= 2:
        base_label = labels[0]
        base = {(str(r.get("problem_id")), int(r.get("rollout_idx", 0))): float(r.get("hard_score", 0.0) or 0.0) for r in loaded[base_label]}
        reports["paired_deltas_vs_first"] = {}
        for label in labels[1:]:
            pairs = []
            for row in loaded[label]:
                key = (str(row.get("problem_id")), int(row.get("rollout_idx", 0)))
                if key in base:
                    pairs.append(float(row.get("hard_score", 0.0) or 0.0) - base[key])
            reports["paired_deltas_vs_first"][label] = {"n": len(pairs), "mean_hard_score_delta": mean(pairs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
