#!/usr/bin/env python3
"""Summarize AgentVerse protocol diagnostics from trajectory JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize AgentVerse traces.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.trajectories.open(encoding="utf-8")
        if line.strip()
    ]
    if not rows:
        raise SystemExit("No AgentVerse trajectories found")
    iterations = [item for row in rows for item in row.get("iterations", [])]
    answers = [answer for item in iterations for answer in item.get("answers", [])]
    evaluations = [item.get("evaluation", {}) for item in iterations]
    termination = Counter(str(row.get("terminated_by")) for row in rows)
    winners = Counter(str(row.get("winning_agent")) for row in rows)
    summary = {
        "schema_version": 1,
        "problems": len(rows),
        "average_iterations": statistics.fmean(
            len(row.get("iterations", [])) for row in rows
        ),
        "early_stop_count": termination["score_threshold"],
        "early_stop_rate": termination["score_threshold"] / len(rows),
        "termination_counts": dict(sorted(termination.items())),
        "recruit_retry_count": sum(bool(row.get("recruit_retried")) for row in rows),
        "recruit_failure_count": sum(not bool(row.get("recruit_parse_ok")) for row in rows),
        "agent_retry_count": sum(bool(answer.get("retried")) for answer in answers),
        "agent_parse_failure_count": sum(not bool(answer.get("parse_ok")) for answer in answers),
        "evaluator_retry_count": sum(bool(item.get("retried")) for item in evaluations),
        "evaluator_parse_failure_count": sum(not bool(item.get("parse_ok")) for item in evaluations),
        "iteration_tie_break_count": sum(bool(item.get("tie_break")) for item in iterations),
        "iteration_tie_break_rate": (
            statistics.fmean(bool(item.get("tie_break")) for item in iterations)
            if iterations else 0.0
        ),
        "final_tie_break_count": sum(bool(row.get("tie_break")) for row in rows),
        "final_tie_break_rate": statistics.fmean(bool(row.get("tie_break")) for row in rows),
        "winning_agent_counts": dict(sorted(winners.items())),
        "normalization_error_count": sum(
            len(item.get("normalization_errors", {})) for item in iterations
        ),
        "average_wall_time_s": statistics.fmean(
            float(row.get("wall_time_s", 0.0)) for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = (
        "===== AgentVerse protocol diagnostics =====\n"
        f"Problems: {summary['problems']}\n"
        f"Average iterations: {summary['average_iterations']:.6f}\n"
        f"Early stops: {summary['early_stop_count']} ({summary['early_stop_rate']:.6f})\n"
        f"Termination: {summary['termination_counts']}\n"
        f"Recruit retries/failures: {summary['recruit_retry_count']}/{summary['recruit_failure_count']}\n"
        f"Agent retries/parse failures: {summary['agent_retry_count']}/{summary['agent_parse_failure_count']}\n"
        f"Evaluator retries/parse failures: {summary['evaluator_retry_count']}/{summary['evaluator_parse_failure_count']}\n"
        f"Iteration tie breaks: {summary['iteration_tie_break_count']} ({summary['iteration_tie_break_rate']:.6f})\n"
        f"Final tie breaks: {summary['final_tie_break_count']} ({summary['final_tie_break_rate']:.6f})\n"
        f"Winning agents: {summary['winning_agent_counts']}\n"
        f"Normalization errors: {summary['normalization_error_count']}\n"
        f"Average wall time/problem: {summary['average_wall_time_s']:.3f}s\n"
    )
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
