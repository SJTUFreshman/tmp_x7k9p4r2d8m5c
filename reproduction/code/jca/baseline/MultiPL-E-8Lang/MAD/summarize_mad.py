#!/usr/bin/env python3
"""Summarize MAD protocol diagnostics from trajectory JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MultiPL-E MAD traces.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.trajectories.open(encoding="utf-8")
        if line.strip()
    ]
    if not rows:
        raise SystemExit("No MAD trajectories found")
    turns = [turn for row in rows for turn in row.get("turns", [])]
    winner_counts = Counter(str(row.get("winning_agent")) for row in rows)
    termination_counts = Counter(str(row.get("terminated_by")) for row in rows)
    summary = {
        "schema_version": 1,
        "problems": len(rows),
        "tie_break_count": sum(bool(row.get("tie_break")) for row in rows),
        "tie_break_rate": statistics.fmean(
            bool(row.get("tie_break")) for row in rows
        ),
        "winning_agent_counts": dict(sorted(winner_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "turns": len(turns),
        "parse_failure_count": sum(not bool(turn.get("parse_ok")) for turn in turns),
        "retry_count": sum(bool(turn.get("retried")) for turn in turns),
        "normalization_error_count": sum(
            len(row.get("normalization_errors", {})) for row in rows
        ),
        "average_wall_time_s": statistics.fmean(
            float(row.get("wall_time_s", 0.0)) for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text = (
        "===== MAD protocol diagnostics =====\n"
        f"Problems: {summary['problems']}\n"
        f"Tie breaks: {summary['tie_break_count']} "
        f"({summary['tie_break_rate']:.6f})\n"
        f"Winning agents: {summary['winning_agent_counts']}\n"
        f"Termination: {summary['termination_counts']}\n"
        f"Turns: {summary['turns']}\n"
        f"Parse failures: {summary['parse_failure_count']}\n"
        f"Retries: {summary['retry_count']}\n"
        f"Normalization errors: {summary['normalization_error_count']}\n"
        f"Average wall time/problem: {summary['average_wall_time_s']:.3f}s\n"
    )
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
