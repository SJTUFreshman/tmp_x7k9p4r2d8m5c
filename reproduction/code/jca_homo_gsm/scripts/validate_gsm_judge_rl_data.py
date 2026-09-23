"""Validate GSM LLM-judge rollout data before offline RWR training."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


AGENTS = ("A1", "A2", "A3")
GroupKey = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-problems", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--min-records-per-agent", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--task-metric", choices=["em", "f1"], default="em")
    parser.add_argument(
        "--expected-start-agent",
        choices=[*AGENTS, "balanced"],
        default="balanced",
        help="Required trajectory start-agent policy.",
    )
    parser.add_argument("--require-reward-polarity", action="store_true")
    parser.add_argument("--stats-output", type=Path, default=None)
    return parser.parse_args()


def close_enough(left: float, right: float, tolerance: float = 1.1e-4) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    args = parse_args()
    if args.expected_problems <= 0 or args.num_rollouts <= 0:
        raise SystemExit("expected problem and rollout counts must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if not args.input.is_file():
        raise SystemExit(f"rollout file not found: {args.input}")

    groups: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    agent_counts: Counter[str] = Counter()
    action_counts: Dict[str, Counter[str]] = {
        agent: Counter() for agent in AGENTS
    }
    start_counts: Counter[str] = Counter()
    rewards: Dict[str, List[float]] = {agent: [] for agent in AGENTS}
    failures: List[str] = []
    row_count = 0

    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"line {line_no}: invalid JSON: {exc}")
                continue

            try:
                problem_id = str(record["problem_id"])
                rollout_idx = int(record["rollout_idx"])
                turn = int(record["turn"])
                agent = str(record["agent_id"])
                reward = float(record["reward"])
                task_reward = float(record["task_reward"])
                judge_score = float(record["judge_score"])
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"line {line_no}: missing or invalid core field: {exc}")
                continue

            if agent not in AGENTS:
                failures.append(f"line {line_no}: invalid agent_id={agent!r}")
                continue
            if not all(math.isfinite(value) for value in (reward, task_reward, judge_score)):
                failures.append(f"line {line_no}: non-finite reward value")
            if not -1.0001 <= reward <= 1.0001:
                failures.append(f"line {line_no}: reward outside [-1, 1]: {reward}")
            if record.get("judge_status") != "scored" or record.get("judge_failed") is not False:
                failures.append(f"line {line_no}: judge did not score this turn")
            if record.get("task_metric") != args.task_metric:
                failures.append(
                    f"line {line_no}: task_metric={record.get('task_metric')!r}, "
                    f"expected {args.task_metric!r}"
                )

            metric_value = float(record.get(args.task_metric, 0.0) or 0.0)
            expected_task_reward = 2.0 * metric_value - 1.0
            expected_reward = (
                args.alpha * expected_task_reward
                + (1.0 - args.alpha) * judge_score
            )
            if not close_enough(task_reward, round(expected_task_reward, 4)):
                failures.append(f"line {line_no}: task_reward formula mismatch")
            if not close_enough(reward, round(expected_reward, 4)):
                failures.append(f"line {line_no}: mixed reward formula mismatch")

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                failures.append(f"line {line_no}: messages must contain system and user")
            response = record.get("response")
            try:
                response_obj = json.loads(response)
            except (TypeError, json.JSONDecodeError):
                failures.append(f"line {line_no}: response is not normalized JSON")
                response_obj = {}
            if response_obj.get("action") != record.get("action"):
                failures.append(f"line {line_no}: response/action mismatch")

            scores = record.get("turn_scores")
            if not isinstance(scores, dict) or not scores:
                failures.append(f"line {line_no}: missing turn_scores")
            elif not close_enough(float(scores.get("judge_score", 99)), judge_score):
                failures.append(f"line {line_no}: turn_scores/judge_score mismatch")

            groups[(problem_id, rollout_idx)].append(record)
            agent_counts[agent] += 1
            action_counts[agent][str(record.get("action"))] += 1
            rewards[agent].append(reward)

    expected_groups = args.expected_problems * args.num_rollouts
    if len(groups) != expected_groups:
        failures.append(
            f"trajectory groups={len(groups)}, expected={expected_groups}"
        )

    for key, records in groups.items():
        turns = sorted(int(record.get("turn", -1)) for record in records)
        if turns != list(range(len(records))):
            failures.append(f"group {key}: turns are not unique and contiguous: {turns}")
        group_starts = {str(record.get("start_agent")) for record in records}
        if len(group_starts) != 1 or not group_starts.issubset(AGENTS):
            failures.append(f"group {key}: inconsistent start_agent={group_starts}")
        else:
            declared_start = next(iter(group_starts))
            start_counts[declared_start] += 1
            first_records = [
                record for record in records if int(record.get("turn", -1)) == 0
            ]
            if len(first_records) == 1:
                actual_start = str(first_records[0].get("agent_id"))
                if actual_start != declared_start:
                    failures.append(
                        f"group {key}: turn-0 agent={actual_start!r} does not match "
                        f"start_agent={declared_start!r}"
                    )

    if args.expected_start_agent == "balanced":
        if set(start_counts) != set(AGENTS):
            failures.append(f"not every agent appears as a start agent: {dict(start_counts)}")
        elif max(start_counts.values()) - min(start_counts.values()) > 1:
            failures.append(f"start-agent routing is not balanced: {dict(start_counts)}")
    elif start_counts != Counter({args.expected_start_agent: expected_groups}):
        failures.append(
            f"start-agent routing must be fixed to {args.expected_start_agent}: "
            f"{dict(start_counts)}"
        )

    minimum = args.min_records_per_agent or expected_groups
    for agent in AGENTS:
        if agent_counts[agent] < minimum:
            failures.append(
                f"{agent} has {agent_counts[agent]} records, needs at least {minimum}"
            )
        if args.require_reward_polarity:
            if not any(value > 0 for value in rewards[agent]):
                failures.append(f"{agent} has no positive-reward records")
            if not any(value < 0 for value in rewards[agent]):
                failures.append(f"{agent} has no negative-reward records")

    stats = {
        "passed": not failures,
        "input": str(args.input),
        "rows": row_count,
        "groups": len(groups),
        "expected_groups": expected_groups,
        "task_metric": args.task_metric,
        "alpha": args.alpha,
        "expected_start_agent": args.expected_start_agent,
        "agent_counts": dict(agent_counts),
        "start_agent_counts": dict(start_counts),
        "action_counts": {
            agent: dict(action_counts[agent]) for agent in AGENTS
        },
        "reward_stats": {
            agent: {
                "min": min(rewards[agent]) if rewards[agent] else None,
                "max": max(rewards[agent]) if rewards[agent] else None,
                "mean": (
                    sum(rewards[agent]) / len(rewards[agent])
                    if rewards[agent]
                    else None
                ),
            }
            for agent in AGENTS
        },
        "failures": failures[:100],
    }
    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(
            f"GSM judge RL data gate failed with {len(failures)} issue(s)"
        )
    print("GSM judge RL data gate: PASS")


if __name__ == "__main__":
    main()
