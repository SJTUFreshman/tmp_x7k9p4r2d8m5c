#!/usr/bin/env python3
"""Build per-agent protocol SFT data from scored Conifer rollouts."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_protocol"))
from conifer_protocol import AGENT_IDS, label_from_step, validate_label  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _response_from_record(record: dict[str, Any]) -> str:
    if isinstance(record.get("response"), str) and record["response"].strip():
        return record["response"]
    step = record.get("step") or {}
    return label_from_step(step)


def _candidate(
    record: dict[str, Any],
    min_hard_score: float,
    allow_judge_failed: bool,
    *,
    source_policy: str,
    require_source_policy: bool,
    require_teacher_3x8b: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    agent = str(record.get("agent_id") or "")
    if agent not in AGENT_IDS:
        return None, "bad_agent"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "missing_messages"
    sampling = record.get("sampling") if isinstance(record.get("sampling"), dict) else {}
    actual_policy = str(record.get("source_policy") or sampling.get("source_policy") or "")
    if require_source_policy and actual_policy != source_policy:
        return None, "source_policy_mismatch"
    if require_teacher_3x8b:
        model_paths = sampling.get("model_paths")
        if not isinstance(model_paths, dict) or set(model_paths) != set(AGENT_IDS):
            return None, "missing_teacher_model_metadata"
        model_names = {str(model_paths[agent]).rstrip("/").rsplit("/", 1)[-1] for agent in AGENT_IDS}
        if any(not name.startswith("Qwen3-8B") for name in model_names) or len(model_names) != 1:
            return None, "not_3x8b_teacher_rollout"
    response = _response_from_record(record)
    valid, reason = validate_label(response, active_agent=agent)
    if not valid:
        return None, "invalid_protocol:" + reason
    if not allow_judge_failed and str(record.get("judge_status")) == "judge_failed":
        return None, "judge_failed"
    try:
        hard_score = float(record.get("hard_score", 0.0))
    except (TypeError, ValueError):
        hard_score = 0.0
    if hard_score < min_hard_score:
        return None, "low_hard_score"
    action = str(record.get("action") or "")
    if action not in {"handoff", "confirm_stop"}:
        try:
            action = json.loads(response).get("action", "")
        except Exception:
            pass
    trajectory_id = str(record.get("trajectory_id") or f"{record.get('problem_id')}::{record.get('rollout_idx', 0)}")
    output = {
        "schema_version": 1,
        # Use the normalized seed group as problem_id for the trainer's
        # grouped split; retain the original row identity separately.
        "problem_id": record.get("group_id") or record.get("problem_id"),
        "source_problem_id": record.get("problem_id"),
        "group_id": record.get("group_id") or record.get("problem_id"),
        "trajectory_id": trajectory_id,
        "rollout_idx": record.get("rollout_idx", 0),
        "turn": record.get("turn", 0),
        "agent_id": agent,
        "start_agent": record.get("start_agent"),
        "action": action,
        "messages": messages,
        "label": response,
        "hard_score": round(hard_score, 5),
        "task_score": record.get("task_score"),
        "judge_status": record.get("judge_status"),
        "source": "conifer_scored_rollout",
        "source_policy": actual_policy or source_policy or "unknown",
        "teacher_rollout": bool(record.get("teacher_rollout", sampling.get("teacher_rollout", False))),
        "sampling": sampling,
    }
    return output, None


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="rl-output from score_conifer_rollouts.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-hard-score", type=float, default=0.35)
    parser.add_argument("--allow-judge-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-per-agent", type=int, default=0)
    parser.add_argument("--balance-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-confirm-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-policy", default="", help="Expected rollout policy identifier.")
    parser.add_argument("--require-source-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-teacher-3x8b", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-distinct-agents", type=int, default=1)
    parser.add_argument("--min-handoffs", type=int, default=0)
    parser.add_argument("--allow-empty-agent", action="store_true", help="Allow a tiny smoke run to omit one role.")
    args = parser.parse_args()
    if not 0.0 <= args.min_hard_score <= 1.0:
        raise SystemExit("--min-hard-score must be in [0,1]")
    if not 0.0 <= args.max_confirm_ratio <= 1.0:
        raise SystemExit("--max-confirm-ratio must be in [0,1]")
    if args.require_source_policy and not args.source_policy:
        raise SystemExit("--source-policy is required with --require-source-policy")
    if not 1 <= args.min_distinct_agents <= len(AGENT_IDS):
        raise SystemExit(f"--min-distinct-agents must be in [1,{len(AGENT_IDS)}]")
    if args.min_handoffs < 0:
        raise SystemExit("--min-handoffs must be non-negative")

    records = read_jsonl(args.input)
    accepted: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    seen: set[tuple[str, int, str]] = set()
    protocol_by_trajectory: dict[str, dict[str, Any]] = {}
    for record in records:
        trajectory_id = str(record.get("trajectory_id") or f"{record.get('problem_id')}::{record.get('rollout_idx', 0)}")
        info = protocol_by_trajectory.setdefault(trajectory_id, {"agents": set(), "handoffs": 0})
        agent = str(record.get("agent_id") or "")
        if agent in AGENT_IDS:
            info["agents"].add(agent)
        action = str(record.get("action") or "")
        if action not in {"handoff", "confirm_stop"}:
            try:
                parsed_response = json.loads(_response_from_record(record))
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed_response = {}
            action = str(parsed_response.get("action") or "")
        if action == "handoff":
            info["handoffs"] += 1
        candidate, reason = _candidate(
            record,
            args.min_hard_score,
            args.allow_judge_failed,
            source_policy=args.source_policy,
            require_source_policy=args.require_source_policy,
            require_teacher_3x8b=args.require_teacher_3x8b,
        )
        if candidate is None:
            drops[reason or "unknown"] += 1
            continue
        key = (str(candidate["trajectory_id"]), int(candidate.get("turn", 0)), str(candidate["agent_id"]))
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append(candidate)

    eligible_trajectories = {
        trajectory_id
        for trajectory_id, info in protocol_by_trajectory.items()
        if len(info["agents"]) >= args.min_distinct_agents and info["handoffs"] >= args.min_handoffs
    }
    if args.min_distinct_agents > 1 or args.min_handoffs > 0:
        filtered = []
        for row in accepted:
            if str(row["trajectory_id"]) in eligible_trajectories:
                filtered.append(row)
            else:
                drops["insufficient_collaboration"] += 1
        accepted = filtered

    rng = random.Random(args.seed)
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENT_IDS}
    for row in accepted:
        by_agent[row["agent_id"]].append(row)
    for agent in AGENT_IDS:
        rows = by_agent[agent]
        rng.shuffle(rows)
        if args.balance_actions and rows:
            handoffs = [row for row in rows if row["action"] == "handoff"]
            confirms = [row for row in rows if row["action"] == "confirm_stop"]
            max_confirms = int(len(rows) * args.max_confirm_ratio)
            if confirms and args.max_confirm_ratio > 0:
                max_confirms = max(1, max_confirms)
            if len(confirms) > max_confirms:
                rng.shuffle(confirms)
                rows = handoffs + confirms[:max_confirms]
                rng.shuffle(rows)
        if args.max_per_agent > 0:
            rows = rows[: args.max_per_agent]
        by_agent[agent] = rows

    all_rows = [row for agent in AGENT_IDS for row in by_agent[agent]]
    start_by_trajectory: dict[str, str] = {}
    conflicting_start_trajectories: set[str] = set()
    for row in all_rows:
        trajectory_id = str(row["trajectory_id"])
        start_agent = str(row.get("start_agent") or "unknown")
        previous_start = start_by_trajectory.get(trajectory_id)
        if previous_start is not None and previous_start != start_agent:
            conflicting_start_trajectories.add(trajectory_id)
        else:
            start_by_trajectory[trajectory_id] = start_agent
    for agent in AGENT_IDS:
        write_jsonl(args.out_dir / f"{agent}.jsonl", by_agent[agent])
    write_jsonl(args.out_dir / "all_agents.jsonl", all_rows)
    stats = {
        "input_records": len(records),
        "accepted_records": len(all_rows),
        "accepted_by_agent": {agent: len(by_agent[agent]) for agent in AGENT_IDS},
        "action_counts": {agent: dict(Counter(row["action"] for row in by_agent[agent])) for agent in AGENT_IDS},
        "unique_trajectories": len({row["trajectory_id"] for row in all_rows}),
        "start_agent_trajectory_counts": dict(Counter(start_by_trajectory.values())),
        "conflicting_start_agent_trajectories": len(conflicting_start_trajectories),
        "min_hard_score": args.min_hard_score,
        "max_confirm_ratio": args.max_confirm_ratio,
        "source_policy": args.source_policy or "unknown",
        "require_source_policy": args.require_source_policy,
        "require_teacher_3x8b": args.require_teacher_3x8b,
        "collaboration_gate": {
            "min_distinct_agents": args.min_distinct_agents,
            "min_handoffs": args.min_handoffs,
            "eligible_trajectories": len(eligible_trajectories),
        },
        "drops": dict(drops),
        "seed": args.seed,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not args.allow_empty_agent and any(len(by_agent[agent]) < 1 for agent in AGENT_IDS):
        raise SystemExit("SFT quality gate failed: at least one agent has no examples")


if __name__ == "__main__":
    main()
