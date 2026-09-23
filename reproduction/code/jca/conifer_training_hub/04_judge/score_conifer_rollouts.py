#!/usr/bin/env python3
"""Score Conifer trajectories and emit per-turn RL records.

Outputs:
  * ``--scored-output``: one enriched row per trajectory
  * ``--rl-output``: one row per agent turn, compatible with the project's
    generic signed RWR trainer

The deterministic path is always run.  LLM judging is optional and cached by
the trajectory key, so a failed network call never destroys a completed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "02_protocol", ROOT / "04_judge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conifer_scoring import (  # noqa: E402
    JudgeEndpointPool,
    build_judge_prompt,
    call_judge,
    check_constraints,
    normalize_judge_result,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _judge_cache_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    payload = {
        "problem_id": row.get("problem_id"),
        "rollout_idx": row.get("rollout_idx", 0),
        "trajectory": row.get("trajectory") or {},
        "final_answer": row.get("final_answer"),
        "judge_model": args.judge_model,
        "judge_api_base": args.judge_api_base,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _judge_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    final_answer = str(row.get("final_answer") or trajectory.get("final_answer") or "")
    final_checks = check_constraints(row.get("problem") or {}, final_answer)
    result: dict[str, Any] = {
        "status": "deterministic_only",
        "final_quality": float(final_checks["hard_score"]),
        "turns": [],
    }
    if args.judge_mode == "llm" and steps:
        try:
            raw = call_judge(
                build_judge_prompt(row.get("problem") or {}, steps, final_answer, final_checks),
                model=args.judge_model,
                api_base=args.judge_api_base,
                api_key=args.judge_api_key,
                timeout=args.judge_timeout,
                max_tokens=args.judge_max_tokens,
                length_retries=args.judge_length_retries,
                endpoint_pool=getattr(args, "judge_endpoint_pool", None),
            )
            result = normalize_judge_result(raw, len(steps))
            if args.require_complete_judge and not result.get("complete", False):
                raise ValueError(f"judge omitted turns: {result.get('missing_turns')}")
            result["status"] = "llm_scored"
        except Exception as exc:
            result = {
                "status": "judge_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "final_quality": float(final_checks["hard_score"]),
                "turns": [],
            }
            if not args.allow_judge_failure:
                raise
    return result


def _make_rl_records(row: dict[str, Any], judge: dict[str, Any], alpha: float) -> list[dict[str, Any]]:
    problem = row.get("problem") or {}
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    final_checks = row.get("final_checks") or check_constraints(problem, row.get("final_answer", ""))
    hard_score = float(final_checks.get("hard_score", row.get("hard_score", 0.0)))
    judge_status = str(judge.get("status", "deterministic_only"))
    semantic_quality = float(judge.get("final_quality", hard_score)) if judge_status == "llm_scored" else hard_score
    # Explicit constraints remain the anchor; the judge supplies semantic
    # quality only when requested.  Reference overlap is diagnostic, not a
    # correctness label, because open answers are non-unique.
    task_score = hard_score if judge_status != "llm_scored" else 0.55 * hard_score + 0.45 * semantic_quality
    task_reward = 2.0 * max(0.0, min(1.0, task_score)) - 1.0
    turn_judges = {int(item["turn"]): item for item in judge.get("turns") or [] if isinstance(item, dict) and str(item.get("turn", "")).lstrip("-").isdigit()}
    records: list[dict[str, Any]] = []
    previous_hard = 0.0
    turn_messages = row.get("turn_messages") or []
    sampling = row.get("sampling") or {}
    for position, step in enumerate(steps):
        turn = int(step.get("turn", position))
        answer = str(step.get("confirmed_answer") or step.get("tentative_answer") or "")
        checks = step.get("hard_checks") or check_constraints(problem, answer)
        current_hard = float(checks.get("hard_score", 0.0))
        transition_gain = current_hard - previous_hard
        previous_hard = current_hard
        score = turn_judges.get(turn) or {}
        reasoning_score = float(score.get("reasoning_score", 0.0))
        action_score = float(score.get("action_score", 0.0))
        content_score = float(score.get("content_score", 0.0))
        process_score = 0.4 * reasoning_score + 0.4 * action_score + 0.2 * content_score
        mixed_process = 0.7 * process_score + 0.3 * max(-1.0, min(1.0, transition_gain))
        reward = alpha * task_reward + (1.0 - alpha) * mixed_process
        response = json.dumps({
            "reasoning": str(step.get("reasoning") or "").strip(),
            "tentative_answer": str(step.get("tentative_answer") or "").strip(),
            "action": str(step.get("action") or ""),
            "handoff_target": step.get("handoff_target"),
            "handoff_note": step.get("handoff_note"),
            "confirmed_answer": step.get("confirmed_answer"),
        }, ensure_ascii=False)
        records.append({
            "schema_version": 1,
            "problem_id": row.get("problem_id"),
            "group_id": row.get("group_id"),
            "rollout_idx": row.get("rollout_idx", 0),
            "trajectory_id": f"{row.get('problem_id')}::{row.get('rollout_idx', 0)}",
            "turn": turn,
            "agent_id": step.get("active_agent"),
            "start_agent": row.get("start_agent"),
            "action": step.get("action"),
            "handoff_target": step.get("handoff_target"),
            "sampling": sampling,
            "source_policy": row.get("source_policy") or sampling.get("source_policy"),
            "teacher_rollout": bool(row.get("teacher_rollout", sampling.get("teacher_rollout", False))),
            "messages": turn_messages[position] if position < len(turn_messages) else [],
            "response": response,
            "reward": round(reward, 5),
            "reward_no_process": round(task_reward, 5),
            "task_reward": round(task_reward, 5),
            "task_score": round(task_score, 5),
            "hard_score": round(current_hard, 5),
            "transition_gain": round(transition_gain, 5),
            "judge_score": round(process_score, 5),
            "reasoning_score": round(reasoning_score, 5),
            "action_score": round(action_score, 5),
            "content_score": round(content_score, 5),
            "turn_scores": score,
            "judge_status": judge_status,
            "final_answer": row.get("final_answer"),
            "final_checks": final_checks,
            "terminated_by": trajectory.get("terminated_by"),
            "problem": problem,
        })
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--scored-output", type=Path, required=True)
    p.add_argument("--rl-output", type=Path, required=True)
    p.add_argument("--judge-mode", choices=["deterministic", "llm"], default="deterministic")
    p.add_argument("--judge-model", default=os.environ.get("CONIFER_JUDGE_MODEL", "gpt-5"))
    p.add_argument("--judge-api-base", default=os.environ.get("JCA_JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--judge-api-key", default=os.environ.get("JCA_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--judge-timeout", type=float, default=300.0)
    p.add_argument("--judge-concurrency", type=int, default=512)
    p.add_argument("--judge-max-tokens", type=int, default=1536)
    p.add_argument("--judge-length-retries", type=int, default=1)
    p.add_argument("--allow-judge-failure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-complete-judge", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-judge-failure-rate", type=float, default=1.0, help="Fail after writing resumable outputs if judge failures exceed this fraction.")
    p.add_argument("--judge-cache", type=Path, default=None, help="JSON cache for LLM judge results; derived from output when omitted.")
    p.add_argument("--judge-cache-save-every", type=int, default=256, help="Persist successful new judge results every N trajectories.")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-rows", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0,1]")
    if args.judge_cache_save_every <= 0:
        raise SystemExit("--judge-cache-save-every must be positive")
    if args.judge_concurrency <= 0 or args.judge_max_tokens <= 0 or args.judge_length_retries < 0:
        raise SystemExit("judge concurrency/token budget must be positive and retries non-negative")
    if not 0.0 <= args.max_judge_failure_rate <= 1.0:
        raise SystemExit("--max-judge-failure-rate must be in [0,1]")
    rows = _read_jsonl(args.input)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("input has no rows")

    args.judge_endpoint_pool = None
    if args.judge_mode == "llm" and args.judge_api_base and "," in args.judge_api_base:
        args.judge_endpoint_pool = JudgeEndpointPool(args.judge_api_base)

    cache_path = args.judge_cache or args.scored_output.with_name(args.scored_output.stem + "_judge_cache.json")
    judge_cache: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        try:
            loaded_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded_cache, dict):
                judge_cache = {str(key): value for key, value in loaded_cache.items() if isinstance(value, dict)}
        except (OSError, json.JSONDecodeError):
            judge_cache = {}
    scored: list[dict[str, Any] | None] = [None] * len(rows)
    cache_hits = 0
    cache_misses = 0
    if args.judge_mode == "deterministic":
        for index, row in enumerate(rows):
            scored[index] = _judge_one(row, args)
    else:
        for index, row in enumerate(rows):
            key = _judge_cache_key(row, args)
            cached = judge_cache.get(key)
            if cached is not None:
                scored[index] = cached
                cache_hits += 1
            else:
                cache_misses += 1
        pending_cache_writes = 0
        worker_count = max(1, min(args.judge_concurrency, cache_misses))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            pending = iter(
                (index, row, _judge_cache_key(row, args))
                for index, row in enumerate(rows)
                if scored[index] is None
            )
            futures = {}
            for _ in range(worker_count):
                try:
                    index, row, key = next(pending)
                except StopIteration:
                    break
                futures[pool.submit(_judge_one, row, args)] = (index, key)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index, key = futures.pop(future)
                    scored[index] = future.result()
                    if scored[index].get("status") == "llm_scored":
                        judge_cache[key] = scored[index]
                        pending_cache_writes += 1
                        if pending_cache_writes >= args.judge_cache_save_every:
                            _write_cache(cache_path, judge_cache)
                            pending_cache_writes = 0
                    try:
                        next_index, next_row, next_key = next(pending)
                    except StopIteration:
                        continue
                    futures[pool.submit(_judge_one, next_row, args)] = (next_index, next_key)
        if cache_hits or cache_misses:
            _write_cache(cache_path, judge_cache)

    enriched: list[dict[str, Any]] = []
    rl_records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for row, judge in zip(rows, scored):
        assert judge is not None
        enriched_row = dict(row)
        enriched_row["judge"] = judge
        enriched_row["final_checks"] = check_constraints(row.get("problem") or {}, row.get("final_answer", ""))
        sampling = row.get("sampling") if isinstance(row.get("sampling"), dict) else {}
        enriched_row["source_policy"] = row.get("source_policy") or sampling.get("source_policy")
        enriched_row["teacher_rollout"] = bool(row.get("teacher_rollout", sampling.get("teacher_rollout", False)))
        enriched.append(enriched_row)
        rl_records.extend(_make_rl_records(enriched_row, judge, args.alpha))
        status = str(judge.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    _write_jsonl(args.scored_output, enriched)
    _write_jsonl(args.rl_output, rl_records)
    stats = {
        "trajectories": len(enriched),
        "turn_records": len(rl_records),
        "judge_mode": args.judge_mode,
        "status_counts": status_counts,
        "judge_cache": {"path": str(cache_path), "hits": cache_hits, "misses": cache_misses},
        "alpha": args.alpha,
        "reward_fields": ["reward", "reward_no_process", "task_reward"],
    }
    if args.judge_mode == "llm":
        stats["judge_failure_rate"] = round(status_counts.get("judge_failed", 0) / len(enriched), 5)
    stats_path = args.scored_output.with_name(args.scored_output.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.judge_mode == "llm":
        failure_rate = float(stats["judge_failure_rate"])
        if failure_rate > args.max_judge_failure_rate:
            raise SystemExit(
                f"judge failure rate {failure_rate:.4f} exceeds "
                f"--max-judge-failure-rate={args.max_judge_failure_rate:.4f}; rerun to retry uncached failures"
            )


if __name__ == "__main__":
    main()
