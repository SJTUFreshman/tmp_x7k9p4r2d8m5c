#!/usr/bin/env python3
"""Evaluate one frozen AFlow workflow on MATH with durable progress."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from jca.baseline.MATH.AFlow import aflow_math_core as core
from jca.baseline.MATH.MAD import math_mad_role_batched as shared
from jca.src.math_eval import load_math_problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--problem-ids-file", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--workflow-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--api-key", default="EMPTY")
    for agent in ("a1", "a2", "a3"):
        parser.add_argument(f"--api-base-{agent}", required=True)
        parser.add_argument(f"--api-model-{agent}", required=True)
        parser.add_argument(f"--model-path-{agent}", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_problems(args: argparse.Namespace) -> list[Any]:
    problems = load_math_problems(args.data_root, split=args.split)
    if args.problem_ids_file is not None:
        requested_ids: list[str] = []
        try:
            with args.problem_ids_file.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not str(value.get("problem_id", "")).strip():
                        raise ValueError(f"missing problem_id at line {line_number}")
                    requested_ids.append(str(value["problem_id"]).strip())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid problem IDs file {args.problem_ids_file}: {exc}") from exc
        if not requested_ids:
            raise SystemExit(f"no problem IDs found in {args.problem_ids_file}")
        if len(requested_ids) != len(set(requested_ids)):
            raise SystemExit(f"duplicate problem IDs in {args.problem_ids_file}")
        problem_by_id = {problem.problem_id: problem for problem in problems}
        missing = [problem_id for problem_id in requested_ids if problem_id not in problem_by_id]
        if missing:
            raise SystemExit(f"unknown problem IDs in {args.problem_ids_file}: {missing[:5]}")
        if args.start != 0 or args.limit != len(requested_ids):
            raise SystemExit(
                "--problem-ids-file requires --start=0 and --limit equal to the file length"
            )
        return [problem_by_id[problem_id] for problem_id in requested_ids]
    selected = problems[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise SystemExit(f"requested {args.limit}, selected {len(selected)}")
    return selected


def read_progress(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    valid_bytes = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break
            row = json.loads(raw)
            if row["problem_id"] in rows:
                raise ValueError(f"duplicate progress record: {row['problem_id']}")
            rows[row["problem_id"]] = row
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def summarize(rows: list[dict[str, Any]], expected: int, workflow_file: Path) -> dict[str, Any]:
    ids = [row["problem_id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError(f"coverage failure rows={len(rows)} unique={len(set(ids))} expected={expected}")
    subjects = {}
    for subject in sorted({row["subject"] for row in rows}):
        selected = [row for row in rows if row["subject"] == subject]
        correct = sum(row["em"] for row in selected)
        subjects[subject] = {"count": len(selected), "correct": int(correct), "em": correct / len(selected)}
    grouped: dict[str, dict[str, Any]] = {}
    retry_calls = Counter()
    op_failures = Counter()
    for row in rows:
        for op in row["op_records"]:
            key = f"{op['op']}|{op['agent']}|{op['model']}"
            bucket = grouped.setdefault(key, {
                "op": op["op"], "agent": op["agent"], "model": op["model"],
                "calls": 0, "successes": 0, "wall_time_s": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            })
            bucket["calls"] += 1
            bucket["successes"] += int(op["success"])
            bucket["wall_time_s"] = round(bucket["wall_time_s"] + op["wall_time_s"], 6)
            if not op["success"]:
                op_failures[op["op"]] += 1
            for attempt in op["attempts"]:
                retry_calls[int(attempt["retry_index"])] += 1
                usage = attempt.get("usage") or {}
                for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    bucket[token_key] += int(usage.get(token_key) or 0)
    correct = sum(row["em"] for row in rows)
    return {
        "workflow_file": str(workflow_file.resolve()), "expected": expected,
        "count": len(rows), "unique_problem_ids": len(set(ids)),
        "correct": int(correct), "em": correct / len(rows), "subjects": subjects,
        "errors": sum(bool(row["error"]) for row in rows),
        "average_ops": sum(row["n_ops"] for row in rows) / len(rows),
        "op_kinds": dict(Counter(kind for row in rows for kind in row["op_kinds"])),
        "op_failures": dict(op_failures), "calls_by_retry_index": dict(sorted(retry_calls.items())),
        "token_usage": core.token_usage_from_records(rows), "by_op_agent_model": grouped,
    }


def main() -> None:
    args = parse_args()
    if args.require_thinking and not args.enable_thinking:
        raise SystemExit("--require-thinking requires --enable-thinking")
    if args.split == "test" and args.limit != 5000:
        print(f"[warning] partial test evaluation limit={args.limit}")
    selected = select_problems(args)
    workflow = core.load_workflow(args.workflow_file, round_id=0)
    config = {
        "generation_seed": args.generation_seed, "max_new_tokens_executor": args.max_new_tokens,
        "temperature_executor": args.temperature, "top_p": args.top_p,
        "request_retries": args.request_retries,
        "enable_thinking": args.enable_thinking,
        "require_thinking": args.require_thinking,
        "max_model_len": 40960,
    }
    endpoints = {
        "A1": core.Endpoint("A1", args.api_base_a1, args.api_model_a1, args.model_path_a1, args.api_key, args.api_timeout),
        "A2": core.Endpoint("A2", args.api_base_a2, args.api_model_a2, args.model_path_a2, args.api_key, args.api_timeout),
        "A3": core.Endpoint("A3", args.api_base_a3, args.api_model_a3, args.model_path_a3, args.api_key, args.api_timeout),
    }
    print(f"AFlow MATH eval split={args.split} n={len(selected)} workflow={args.workflow_file}")
    if args.dry_run:
        print(f"first={selected[0].problem_id}; no server requests made")
        return
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output exists: {args.output}")
    if args.progress.exists() and not args.resume:
        raise SystemExit(f"progress exists; use --resume: {args.progress}")
    by_id = read_progress(args.progress) if args.resume else {}
    if by_id and not args.progress.exists():
        raise AssertionError("unreachable")
    missing = [problem for problem in selected if problem.problem_id not in by_id]
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    with args.progress.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as pool:
            futures = {
                pool.submit(core.execute_workflow, workflow, problem, endpoints, config, core.THIS_DIR / "prompts"): problem
                for problem in missing
            }
            for future in as_completed(futures):
                problem = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "problem_id": problem.problem_id, "subject": problem.subject, "level": problem.level,
                        "prediction": "", "gold_answer": problem.gold_answer, "em": 0.0,
                        "correct": False, "error": f"{type(exc).__name__}: {exc}",
                        "n_ops": 0, "op_kinds": [], "op_records": [], "wall_time_s": 0.0,
                    }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                by_id[row["problem_id"]] = row
                if len(by_id) == 1 or len(by_id) % 250 == 0 or len(by_id) == len(selected):
                    print(f"progress={len(by_id)}/{len(selected)}", flush=True)
    expected_ids = [problem.problem_id for problem in selected]
    if set(by_id) != set(expected_ids):
        raise ValueError(f"coverage mismatch got={len(by_id)} expected={len(expected_ids)}")
    rows = [by_id[problem_id] for problem_id in expected_ids]
    summary = summarize(rows, len(selected), args.workflow_file)
    shared.atomic_jsonl(args.output, rows)
    shared.atomic_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
