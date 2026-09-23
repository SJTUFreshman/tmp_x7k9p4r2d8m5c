#!/usr/bin/env python3
"""Search MATH AFlow workflows on a fixed shuffled train-20 subset."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from jca.baseline.MATH.AFlow import aflow_math_core as core
from jca.baseline.MATH.MAD import math_mad_role_batched as shared
from jca.src.math_eval import load_math_problems


STATE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-workflow", type=Path, required=True)
    parser.add_argument("--search-size", type=int, default=20)
    parser.add_argument("--search-seed", type=int, default=20260810)
    parser.add_argument("--rng-seed", type=int, default=20260810)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens-executor", type=int, default=8192)
    parser.add_argument("--max-new-tokens-optimizer", type=int, default=2048)
    parser.add_argument("--temperature-executor", type=float, default=0.7)
    parser.add_argument("--temperature-optimizer", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--api-key", default="EMPTY")
    for agent in ("a1", "a2", "a3", "optimizer"):
        parser.add_argument(f"--api-base-{agent}", required=True)
        parser.add_argument(f"--api-model-{agent}", required=True)
        parser.add_argument(f"--model-path-{agent}", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    valid_bytes = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break
            rows.append(json.loads(raw))
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def evaluate_resumable(
    workflow: Any, problems: list[Any], endpoints: dict[str, core.Endpoint],
    config: dict[str, Any], progress_path: Path, max_concurrency: int,
) -> list[dict[str, Any]]:
    existing = read_complete_jsonl(progress_path)
    by_id = {row["problem_id"]: row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError(f"duplicate problem IDs in {progress_path}")
    missing = [problem for problem in problems if problem.problem_id not in by_id]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
            futures = {
                pool.submit(core.execute_workflow, workflow, problem, endpoints, config, core.THIS_DIR / "prompts"): problem
                for problem in missing
            }
            for completed, future in enumerate(as_completed(futures), 1):
                problem = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "problem_id": problem.problem_id, "subject": problem.subject,
                        "level": problem.level, "prediction": "", "gold_answer": problem.gold_answer,
                        "em": 0.0, "correct": False, "error": f"{type(exc).__name__}: {exc}",
                        "n_ops": 0, "op_kinds": [], "op_records": [], "wall_time_s": 0.0,
                    }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                by_id[row["problem_id"]] = row
                print(f"  workflow progress={len(by_id)}/{len(problems)}", flush=True)
    expected = {problem.problem_id for problem in problems}
    if set(by_id) != expected:
        raise ValueError(f"workflow coverage mismatch: got={len(by_id)} expected={len(expected)}")
    return [by_id[problem.problem_id] for problem in problems]


def task_manifest(problems: list[Any]) -> list[dict[str, str]]:
    return [
        {"problem_id": problem.problem_id, "subject": problem.subject, "level": problem.level}
        for problem in problems
    ]


def main() -> None:
    args = parse_args()
    if args.require_thinking and not args.enable_thinking:
        raise SystemExit("--require-thinking requires --enable-thinking")
    if args.search_size != 20 or args.max_iterations != 20:
        raise SystemExit("canonical MATH AFlow requires search-size=20 and max-iterations=20")
    if not args.initial_workflow.exists():
        raise SystemExit(f"initial workflow missing: {args.initial_workflow}")
    all_problems = load_math_problems(args.data_root, split="train")
    shuffled = list(all_problems)
    random.Random(args.search_seed).shuffle(shuffled)
    problems = shuffled[:args.search_size]
    manifest = task_manifest(problems)
    if args.dry_run:
        core.validate_workflow_source(args.initial_workflow.read_text(encoding="utf-8"))
        print(f"AFlow MATH search tasks={len(problems)} proposals={args.max_iterations} run_dir={args.run_dir}")
        print(f"search_task_ids={[problem.problem_id for problem in problems]}")
        print("dry-run passed; no files or server requests created")
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "search_tasks.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("search task manifest differs from fixed seed selection")
    else:
        shared.atomic_json(manifest_path, manifest)

    config = {
        "search_size": args.search_size, "search_seed": args.search_seed,
        "rng_seed": args.rng_seed, "max_iterations": args.max_iterations,
        "generation_seed": args.generation_seed,
        "max_new_tokens_executor": args.max_new_tokens_executor,
        "max_new_tokens_optimizer": args.max_new_tokens_optimizer,
        "temperature_executor": args.temperature_executor,
        "temperature_optimizer": args.temperature_optimizer, "top_p": args.top_p,
        "request_retries": args.request_retries,
        "enable_thinking": args.enable_thinking,
        "require_thinking": args.require_thinking,
        "max_model_len": 40960,
        "model_paths": {
            "A1": args.model_path_a1, "A2": args.model_path_a2,
            "A3": args.model_path_a3, "OPT": args.model_path_optimizer,
        },
    }
    endpoints = {
        "A1": core.Endpoint("A1", args.api_base_a1, args.api_model_a1, args.model_path_a1, args.api_key, args.api_timeout),
        "A2": core.Endpoint("A2", args.api_base_a2, args.api_model_a2, args.model_path_a2, args.api_key, args.api_timeout),
        "A3": core.Endpoint("A3", args.api_base_a3, args.api_model_a3, args.model_path_a3, args.api_key, args.api_timeout),
    }
    optimizer_endpoint = core.Endpoint(
        "OPT", args.api_base_optimizer, args.api_model_optimizer,
        args.model_path_optimizer, args.api_key, args.api_timeout,
    )
    print(f"AFlow MATH search tasks={len(problems)} proposals={args.max_iterations} run_dir={args.run_dir}")
    print(f"search_task_ids={[problem.problem_id for problem in problems]}")

    state_path = args.run_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION or state.get("config") != config:
            raise SystemExit("existing search state configuration mismatch")
    else:
        state = {"version": STATE_VERSION, "config": config, "nodes": []}
        shared.atomic_json(state_path, state)

    nodes = state["nodes"]
    started = time.monotonic()
    if not nodes:
        source = args.initial_workflow.read_text(encoding="utf-8")
        source_path = core.write_workflow(args.run_dir, 0, "initial", source)
        workflow = core.load_workflow(source_path, round_id=0)
        records = evaluate_resumable(
            workflow, problems, endpoints, config, args.run_dir / "progress" / "round_00.jsonl",
            args.max_concurrency,
        )
        score = sum(record["em"] for record in records) / len(problems)
        nodes.append({
            "round_id": 0, "name": "initial", "parent_round": None,
            "source_file": str(source_path.resolve()), "dev_em": score,
            "dev_records": records, "visits": 1, "proposed_by": "manual",
            "parse_ok": True, "reject_reason": None, "optimizer_attempts": [],
        })
        shared.atomic_json(state_path, state)
        print(f"[round 0] initial EM={score:.3f}")

    for round_id in range(1, args.max_iterations + 1):
        if any(int(node["round_id"]) == round_id for node in nodes):
            pending_path = args.run_dir / "pending_node.json"
            if pending_path.exists():
                stale = json.loads(pending_path.read_text(encoding="utf-8"))
                if int(stale["round_id"]) == round_id:
                    pending_path.unlink()
            continue
        pending_path = args.run_dir / "pending_node.json"
        if pending_path.exists():
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if int(pending["round_id"]) != round_id:
                raise SystemExit(f"pending round mismatch: {pending['round_id']} vs {round_id}")
        else:
            parent = core.ucb_parent(nodes, round_id, args.rng_seed)
            parent["visits"] = int(parent["visits"]) + 1
            source, attempts, reject_reason = core.propose_workflow(
                parent, optimizer_endpoint, config, core.THIS_DIR / "prompts" / "optimizer_propose.md", round_id,
            )
            source_path = core.write_workflow(
                args.run_dir, round_id, "proposed" if reject_reason is None else "rejected", source,
            )
            pending = {
                "round_id": round_id, "parent_round": parent["round_id"],
                "source_file": str(source_path.resolve()), "optimizer_attempts": attempts,
                "reject_reason": reject_reason,
            }
            shared.atomic_json(pending_path, pending)
            shared.atomic_json(state_path, state)
        if pending["reject_reason"] is not None:
            nodes.append({
                "round_id": round_id, "name": "rejected",
                "parent_round": pending["parent_round"], "source_file": pending["source_file"],
                "dev_em": 0.0, "dev_records": [], "visits": 0,
                "proposed_by": "optimizer", "parse_ok": False,
                "reject_reason": pending["reject_reason"],
                "optimizer_attempts": pending["optimizer_attempts"],
            })
        else:
            workflow = core.load_workflow(Path(pending["source_file"]), round_id=round_id)
            records = evaluate_resumable(
                workflow, problems, endpoints, config,
                args.run_dir / "progress" / f"round_{round_id:02d}.jsonl", args.max_concurrency,
            )
            score = sum(record["em"] for record in records) / len(problems)
            nodes.append({
                "round_id": round_id, "name": "proposed",
                "parent_round": pending["parent_round"], "source_file": pending["source_file"],
                "dev_em": score, "dev_records": records, "visits": 1,
                "proposed_by": "optimizer", "parse_ok": True, "reject_reason": None,
                "optimizer_attempts": pending["optimizer_attempts"],
            })
            print(f"[round {round_id}] EM={score:.3f} parent={pending['parent_round']}")
        shared.atomic_json(state_path, state)
        pending_path.unlink(missing_ok=True)

    best = core.best_node(nodes)
    summary = {
        "run_dir": str(args.run_dir.resolve()), "total_nodes": len(nodes),
        "parse_ok_nodes": sum(node["parse_ok"] for node in nodes),
        "best_round": best["round_id"], "best_em": best["dev_em"],
        "best_source": best["source_file"], "wall_time_s": round(time.monotonic() - started, 3),
    }
    shared.atomic_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "summary.txt").write_text(
        "\n".join(f"{key.upper()}={value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
