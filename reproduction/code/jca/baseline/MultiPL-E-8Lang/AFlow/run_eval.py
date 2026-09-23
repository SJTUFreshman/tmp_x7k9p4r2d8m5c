#!/usr/bin/env python3
"""Generate test30 completions with one frozen AFlow workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from aflow_multipl_e_adapter import (
    completion_payload,
    configure_core,
    load_manifest_tasks,
    normalize_candidate,
    write_gzip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen AFlow on test30.")
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--workflow-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--timings-file", type=Path)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--max-problems-per-language", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def limit_per_language(tasks: list[Any], maximum: int | None) -> list[Any]:
    if maximum is None:
        return tasks
    counts: dict[tuple[str, str], int] = {}
    selected = []
    for task in tasks:
        key = (task.root_dataset, task.language)
        if counts.get(key, 0) >= maximum:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(task)
    return selected


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def run_one(problem: Any, workflow: Any, callers: Any, workflow_file: Path):
    from workflow import run_workflow_on_problem

    started = time.monotonic()
    record = run_workflow_on_problem(workflow, problem, callers)
    raw_completion = record.final_answer or ""
    completion = ""
    error = record.error
    if not error and raw_completion:
        try:
            completion = normalize_candidate(problem, raw_completion)
        except Exception as exc:
            error = f"Normalization failed: {type(exc).__name__}: {exc}"
    elif not error:
        error = "Workflow returned an empty continuation"
    status = "exception" if error else "stop"
    trace = {
        "schema_version": 1,
        "root_dataset": problem.root_dataset,
        "language": problem.language,
        "problem_id": problem.name,
        "workflow_file": str(workflow_file),
        "workflow_round": workflow.round_id,
        "raw_final_completion": raw_completion,
        "final_completion": completion,
        "op_calls": [asdict(call) for call in record.op_calls],
        "terminated_by": status,
        "error": error,
        "wall_time_s": round(time.monotonic() - started, 3),
        "thinking_mode": "disabled",
    }
    return completion_payload(problem, completion, status, workflow_file), trace


def main() -> int:
    args = parse_args()
    if args.max_concurrency < 1 or args.max_new_tokens < 1:
        raise SystemExit("token and concurrency limits must be positive")
    if args.max_problems_per_language is not None and args.max_problems_per_language < 1:
        raise SystemExit("--max-problems-per-language must be positive")
    modules = configure_core()
    workflow_module = modules["workflow"]
    tasks = limit_per_language(
        load_manifest_tasks(
            args.split_manifest, partition="test", languages=args.languages
        ),
        args.max_problems_per_language,
    )
    if not tasks:
        raise SystemExit("No test30 tasks selected")
    workflow = workflow_module.load_workflow_from_file(args.workflow_file, round_id=0)
    print(f"AFlow frozen workflow={args.workflow_file} test_tasks={len(tasks)}")
    print("thinking_mode=disabled hidden_test_access=generation:none")
    if args.dry_run:
        print(f"dry-run first={tasks[0].id}")
        return 0

    from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    solve_callers = [
        OpenAIChatLLMCaller(
            getattr(args, f"api_base_{agent}"),
            getattr(args, f"api_model_{agent}"),
            generation=generation,
            timeout=args.api_timeout,
            response_format={"type": "json_object"},
        )
        for agent in ("a1", "a2", "a3")
    ]
    callers = workflow_module.ExecutorCallers(
        solve_callers,
        solve_callers[2],
        ["A1_1.7B", "A2_4B", "A3_8B"],
        "A3_8B",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.trajectory_output.open("w", encoding="utf-8") as trace_handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(tasks))) as pool:
            futures = {
                pool.submit(run_one, task, workflow, callers, args.workflow_file): task
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                problem = futures[future]
                try:
                    payload, trace = future.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    trace = {
                        "schema_version": 1,
                        "root_dataset": problem.root_dataset,
                        "language": problem.language,
                        "problem_id": problem.name,
                        "workflow_file": str(args.workflow_file),
                        "workflow_round": 0,
                        "raw_final_completion": "",
                        "final_completion": "",
                        "op_calls": [],
                        "terminated_by": "exception",
                        "error": error,
                        "wall_time_s": 0.0,
                    }
                    payload = completion_payload(
                        problem, "", "exception", args.workflow_file
                    )
                output_path = (
                    args.output_dir
                    / problem.root_dataset
                    / problem.language
                    / f"{problem.name}.json.gz"
                )
                write_gzip(output_path, payload)
                trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_handle.flush()
                append_jsonl(
                    args.timings_file,
                    {
                        "event": "generation_completed",
                        "root_dataset": problem.root_dataset,
                        "language": problem.language,
                        "problem_id": problem.name,
                        "completion_file": str(output_path.resolve()),
                        "total_elapsed_seconds": round(time.monotonic() - started, 3),
                        "status": trace["terminated_by"],
                    },
                )
                print(
                    f"[{index}/{len(tasks)}] {problem.id} "
                    f"status={trace['terminated_by']} ops={len(trace['op_calls'])}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
