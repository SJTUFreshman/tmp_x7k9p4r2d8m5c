#!/usr/bin/env python3
"""Search AFlow workflows on 20 fixed MultiPL-E train70 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from aflow_multipl_e_adapter import (
    PROJECT_ROOT,
    PublicTestWorkflowEvaluator,
    configure_core,
    load_manifest_tasks,
    select_search_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFlow search on MultiPL-E train70.")
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-workflow", type=Path, required=True)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--search-size", type=int, default=20)
    parser.add_argument("--search-seed", type=int, default=20260810)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=20260810)
    parser.add_argument("--max-new-tokens-executor", type=int, default=4096)
    parser.add_argument("--max-new-tokens-optimizer", type=int, default=2048)
    parser.add_argument("--temperature-executor", type=float, default=0.7)
    parser.add_argument("--temperature-optimizer", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-base-optimizer", default="http://127.0.0.1:8204/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-model-optimizer", default="Optimizer_14B")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--eval-image", required=True)
    parser.add_argument("--docker-exec", default="docker")
    parser.add_argument("--eval-shards", type=int, default=20)
    parser.add_argument("--eval-inner-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke_mode and (args.max_iterations != 20 or args.search_size != 20):
        raise SystemExit("Canonical MultiPL-E AFlow requires 20 workflows x 20 tasks")
    if args.max_concurrency < 1 or args.eval_shards < 1 or args.eval_inner_workers < 1:
        raise SystemExit("concurrency and evaluator worker counts must be positive")
    modules = configure_core()
    optimizer = modules["optimizer"]
    workflow_module = modules["workflow"]
    from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller

    all_train = load_manifest_tasks(
        args.split_manifest, partition="train", languages=args.languages
    )
    search_tasks = select_search_tasks(all_train, args.search_size, args.search_seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "search_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": problem.id,
                    "root_dataset": problem.root_dataset,
                    "language": problem.language,
                    "problem_id": problem.name,
                }
                for problem in search_tasks
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"AFlow search train_tasks={len(all_train)} selected={len(search_tasks)} "
        f"iterations={args.max_iterations}"
    )
    for problem in search_tasks:
        print(f"  search_task={problem.id}")
    if args.dry_run:
        return 0

    executor_generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens_executor,
        temperature=args.temperature_executor,
        top_p=args.top_p,
        enable_thinking=False,
    )
    callers = []
    for agent in ("a1", "a2", "a3"):
        callers.append(
            OpenAIChatLLMCaller(
                getattr(args, f"api_base_{agent}"),
                getattr(args, f"api_model_{agent}"),
                generation=executor_generation,
                timeout=args.api_timeout,
                response_format={"type": "json_object"},
            )
        )
    executor_callers = workflow_module.ExecutorCallers(
        callers, callers[2], ["A1_1.7B", "A2_4B", "A3_8B"], "A3_8B"
    )
    optimizer_caller = OpenAIChatLLMCaller(
        args.api_base_optimizer,
        args.api_model_optimizer,
        generation=GenerationOptions(
            max_new_tokens=args.max_new_tokens_optimizer,
            temperature=args.temperature_optimizer,
            top_p=args.top_p,
            enable_thinking=False,
        ),
        timeout=args.api_timeout,
    )
    optimizer.evaluate_workflow = PublicTestWorkflowEvaluator(
        root=args.run_dir / "public_test_evaluations",
        evaluator_script=args.evaluator_script,
        eval_image=args.eval_image,
        docker_exec=args.docker_exec,
        shards=args.eval_shards,
        inner_workers=args.eval_inner_workers,
    )
    started = time.monotonic()
    nodes = optimizer.run_mcts(
        root=args.run_dir,
        initial_source_path=args.initial_workflow,
        dev_problems=search_tasks,
        executor_callers=executor_callers,
        optimizer_caller=optimizer_caller,
        max_iterations=args.max_iterations,
        max_concurrency=args.max_concurrency,
        rng_seed=args.rng_seed,
        optimizer_caller_id=args.api_model_optimizer,
    )
    best = optimizer.best_node(nodes)
    if best is None:
        raise SystemExit("Search produced no valid workflow")
    summary = (
        f"SEARCH_PARTITION=train70\nSEARCH_TASKS={len(search_tasks)}\n"
        f"MAX_ITERATIONS={args.max_iterations}\nBEST_ROUND={best.round_id}\n"
        f"BEST_PUBLIC_TEST_PASS_RATE={best.dev_em:.6f}\n"
        f"BEST_SOURCE={best.source_file}\nWALL_TIME_S={time.monotonic() - started:.1f}\n"
    )
    (args.run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
