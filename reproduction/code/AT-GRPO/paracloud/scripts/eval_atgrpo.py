#!/usr/bin/env python3
"""Evaluate a trained AT-GRPO team on its dataset's official eval split.

Greedy, one rollout per problem. Problems are evaluated concurrently while each
problem keeps its turn-level agent calls synchronized. The untrained base team
is optional via ``--include-base`` for explicit comparison runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atgrpo.config import AGENTS, load_config  # noqa: E402
from atgrpo.logging_utils import write_json_atomic  # noqa: E402
from atgrpo.rollout import run_group  # noqa: E402
from atgrpo.serving import build_fleet  # noqa: E402
from atgrpo.tasks import EvalResult, build_task  # noqa: E402
from atgrpo.transport import (  # noqa: E402
    GenerationOptions,
    build_openai_caller,
    build_mock_callers,
)


def build_callers(config, base_urls, names, *, temperature):
    if config.mock:
        return build_mock_callers(
            accuracy=float(config.task_options.get("mock_accuracy", 0.5)),
            malformed_rate=0.0,
        )
    generation = GenerationOptions(
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=config.rollout.max_new_tokens,
        enable_thinking=config.train.enable_thinking,
    )
    return {
        agent: build_openai_caller(
            base_urls[agent], names[agent], generation=generation
        )
        for agent in AGENTS
    }


EVAL_GEN_FAILURES = []

def evaluate(
    config,
    task,
    fleet,
    names,
    *,
    limit,
    temperature,
    seed,
    max_concurrency=16,
):
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    problems = task.load("eval")
    if limit:
        problems = problems[:limit]
    callers = build_callers(config, fleet.base_urls(), names, temperature=temperature)

    def evaluate_one(index: int, problem):
        group = run_group(
            task, problem, callers,
            group_size=1,
            iteration=0,
            t_max=config.rollout.t_max,
            seed=seed,
            joint_mode=config.rollout.joint_mode,
            step_retries=config.rollout.step_retries,
            max_workers=1,
        )
        if len(group) != 1:
            raise RuntimeError(
                f"expected one trajectory for {problem.problem_id}, got {len(group)}"
            )
        trajectory = group[0]
        if getattr(trajectory, "terminated_by", None) == "exception":
            EVAL_GEN_FAILURES.append(
                (problem.problem_id,
                 str(getattr(trajectory, 'error', None) or 'unknown'))
            )
            print('[eval] generation failed (counted, not fatal) for '
                  + str(problem.problem_id), flush=True)
        return index, trajectory

    trajectories_by_index = [None] * len(problems)
    worker_count = min(max_concurrency, len(problems)) if problems else 1
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(evaluate_one, index, problem): index
            for index, problem in enumerate(problems)
        }
        completed = 0
        for future in as_completed(futures):
            index, trajectory = future.result()
            trajectories_by_index[index] = (problems[index], trajectory)
            completed += 1
            if completed == 1 or completed == len(problems) or completed % worker_count == 0:
                print(
                    f"[eval] completed {completed}/{len(problems)} problems",
                    flush=True,
                )

    trajectories = [item for item in trajectories_by_index if item is not None]

    scored = task.team_reward_batch([(p, t) for p, t in trajectories])
    results = [
        EvalResult(
            problem_id=problem.problem_id,
            final_answer=traj.final_answer,
            score=float(reward),
            detail=detail,
        )
        for (problem, traj), (reward, detail) in zip(trajectories, scored)
    ]
    return results, trajectories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--iteration", default="last", help="last|base|<N>")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="also evaluate the untrained base team (off by default)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="number of problems evaluated concurrently (default: 16)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1")

    config = load_config(args.config)
    if args.max_model_len:
        config.serving.max_model_len = args.max_model_len
    task = build_task(
        config.task,
        **({"reward_mode": config.reward_mode} if config.task == "multipl_e" else {}),
        **{
            key: value
            for key, value in config.task_options.items()
            if not key.startswith("mock_")
        },
    )
    run_dir = Path(args.run_dir or ROOT / "runs" / config.task / config.run_id)
    out_dir = Path(args.output or run_dir / "eval" / str(args.iteration))
    out_dir.mkdir(parents=True, exist_ok=True)

    fleet = build_fleet(config, log_dir=run_dir / "logs")
    try:
        fleet.launch()
        fleet.wait_healthy()
        summaries = {}
        arms = []
        if args.iteration != "base":
            adapters = {}
            for agent in AGENTS:
                directory = run_dir / "adapters" / agent
                if args.iteration == "last":
                    candidates = sorted(directory.glob("iter_*"))
                    if not candidates:
                        raise SystemExit(f"no adapters found in {directory}")
                    adapters[agent] = candidates[-1]
                else:
                    adapters[agent] = directory / f"iter_{int(args.iteration):04d}"
            names = fleet.publish_adapters(0, adapters)
            arms.append(("trained", names))
        if args.iteration == "base" or args.include_base:
            arms.append(("base", {a: f"{a}_base" for a in AGENTS}))

        for arm, names in arms:
            started = time.time()
            results, _ = evaluate(
                config, task, fleet, names,
                limit=args.limit,
                temperature=args.temperature,
                seed=args.seed,
                max_concurrency=args.max_concurrency,
            )
            metric = task.eval_metric(results)
            metric["arm"] = arm
            metric["seconds"] = time.time() - started
            summaries[arm] = metric
            with (out_dir / f"results_{arm}.jsonl").open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(
                        json.dumps(
                            {
                                "problem_id": result.problem_id,
                                "final_answer": result.final_answer,
                                "score": result.score,
                                "detail": result.detail,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"[{arm}] {task.headline_metric} = {metric.get(task.headline_metric)}")

        write_json_atomic(
            out_dir / "summary.json",
            {
                "task": config.task,
                "headline_metric": task.headline_metric,
                "iteration": args.iteration,
                "generation_failures": len(EVAL_GEN_FAILURES),
                "generation_failure_ids": [x[0] for x in EVAL_GEN_FAILURES[:50]],
                "arms": summaries,
            },
        )
        print(f"wrote {out_dir / 'summary.json'}")
    finally:
        fleet.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
