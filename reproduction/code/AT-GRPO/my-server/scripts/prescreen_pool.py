#!/usr/bin/env python3
"""Filter the train pool down to prompts that can produce a gradient.

With a binary reward and group size G, a prompt the base team always solves (or
never solves) has zero reward variance, hence zero advantage, hence contributes
nothing -- while still consuming G rollouts. On MATH especially, most prompts
are in one of those two buckets for a 1.7B+4B+8B team.

This samples each candidate G times with the UNTRAINED team and keeps the ones
whose pass count lands inside ``--keep-band``. The result is reusable across
runs, so the cost is paid once per dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atgrpo.config import AGENTS, load_config  # noqa: E402
from atgrpo.rollout import run_group, score_groups  # noqa: E402
from atgrpo.serving import build_fleet  # noqa: E402
from atgrpo.tasks import build_task  # noqa: E402
from atgrpo.transport import (  # noqa: E402
    GenerationOptions,
    build_openai_caller,
    build_mock_callers,
)


def resolve_group_size(config, requested: int) -> int:
    return requested or config.atgrpo.group_size_K


def validate_prescreen_args(
    *, group_size: int, keep_low: int, keep_high: int, sample: int
) -> None:
    if group_size < 2:
        raise ValueError("group size must be >= 2")
    if keep_low < 0 or keep_high < keep_low or keep_high > group_size:
        raise ValueError(
            f"keep band must satisfy 0 <= low <= high <= group size ({group_size})"
        )
    if sample < 0:
        raise ValueError("sample must be >= 0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=0, help="0 = use config value")
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--keep-low", type=int, default=-1)
    parser.add_argument("--keep-high", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--problem-concurrency", type=int, default=0,
        help="number of prompts screened concurrently; 0 = choose from max_concurrency",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    group_size = resolve_group_size(config, args.group_size)
    low = args.keep_low if args.keep_low >= 0 else config.pool.keep_band[0]
    high = args.keep_high if args.keep_high >= 0 else config.pool.keep_band[1]
    sample = args.sample or config.pool.prescreen_sample
    validate_prescreen_args(
        group_size=group_size, keep_low=low, keep_high=high, sample=sample
    )
    if args.problem_concurrency < 0:
        raise ValueError("problem concurrency must be >= 0")

    task = build_task(
        config.task,
        **({"reward_mode": config.reward_mode} if config.task == "multipl_e" else {}),
    )
    problems = task.load("train")
    rng = random.Random(args.seed)
    if sample and sample < len(problems):
        problems = rng.sample(problems, sample)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary_output.unlink(missing_ok=True)
    fleet = build_fleet(config, log_dir=ROOT / "runs" / "_shared" / "prescreen_logs")
    try:
        fleet.launch()
        fleet.wait_healthy()
        if config.mock:
            callers = build_mock_callers(accuracy=0.5, malformed_rate=0.0)
        else:
            generation = GenerationOptions(
                temperature=config.rollout.temperature,
                top_p=config.rollout.top_p,
                max_new_tokens=config.rollout.max_new_tokens,
                enable_thinking=config.train.enable_thinking,
            )
            base_urls = fleet.base_urls()
            # Deliberately the BASE models: prescreening must describe the
            # starting policy, not a partially trained one.
            callers = {
                a: build_openai_caller(
                    base_urls[a], f"{a}_base", generation=generation
                )
                for a in AGENTS
            }

        kept = 0
        started = time.time()
        problem_workers = args.problem_concurrency or max(
            1, min(4, config.rollout.max_concurrency // max(1, group_size))
        )
        group_workers = max(
            1, config.rollout.max_concurrency // problem_workers
        )

        def screen_one(item: tuple[int, Any]) -> tuple[str, int, bool]:
            index, problem = item
            group = run_group(
                task, problem, callers,
                group_size=group_size, iteration=0,
                t_max=config.rollout.t_max, seed=args.seed + index,
                joint_mode=config.rollout.joint_mode,
                step_retries=config.rollout.step_retries,
                max_workers=group_workers,
            )
            score_groups(task, [(problem, group)])
            passes = sum(1 for trajectory in group if (trajectory.team_reward or 0.0) >= 0.5)
            return problem.problem_id, passes, low <= passes <= high

        print(
            f"screening {len(problems)} prompts with {problem_workers} prompt workers "
            f"and {group_workers} rollout workers each",
            flush=True,
        )
        with temporary_output.open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=problem_workers) as pool:
                screened = pool.map(screen_one, enumerate(problems))
                for index, (problem_id, passes, keep) in enumerate(screened):
                    kept += int(keep)
                    handle.write(
                        json.dumps(
                            {
                                "problem_id": problem_id,
                                "passes": passes,
                                "group_size": group_size,
                                "keep": keep,
                            }
                        )
                        + "\n"
                    )
                    handle.flush()
                    if (index + 1) % 50 == 0:
                        rate = (index + 1) / max(1e-9, time.time() - started)
                        print(
                            f"  {index + 1}/{len(problems)} kept={kept} "
                            f"({rate:.2f} prompts/s)",
                            flush=True,
                        )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_output.replace(output)
        print(f"kept {kept}/{len(problems)} prompts -> {output}")
    finally:
        try:
            fleet.shutdown()
        finally:
            temporary_output.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
