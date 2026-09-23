"""Evaluate a committed MAPORL checkpoint with optional paired base debate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maporl.config import AGENTS, load_config
from maporl.evaluation import build_eval_callers, evaluate, resolve_checkpoint
from maporl.logging_utils import write_json_atomic
from maporl.serving import build_fleet
from maporl.tasks.base import build_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", default="last")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-concurrency", type=int)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative; zero evaluates the complete split")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    config = load_config(args.config)
    if config.mock:
        parser.error("this CLI requires actual model inference; mock evaluation is test-only")
    checkpoint, iteration, identity = resolve_checkpoint(args.run_dir, args.iteration)
    if identity["config_sha256"] and identity["config_sha256"] != config.sha256():
        parser.error("evaluation config differs from the committed checkpoint config")
    if args.temperature is not None:
        if args.temperature <= 0:
            parser.error('--temperature must be > 0')
        config.rollout.temperature = args.temperature
    options = dict(config.task_options)
    if config.task == "multipl_e":
        options["reward_mode"] = config.reward_mode
    task = build_task(config.task, **options)
    problems = task.load("eval")
    if args.limit:
        problems = problems[:args.limit]
    output_dir = args.output_dir or args.run_dir / "eval" / checkpoint.name
    fleet = build_fleet(config, log_dir=output_dir / "logs")
    try:
        fleet.launch()
        fleet.wait_healthy()
        names = fleet.publish_adapters(iteration, {
            agent: checkpoint / "policies" / agent
            for agent in AGENTS[:config.maporl.agent_num]
        })
        summary = {}
        for mode in (["maporl", "base"] if args.include_base else ["maporl"]):
            summary[mode] = evaluate(
                config, task, problems,
                build_eval_callers(
                    config, fleet, policy_names=names, mode=mode,
                    adapter_version=identity["manifest_sha256"],
                ),
                output_dir=output_dir / mode,
                checkpoint_identity=identity,
                mode=mode, seed=args.seed, batch_size=args.batch_size,
                max_concurrency=args.max_concurrency,
            )
        for mode_name in summary:
            summary[mode_name]['temperature'] = config.rollout.temperature
        write_json_atomic(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
