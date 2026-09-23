#!/usr/bin/env python3
"""Run the MAGRPO online loop for one dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magrpo.config import AGENTS, load_config  # noqa: E402
from magrpo.logging_utils import build_manifest, sha256_file  # noqa: E402
from magrpo.loop import MagrpoLoop  # noqa: E402
from magrpo.serving import build_fleet  # noqa: E402
from magrpo.tasks import build_task  # noqa: E402
from magrpo.trainer import AgentTrainer  # noqa: E402
from magrpo.transport import (  # noqa: E402
    GenerationOptions,
    OpenAIChatCaller,
    build_mock_callers,
)


def make_callers_factory(config, fleet):
    """Callers are rebuilt each iteration so the served name tracks the swap."""
    if config.mock:
        gold = config.task_options.get("mock_gold") or {}
        accuracy = float(config.task_options.get("mock_accuracy", 0.5))
        malformed = float(config.task_options.get("mock_malformed_rate", 0.05))
        field = config.task_options.get("mock_answer_field", "tentative_answer")

        def mock_factory(_base_urls):
            return build_mock_callers(
                accuracy=accuracy,
                malformed_rate=malformed,
                gold_by_problem=gold,
                answer_field=field,
            )

        return mock_factory

    def factory(base_urls):
        generation = GenerationOptions(
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            max_new_tokens=config.rollout.max_new_tokens,
            enable_thinking=config.train.enable_thinking,
        )
        return {
            agent: OpenAIChatCaller(
                base_urls[agent],
                fleet.current_names.get(agent, f"{agent}_base"),
                generation=generation,
            )
            for agent in AGENTS
        }

    return factory


def resolve_warm_start(source: Path):
    source = source.resolve()
    state_path = source / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = int(state["iteration"])
    if completed < 1:
        raise ValueError("warm start requires a committed training iteration")
    adapters = {}
    hashes = {}
    for agent in AGENTS:
        adapter = source / "adapters" / agent / f"iter_{completed - 1:04d}"
        weights = adapter / "adapter_model.safetensors"
        if not (adapter / "adapter_config.json").is_file() or not weights.is_file():
            raise FileNotFoundError(f"missing committed warm-start adapter: {adapter}")
        adapters[agent] = adapter
        hashes[agent] = sha256_file(weights)
    metadata = {
        "warm_start_from": str(source),
        "warm_start_mode": "weights_only_fresh_optimizer",
        "warm_start_source_completed_iterations": completed,
        "warm_start_source_adapter_iteration": completed - 1,
        "warm_start_source_elapsed_seconds": state.get("elapsed_seconds"),
        "warm_start_source_state_sha256": sha256_file(state_path),
        "warm_start_adapter_sha256": hashes,
        "warm_start_warning": "Optimizer state is unavailable. This is a new phase with fresh optimizer, iteration counter and prompt pool; not an exact resume.",
    }
    return adapters, metadata


def build_trainers(config, *, tiny_model: str | None = None, init_adapters=None):
    trainers = {}
    for agent in AGENTS:
        device = "cpu"
        if not config.mock:
            assigned = str(config.serving.train_gpus[agent]).strip()
            if not assigned.isdigit():
                raise ValueError(
                    f"{agent}: train_gpus must name one CUDA device; "
                    f"multi-device training is not implemented: {assigned!r}"
                )
            device = f"cuda:{int(assigned)}"
        init_adapter = str(init_adapters[agent]) if init_adapters else None
        trainers[agent] = AgentTrainer(
            agent,
            base_model=tiny_model or config.serving.models[agent],
            lora=config.train.lora,
            learning_rate=config.train.learning_rate[agent],
            max_seq_length=config.train.max_seq_length,
            per_device_batch_size=config.train.per_device_batch_size[agent],
            weight_decay=config.train.weight_decay,
            max_grad_norm=config.train.max_grad_norm,
            warmup_iters=config.train.warmup_iters,
            gradient_checkpointing=config.train.gradient_checkpointing,
            enable_thinking=config.train.enable_thinking,
            device=device,
            init_adapter=init_adapter,
        )
    return trainers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument(
        "--ignore-wall-clock-limit",
        action="store_true",
        help="ignore the config wall-clock cap; iteration limit still applies",
    )
    parser.add_argument("--prescreen", default=None, help="allowlist jsonl")
    parser.add_argument("--tiny-model", default=None, help="smoke-test model dir")
    parser.add_argument("--warm-start-from", default=None, help="prior run directory; loads committed LoRA weights with a fresh optimizer")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir or ROOT / "runs" / config.task / config.run_id)
    source_adapters = None
    manifest_extra = None
    if args.warm_start_from:
        if args.resume:
            parser.error("--warm-start-from and --resume are mutually exclusive")
        if run_dir.resolve() == Path(args.warm_start_from).resolve():
            parser.error("warm start requires a new run directory")
        if (run_dir / "state.json").exists() or (run_dir / "manifest.json").exists():
            parser.error("warm start destination already contains a run")
        source_adapters, manifest_extra = resolve_warm_start(Path(args.warm_start_from))
        print(f"WARNING: {manifest_extra['warm_start_warning']}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_options = {
        key: value for key, value in config.task_options.items()
        if not key.startswith("mock_")
    }
    if config.task == "multipl_e":
        task_options["reward_mode"] = config.reward_mode
    task = build_task(config.task, **task_options)

    fleet = build_fleet(config, log_dir=run_dir / "logs")
    try:
        fleet.launch()
        fleet.wait_healthy()
        trainers = build_trainers(config, tiny_model=args.tiny_model, init_adapters=source_adapters)
        loop = MagrpoLoop(
            config,
            task,
            fleet,
            trainers,
            run_dir=run_dir,
            callers_factory=make_callers_factory(config, fleet),
            prescreen_path=Path(args.prescreen) if args.prescreen else None,
        )
        provenance = dict(manifest_extra or {})
        if args.ignore_wall_clock_limit:
            provenance["runtime_override"] = {
                "ignore_wall_clock_limit": True,
                "config_wall_clock_hours": config.budget.max_wall_clock_hours,
            }
        loop.logger.write_manifest(build_manifest(config, root=ROOT, extra=provenance))
        if source_adapters is not None:
            fleet.publish_adapters(0, source_adapters)
            loop.logger.print("published warm-start adapters before first rollout")
        if args.resume:
            loop.load_state()
        loop.run(
            max_iterations=args.max_iterations or None,
            ignore_wall_clock_limit=args.ignore_wall_clock_limit,
        )
    finally:
        fleet.shutdown()
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
