#!/usr/bin/env python3
"""Run the AT-GRPO online loop for one dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atgrpo.config import AGENTS, load_config  # noqa: E402
from atgrpo.logging_utils import (  # noqa: E402
    adapter_checkpoint_complete,
    build_manifest,
    project_code_sha256,
)
from atgrpo.loop import ATGRPOLoop  # noqa: E402
from atgrpo.serving import build_fleet  # noqa: E402
from atgrpo.tasks import build_task  # noqa: E402
from atgrpo.trainer import AgentTrainer  # noqa: E402
from atgrpo.transport import (  # noqa: E402
    GenerationOptions,
    build_openai_caller,
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
            agent: build_openai_caller(
                base_urls[agent],
                fleet.current_names.get(agent, f"{agent}_base"),
                generation=generation,
                # A failed vLLM request invalidates the whole online iteration.
                # Retrying it in-place can occupy a global rollout slot for up
                # to another 600 seconds and only delays the fail-closed path.
                retries=0,
            )
            for agent in AGENTS
        }

    return factory


def trainer_device(config, agent: str) -> str:
    """Resolve this agent's training device from ``serving.train_gpus``.

    Must be explicit. All three trainers live in one process, so
    CUDA_VISIBLE_DEVICES cannot separate them, and leaving ``device=None``
    resolves to ``cuda:0`` -- which under the split44 plan is a *serving* GPU.
    The result is all three trainers piling onto the A1 vLLM server's card while
    the four training GPUs sit idle.
    """
    if config.mock:
        return "cpu"
    spec = config.serving.train_gpus.get(agent, "")
    first = next((g.strip() for g in spec.split(",") if g.strip()), None)
    if first is None:
        raise ValueError(
            f"serving.train_gpus has no entry for {agent}; refusing to fall back "
            "to cuda:0, which is a serving GPU under the split44 plan"
        )
    return f"cuda:{first}"


def build_trainers(
    config,
    *,
    tiny_model: str | None = None,
    init_adapters: dict[str, str] | None = None,
):
    trainers = {}
    init_adapters = init_adapters or {}
    for agent in AGENTS[: config.atgrpo.agent_num]:
        device = trainer_device(config, agent)
        source = init_adapters.get(agent)
        suffix = f" (resume {source})" if source else ""
        print(f"[train] {agent} -> {device}{suffix}", flush=True)
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
            init_adapter=source,
        )
    return trainers


def load_resume_state(
    config, run_dir: Path, *, expected_code_sha256: str | None = None
) -> dict:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"resume requested but state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("config_sha256") != config.sha256():
        raise RuntimeError(
            "config changed since this run was created; resume would mix two "
            "different setups. Use a new run_id."
        )
    expected_code_sha256 = expected_code_sha256 or project_code_sha256(ROOT)
    if state.get("code_sha256") != expected_code_sha256:
        raise RuntimeError(
            "code changed since this run was created, or the checkpoint "
            "predates code fingerprinting; resume would mix training semantics. "
            "Use a new run_id."
        )
    adapters = state.get("adapters") or {}
    missing = [
        agent
        for agent in AGENTS[: config.atgrpo.agent_num]
        if not adapters.get(agent)
        or not adapter_checkpoint_complete(adapters[agent], require_optimizer=True)
    ]
    if missing:
        raise FileNotFoundError(
            "resume state has missing or incomplete adapter checkpoints "
            f"(model/config/optimizer required) for: {missing}"
        )
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--prescreen", default=None, help="allowlist jsonl")
    parser.add_argument("--tiny-model", default=None, help="smoke-test model dir")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir or ROOT / "runs" / config.task / config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_code_sha256 = project_code_sha256(ROOT)
    resume_state = (
        load_resume_state(
            config, run_dir, expected_code_sha256=run_code_sha256
        )
        if args.resume
        else None
    )

    task = build_task(
        config.task,
        **({"reward_mode": config.reward_mode} if config.task == "multipl_e" else {}),
        **{
            key: value
            for key, value in config.task_options.items()
            if not key.startswith("mock_")
        },
    )

    if args.max_model_len:
        config.serving.max_model_len = args.max_model_len
        print('[train] max_model_len override -> '
              + str(config.serving.max_model_len), flush=True)
    fleet = build_fleet(config, log_dir=run_dir / "logs")
    try:
        fleet.launch()
        fleet.wait_healthy()
        trainers = build_trainers(
            config,
            tiny_model=args.tiny_model,
            init_adapters=(resume_state or {}).get("adapters"),
        )
        loop = ATGRPOLoop(
            config,
            task,
            fleet,
            trainers,
            run_dir=run_dir,
            callers_factory=make_callers_factory(config, fleet),
            prescreen_path=Path(args.prescreen) if args.prescreen else None,
            code_sha256=run_code_sha256,
        )
        if not args.resume or not loop.logger.manifest_path.is_file():
            loop.logger.write_manifest(
                build_manifest(
                    config, root=ROOT, code_sha256=run_code_sha256
                )
            )
        if args.resume:
            if not loop.load_state():
                raise RuntimeError("resume state disappeared after trainer startup")
            fleet.publish_adapters(
                loop.iteration - 1,
                {
                    agent: Path(path)
                    for agent, path in (resume_state or {})["adapters"].items()
                },
            )
        loop.run(max_iterations=args.max_iterations or None)
    finally:
        fleet.shutdown()
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
