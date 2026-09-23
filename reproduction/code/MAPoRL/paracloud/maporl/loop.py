from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import pickle
import random
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import AGENTS, Config
from .logging_utils import RunLogger, build_manifest, sha256_file, write_json_atomic
from .penalty import apply_penalty, evaluate_penalty, memoized_decoder
from .pool import PromptPool
from .reward_rules import shaped_rewards
from .rollout import run_debate, score_trajectories
from .serving import build_fleet
from .trajectory import write_trajectories
from .transport import GenerationOptions, OpenAIChatCaller


def runtime_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "maporl").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    paths += list((root / "vendor").glob("*.py")) + list((root / "configs").glob("*.yaml"))
    for suffix in ("*.py", "*.sh", "*.sbatch"):
        paths += list((root / "deploy").glob(suffix))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def validate_runtime(config: Config) -> None:
    config.validate()
    if config.mock:
        raise ValueError("The training entrypoint requires real models; use the CPU test suite for mock checks")
    if config.maporl.adapter_mode != "collaboration" or config.maporl.task_training:
        raise ValueError("The fleet runtime currently supports collaboration adapters with frozen turn 0 only")
    if config.maporl.agent_num < 2 or config.maporl.round_num < 2:
        raise ValueError("Collaboration training requires at least two agents and two rounds")
    if config.pool.prescreen:
        raise ValueError("MAPoRL prescreening is not implemented; use the configured unfiltered prompt pool")
    if config.serving.gpu_plan not in {"split44", "split66"}:
        raise ValueError("This runtime uses disjoint serving and training GPUs")
    serve = {index for value in config.serving.serve_gpus.values() for index in value.split(",")}
    train = {index for value in config.serving.train_gpus.values() for index in value.split(",")}
    if serve & train:
        raise ValueError("Serving and training GPUs must not overlap")
    if config.rollout.prompts_per_iter < 1 or config.rollout.max_concurrency < 1:
        raise ValueError("Training batch size and concurrency must be positive")
    if config.train.max_seq_length < config.serving.max_model_len:
        raise ValueError("Training context must cover the complete serving context without truncation")
    if config.budget.max_iterations < 1 or config.budget.max_wall_clock_hours <= 0:
        raise ValueError("Training budgets must be positive")


def build_callers(config: Config, fleet, names: dict[str, str], iteration: int):
    options = GenerationOptions(
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        max_new_tokens=config.rollout.response_length,
        enable_thinking=config.train.enable_thinking,
        force_fixed_length=config.rollout.force_fixed_length,
        guided_decoding=False,
        require_token_ids=True,
        logprobs_mode="processed_logprobs",
    )
    callers = {}
    for agent in AGENTS[:config.maporl.agent_num]:
        base = OpenAIChatCaller(
            fleet.base_urls()[agent], f"{agent}_base", generation=options,
            adapter_version="base", retries=0,
        )
        collaboration = OpenAIChatCaller(
            fleet.base_urls()[agent], names[agent], generation=options,
            adapter_version=str(iteration), retries=0,
        )
        for turn in range(config.maporl.round_num):
            callers[(turn, agent)] = base if turn == 0 else collaboration
    return callers


def _dataset_identity(task, problems) -> str:
    digest = hashlib.sha256()
    for problem in problems:
        payload = [problem.problem_id, problem.raw, task.user_prompt(problem)]
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_checkpoint(path: Path, config_sha256: str) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    files = manifest.get("files", {})
    if not files:
        raise ValueError(f"Checkpoint contains no recorded files: {path}")
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*")
              if item.is_file() and item != path / "manifest.json"}
    if actual != set(files):
        raise ValueError(f"Checkpoint file inventory mismatch: {path}")
    for relative, expected in files.items():
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(path) or sha256_file(candidate) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {relative}")
    return manifest


def _commit_checkpoint(staging: Path, destination: Path, config: Config, completed: int) -> None:
    files = {item.relative_to(staging).as_posix(): sha256_file(item)
             for item in sorted(staging.rglob("*")) if item.is_file()}
    write_json_atomic(staging / "manifest.json", {
        "files": files, "config_sha256": config.sha256(), "completed_iterations": completed,
    })
    if destination.exists():
        destination.rename(destination.with_name(f"orphaned_{destination.name}_{uuid.uuid4().hex}"))
    staging.rename(destination)


def _policy_paths(checkpoint: Path, agents) -> dict[str, Path]:
    return {agent: checkpoint / "policies" / agent for agent in agents}


def _new_staging(run_dir: Path) -> Path:
    staging = run_dir / "checkpoints" / f".building_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    return staging


def _trainer_factory(config, agent):
    from .agent import AgentTrainer
    return AgentTrainer(config, agent)


def _save_initial(config, run_dir, agents, trainer_factory) -> Path:
    staging = _new_staging(run_dir)
    try:
        for agent in agents:
            trainer = trainer_factory(config, agent)
            try:
                trainer.save_checkpoint(staging / "agents" / agent)
                trainer.export_policy(staging / "policies" / agent, turn=1)
            finally:
                trainer.close()
        destination = run_dir / "checkpoints" / "initial"
        _commit_checkpoint(staging, destination, config, 0)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def apply_rewards(config, task, problems, trajectories, tokenizers) -> dict[str, Any]:
    if len(trajectories) != len(problems) or not trajectories:
        raise ValueError("Rollout batch size mismatch")
    for problem, trajectory in zip(problems, trajectories, strict=True):
        if trajectory.error or not trajectory.complete or trajectory.problem_id != problem.problem_id:
            raise RuntimeError(f"Incomplete rollout for {problem.problem_id}: {trajectory.error}")
    score_trajectories(task, list(zip(problems, trajectories)), binarize=config.maporl.binarize)
    causes = Counter()
    all_records = [record for trajectory in trajectories for record in trajectory.records.values()]
    decoders = {agent: memoized_decoder(tokenizer) for agent, tokenizer in tokenizers.items()}
    for record in all_records:
        if record.prompt_token_ids is None or not record.response_token_ids:
            raise ValueError(f"Missing actual sampled tokens for {record.agent}, turn {record.turn}")
        if record.score is None or record.correctness is None:
            raise ValueError("Grader did not populate a trajectory score")
        if config.rollout.non_eos_penalty:
            penalty = evaluate_penalty(
                token_ids=record.response_token_ids,
                sequence_length=len(record.response_token_ids),
                answer_extracted=bool(record.answer), turn=record.turn,
                task_training=config.maporl.task_training,
                min_output_length=config.rollout.min_output_length,
                decode=decoders[record.agent],
            )
            record.penalized = not penalty.passed
            record.penalty_cause = penalty.cause
            record.score = apply_penalty(record.score, penalty, penalty_reward_value=config.rollout.penalty_reward_value)
            if record.penalized:
                causes[penalty.cause] += 1
    agents = AGENTS[:config.maporl.agent_num]
    scores = {(turn, index): [trajectory.get(turn, agent).score for trajectory in trajectories]
              for turn in range(config.maporl.round_num) for index, agent in enumerate(agents)}
    correctness = {(turn, index): [trajectory.get(turn, agent).correctness for trajectory in trajectories]
                   for turn in range(config.maporl.round_num) for index, agent in enumerate(agents)}
    rewards, diagnostics = shaped_rewards(
        scores_turn_agent=scores, correctnesses_turn_agent=correctness,
        total_rounds=config.maporl.round_num, total_agents=len(agents),
        finished_question=[trajectory.finished_turn for trajectory in trajectories],
        alpha=config.maporl.alpha, rule_horizon=config.maporl.rule_horizon,
        rule_agent_share=config.maporl.rule_agent_share, discount_factor=config.maporl.rule_discount,
        correct_threshold=config.maporl.correct_threshold, wrong_threshold=config.maporl.wrong_threshold,
        others_tiebreak=config.maporl.others_tiebreak, forward_sign=config.maporl.forward_sign,
    )
    for (turn, index), values in rewards.items():
        for trajectory, reward in zip(trajectories, values, strict=True):
            if reward is not None and not math.isfinite(reward):
                raise ValueError("Non-finite shaped reward")
            trajectory.get(turn, agents[index]).reward = reward
    per_turn = {str(turn): {agent: fmean(correctness[(turn, index)]) for index, agent in enumerate(agents)}
                for turn in range(config.maporl.round_num)}
    count = len(all_records)
    potential_bonus_events = max(1, 2 * len(trajectories) * len(agents) * (config.maporl.round_num - 1))
    valid_rewards = [record.reward for record in all_records if record.reward is not None]
    return {
        "n_prompts": len(problems), "n_records": count,
        "reward_mean": fmean(valid_rewards) if valid_rewards else 0.0,
        "protocol_failure_frac": sum(record.parsed is None for record in all_records) / count,
        "truncated_frac": sum(record.finish_reason == "length" for record in all_records) / count,
        "penalty_frac": sum(record.penalized for record in all_records) / count,
        "penalty_causes": dict(causes), "accuracy_by_turn_agent": per_turn,
        "delta_to_turn0": {str(turn): fmean(per_turn[str(turn)].values()) - fmean(per_turn["0"].values())
                           for turn in range(config.maporl.round_num)},
        "bonus_diagnostics": diagnostics,
        "bonus_deadzone_frac": (diagnostics["deadzone_backward"] + diagnostics["deadzone_forward"]) / potential_bonus_events,
        "max_prompt_tokens": max(len(record.prompt_token_ids) for record in all_records),
        "max_response_tokens": max(len(record.response_token_ids) for record in all_records),
    }


def _prune_checkpoints(run_dir: Path, keep: int, current: Path):
    candidates = sorted((run_dir / "checkpoints").glob("iter_[0-9][0-9][0-9][0-9]"))
    for path in candidates[:-max(1, keep)]:
        if path != current:
            shutil.rmtree(path)


def _finish_pending_evaluation(config, task, run_dir, state, fleet, names):
    pending = state.get("pending_eval")
    if pending is None:
        return
    from .evaluation import evaluate, build_eval_callers, resolve_checkpoint

    if pending["checkpoint"] != state["checkpoint"]:
        raise ValueError("Pending evaluation does not match the published checkpoint")
    _, _, identity = resolve_checkpoint(run_dir, "last")
    if identity != pending["checkpoint_identity"]:
        raise ValueError("Pending evaluation checkpoint identity changed")
    output_dir = (run_dir / pending["output_dir"]).resolve()
    if not output_dir.is_relative_to(run_dir / "eval"):
        raise ValueError("Pending evaluation output escapes the run evaluation directory")
    evaluation_problems = task.load("eval")[:config.budget.eval_subset]
    evaluate(
        config, task, evaluation_problems,
        build_eval_callers(
            config, fleet, policy_names=names, mode="maporl",
            adapter_version=identity["manifest_sha256"],
        ),
        output_dir=output_dir, checkpoint_identity=identity, mode="maporl",
    )
    state.pop("pending_eval")


def _wall_clock_limit_reached(previous_elapsed: float, started: float,
                              limit_hours: float | None) -> bool:
    """Return whether an enabled wall-clock limit has been consumed."""
    if limit_hours is None:
        return False
    return previous_elapsed + time.monotonic() - started >= limit_hours * 3600


def _resolve_target_iterations(config_max: int, max_iterations: int | None,
                               target_iterations: int | None) -> int:
    if target_iterations is not None:
        if isinstance(target_iterations, bool) or not isinstance(target_iterations, int) or target_iterations < 1:
            raise ValueError("target_iterations must be a positive integer")
        return target_iterations
    return min(config_max, max_iterations) if max_iterations is not None else config_max


def train(
    config: Config, run_dir: str | Path, *, resume: bool = False,
    max_iterations: int | None = None, fleet=None, perform_periodic_eval: bool = True,
    trainer_factory=None, task=None, tokenizers=None,
    ignore_wall_clock_limit: bool = False,
    target_iterations: int | None = None,
) -> dict[str, Any]:
    validate_runtime(config)
    from .tasks import build_task

    run_dir = Path(run_dir).resolve()
    logger = RunLogger(run_dir)
    agents = AGENTS[:config.maporl.agent_num]
    trainer_factory = trainer_factory or _trainer_factory
    task = task or build_task(config.task, **config.task_options, **({"reward_mode": config.reward_mode} if config.task == "multipl_e" and "reward_mode" not in config.task_options else {}))
    problems = task.load("train")
    dataset_sha = _dataset_identity(task, problems)
    pool = PromptPool(problems, seed=config.pool.shuffle_seed)
    state = logger.load_state()
    if resume:
        if state is None:
            raise ValueError("Resume requested but no committed state exists")
        if state["config_sha256"] != config.sha256() or state["dataset_sha256"] != dataset_sha:
            raise ValueError("Resume configuration or training dataset differs from the committed run")
        checkpoint = (run_dir / state["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(run_dir / "checkpoints"):
            raise ValueError("Checkpoint escapes the run directory")
        verify_checkpoint(checkpoint, config.sha256())
        pool.load_state(state["pool_state"])
        random.setstate(pickle.loads(base64.b64decode(state["python_rng"])))
    elif state is not None or logger.manifest_path.exists():
        raise FileExistsError("Run directory already contains training state; use --resume or a fresh directory")
    else:
        random.seed(config.seed)
        logger.write_manifest(build_manifest(config, root=Path(__file__).resolve().parents[1], extra={
            "dataset_sha256": dataset_sha, "runtime_sha256": runtime_fingerprint(),
        }))
    if tokenizers is None:
        from transformers import AutoTokenizer
        tokenizers = {
            agent: AutoTokenizer.from_pretrained(config.serving.models[agent], local_files_only=True)
            for agent in agents
        }
    owns_fleet = fleet is None
    fleet = fleet or build_fleet(config, log_dir=run_dir / "logs")
    started = time.monotonic()
    previous_elapsed = float(state["elapsed_seconds"]) if state else 0.0
    completed = int(state["completed_iterations"]) if state else 0
    target = _resolve_target_iterations(
        config.budget.max_iterations, max_iterations, target_iterations,
    )
    effective_wall_clock_limit = None if ignore_wall_clock_limit else config.budget.max_wall_clock_hours

    logger.log_invocation({
        "resume": resume,
        "ignore_wall_clock_limit": ignore_wall_clock_limit,
        "configured_max_wall_clock_hours": config.budget.max_wall_clock_hours,
        "effective_max_wall_clock_hours": effective_wall_clock_limit,
        "previous_elapsed_seconds": previous_elapsed,
        "completed_iterations": completed,
        "checkpoint": state.get("checkpoint") if state else None,
        "target_iterations": target,
        "requested_target_iterations": target_iterations,
        "legacy_max_iterations": max_iterations,
        "perform_periodic_eval": perform_periodic_eval,
        "config_sha256": config.sha256(),
        "runtime_sha256": runtime_fingerprint(),
    })
    logger.print(
        "Wall-clock stop disabled for this invocation"
        if effective_wall_clock_limit is None
        else f"Wall-clock stop enabled: {effective_wall_clock_limit:g} cumulative hours"
    )

    def save_elapsed():
        if state is not None:
            state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
            logger.save_state(state)

    persistent_trainers = None
    try:
        if owns_fleet:
            fleet.launch()
            fleet.wait_healthy()
        if state is None:
            checkpoint = _save_initial(config, run_dir, agents, trainer_factory)
            initial_state = {
                "checkpoint": checkpoint.relative_to(run_dir).as_posix(), "completed_iterations": 0,
                "config_sha256": config.sha256(), "dataset_sha256": dataset_sha,
                "elapsed_seconds": time.monotonic() - started, "pool_state": copy.deepcopy(pool.state()),
                "python_rng": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii"),
            }
            logger.save_state(initial_state)
            state = initial_state
        names = fleet.publish_adapters(completed, _policy_paths(checkpoint, agents))
        logger.print(f"Published checkpoint {checkpoint.name}; pool={len(problems)}, completed={completed}")
        _finish_pending_evaluation(config, task, run_dir, state, fleet, names)
        save_elapsed()
        if config.ppo.parallel_agent_updates:
            persistent_trainers = {agent: trainer_factory(config, agent) for agent in agents}
            for agent, trainer in persistent_trainers.items():
                trainer.load_checkpoint(checkpoint / "agents" / agent)
        for iteration in range(completed, target):
            if _wall_clock_limit_reached(previous_elapsed, started, effective_wall_clock_limit):
                logger.print("Training time budget reached")
                break
            batch = pool.next_batch(config.rollout.prompts_per_iter)
            callers = build_callers(config, fleet, names, iteration)
            logger.print(f"Iteration {iteration}: generating {len(batch)} full debates")
            trajectories = run_debate(
                task, batch, callers, agents=agents, round_num=config.maporl.round_num,
                seed=config.seed, iteration=iteration, step_retries=config.rollout.step_retries,
                max_workers=max(1, config.rollout.max_concurrency // len(agents)), fail_fast=True,
            )
            reward_summary = apply_rewards(config, task, batch, trajectories, tokenizers)
            write_trajectories(run_dir / "rollouts" / f"iter_{iteration:04d}.jsonl.gz", trajectories)
            logger.print(f"Iteration {iteration}: rewards={reward_summary['reward_mean']:.4f}, protocol_failures={reward_summary['protocol_failure_frac']:.3f}")
            staging = _new_staging(run_dir)
            metrics = {"iteration": iteration, "reward_summary": reward_summary, "agents": {}}
            try:
                def update_agent(agent):
                    trainer = persistent_trainers[agent] if persistent_trainers is not None else trainer_factory(config, agent)
                    try:
                        if persistent_trainers is None:
                            trainer.load_checkpoint(checkpoint / "agents" / agent)
                        records = [record for trajectory in trajectories for record in trajectory.agent_records(agent)]
                        result = trainer.update(records, iteration=iteration)
                        trainer.save_checkpoint(staging / "agents" / agent)
                        trainer.export_policy(staging / "policies" / agent, turn=1)
                        return agent, result
                    finally:
                        if persistent_trainers is None:
                            trainer.close()

                if persistent_trainers is not None:
                    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                        for agent, result in executor.map(update_agent, agents):
                            metrics["agents"][agent] = result
                            logger.print(f"Iteration {iteration}: {agent} update saved")
                else:
                    for agent in agents:
                        updated_agent, result = update_agent(agent)
                        metrics["agents"][updated_agent] = result
                        logger.print(f"Iteration {iteration}: {updated_agent} update saved")
                destination = run_dir / "checkpoints" / f"iter_{iteration:04d}"
                _commit_checkpoint(staging, destination, config, iteration + 1)
                checkpoint = destination
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            metrics["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
            next_state = {
                "checkpoint": checkpoint.relative_to(run_dir).as_posix(), "completed_iterations": iteration + 1,
                "config_sha256": config.sha256(), "dataset_sha256": dataset_sha,
                "elapsed_seconds": metrics["elapsed_seconds"], "pool_state": copy.deepcopy(pool.state()),
                "python_rng": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii"),
                "last_metrics": metrics,
            }
            if perform_periodic_eval and config.budget.eval_every > 0 and (iteration + 1) % config.budget.eval_every == 0:
                next_state["pending_eval"] = {
                    "checkpoint": next_state["checkpoint"],
                    "checkpoint_identity": {
                        "checkpoint": checkpoint.name,
                        "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
                        "config_sha256": config.sha256(),
                    },
                    "output_dir": f"eval/iter_{iteration:04d}/maporl",
                }
            logger.save_state(next_state)
            state = next_state
            logger.log_metrics(metrics)
            names = fleet.publish_adapters(iteration + 1, _policy_paths(checkpoint, agents))
            logger.print(f"Committed iteration {iteration}; published updated collaboration adapters")
            _finish_pending_evaluation(config, task, run_dir, state, fleet, names)
            save_elapsed()
            _prune_checkpoints(run_dir, config.budget.keep_last, checkpoint)
        return state
    finally:
        try:
            if persistent_trainers is not None:
                for trainer in persistent_trainers.values():
                    trainer.close()
            if owns_fleet:
                fleet.shutdown()
        finally:
            save_elapsed()
