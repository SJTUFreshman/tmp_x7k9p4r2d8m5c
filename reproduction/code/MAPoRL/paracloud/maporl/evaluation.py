"""Resumable, paired evaluation of the final debate and its initial drafts."""
from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import AGENTS, Config
from .logging_utils import sha256_file, write_json_atomic
from .rollout import run_debate_one
from .tasks.base import EvalResult, Problem, TaskAdapter
from .trajectory import DebateTrajectory
from .transport import GenerationOptions, OpenAIChatCaller


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def problem_seed(seed: int, problem_id: str) -> int:
    return int(_digest([int(seed), problem_id])[:16], 16) % (2**31 - 1)


def _cache_path(directory: Path, problem_id: str) -> Path:
    return directory / (_digest(problem_id) + ".json")


def _read_cache(path: Path, problem_id: str, identity: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("problem_id") != problem_id:
        raise ValueError(f"evaluation cache has a different problem id: {path}")
    if payload.get("evaluation_identity") != identity:
        raise ValueError(f"evaluation cache identity mismatch: {path}")
    return payload


def _valid_trajectory(trajectory: DebateTrajectory, config: Config) -> None:
    expected_agents = tuple(AGENTS[:config.maporl.agent_num])
    if trajectory.error:
        raise RuntimeError(f"{trajectory.problem_id}: {trajectory.error}")
    if (
        not trajectory.complete
        or trajectory.agents != expected_agents
        or trajectory.round_num != config.maporl.round_num
    ):
        raise RuntimeError(f"incomplete evaluation debate: {trajectory.problem_id}")


def _score_batch(
    task: TaskAdapter,
    problems: Sequence[Problem],
    trajectories: Sequence[DebateTrajectory],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str | None], int] = {}
    items: list[tuple[Problem, str | None]] = []
    turn0_answers: dict[str, str | None] = {}
    for problem, trajectory in zip(problems, trajectories, strict=True):
        turn0 = task.aggregate_answers(
            {record.agent: record.answer for record in trajectory.turn_records(0)}
        )
        turn0_answers[problem.problem_id] = turn0
        answers = [trajectory.final_answer, turn0]
        answers.extend(record.answer for record in trajectory.records.values())
        for answer in answers:
            key = (problem.problem_id, answer)
            if key not in unique:
                unique[key] = len(items)
                items.append((problem, answer))
    scores = task.score_answers(items)
    if len(scores) != len(items):
        raise RuntimeError(f"grader returned {len(scores)} scores for {len(items)} answers")
    for value, detail in scores:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"grader returned invalid evaluation score: {value!r}")
        if not isinstance(detail, dict):
            raise TypeError("grader score detail must be a dictionary")

    def result(problem_id: str, answer: str | None) -> EvalResult:
        value, detail = scores[unique[(problem_id, answer)]]
        return EvalResult(problem_id, answer, float(value), dict(detail))

    rows = []
    for problem, trajectory in zip(problems, trajectories, strict=True):
        for record in trajectory.records.values():
            graded = result(problem.problem_id, record.answer)
            record.score = graded.score
            record.score_detail = graded.detail
        rows.append({
            **asdict(result(problem.problem_id, trajectory.final_answer)),
            "turn0": asdict(result(problem.problem_id, turn0_answers[problem.problem_id])),
            "trajectory": trajectory.to_dict(),
        })
    return rows


def _metric_result(payload: Mapping[str, Any]) -> EvalResult:
    return EvalResult(
        problem_id=str(payload["problem_id"]),
        final_answer=payload.get("final_answer"),
        score=float(payload["score"]),
        detail=dict(payload.get("detail", {})),
    )


def evaluate(
    config: Config,
    task: TaskAdapter,
    problems: Sequence[Problem],
    callers: Mapping[Any, Any],
    *,
    output_dir: str | Path,
    checkpoint_identity: Mapping[str, Any],
    mode: str = "maporl",
    seed: int | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    """Evaluate supplied policies; reuse only caches with identical provenance."""
    config.validate()
    if mode not in {"maporl", "base"}:
        raise ValueError("evaluation mode must be maporl or base")
    if not checkpoint_identity:
        raise ValueError("checkpoint_identity must identify the evaluated weights")
    if not problems:
        raise ValueError("evaluation requires at least one problem")
    ids = [problem.problem_id for problem in problems]
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation problem ids must be unique")
    seed = config.seed if seed is None else int(seed)
    batch_size = config.rollout.prompts_per_iter if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    max_concurrency = config.rollout.max_concurrency if max_concurrency is None else max_concurrency
    if max_concurrency < 1:
        raise ValueError("evaluation max_concurrency must be positive")
    agents = AGENTS[:config.maporl.agent_num]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "config_sha256": config.sha256(),
        "config": config.to_dict(),
        "checkpoint": dict(checkpoint_identity),
        "mode": mode,
        "seed": seed,
        "task": task.name,
        "data_root": task.data_root,
        "task_options": task.options,
        "problem_ids": ids,
        "problems_sha256": _digest([
            {"problem": asdict(problem), "prompt": task.user_prompt(problem)}
            for problem in problems
        ]),
        "system_prompts": {agent: task.system_prompt(agent) for agent in agents},
        "response_schemas": {agent: task.response_schema(agent) for agent in agents},
        "guided_decoding": False,
        "logprobs_mode": "processed_logprobs",
        "require_token_ids": True,
    }
    identity = _digest(manifest)
    manifest["evaluation_identity"] = identity
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(manifest, default=str)):
            raise ValueError(
                f"evaluation provenance changed; use a different output directory: {output_dir}"
            )
    else:
        if any((output_dir / name).exists() for name in ("responses", "results")):
            raise ValueError(f"evaluation caches exist without a manifest: {output_dir}")
        write_json_atomic(manifest_path, manifest)
    response_dir = output_dir / "responses"
    result_dir = output_dir / "results"
    response_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    rows: dict[str, dict[str, Any]] = {}
    for problem in problems:
        path = _cache_path(result_dir, problem.problem_id)
        if path.is_file():
            row = _read_cache(path, problem.problem_id, identity)
            _valid_trajectory(DebateTrajectory.from_dict(row["trajectory"]), config)
            rows[problem.problem_id] = row
    resumed = len(rows)
    pending = [problem for problem in problems if problem.problem_id not in rows]
    print(f"eval {mode}: resumed {resumed}/{len(problems)} results; "
          f"batch_size={batch_size} max_concurrency={max_concurrency}", flush=True)

    def generate(problem: Problem) -> DebateTrajectory:
        path = _cache_path(response_dir, problem.problem_id)
        if path.is_file():
            payload = _read_cache(path, problem.problem_id, identity)
            trajectory = DebateTrajectory.from_dict(payload["trajectory"])
        else:
            trajectory = run_debate_one(
                task, problem, dict(callers), agents=agents,
                round_num=config.maporl.round_num,
                seed=problem_seed(seed, problem.problem_id),
                step_retries=config.rollout.step_retries,
            )
            _valid_trajectory(trajectory, config)
            write_json_atomic(path, {
                "problem_id": problem.problem_id,
                "evaluation_identity": identity,
                "trajectory": trajectory.to_dict(),
            })
        _valid_trajectory(trajectory, config)
        return trajectory

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        workers = max(1, min(len(batch), max_concurrency // len(agents)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            trajectories = list(pool.map(generate, batch))
        scored = _score_batch(task, batch, trajectories)
        for row in scored:
            row["evaluation_identity"] = identity
            write_json_atomic(_cache_path(result_dir, row["problem_id"]), row)
            _cache_path(response_dir, row["problem_id"]).unlink(missing_ok=True)
            rows[row["problem_id"]] = row
        write_json_atomic(output_dir / "progress.json", {
            "mode": mode, "completed": len(rows), "total": len(problems),
            "resumed": resumed, "seconds_this_call": time.monotonic() - started,
        })
        print(f"eval {mode}: {len(rows)}/{len(problems)} completed", flush=True)

    ordered = [rows[problem.problem_id] for problem in problems]
    results = [_metric_result(row) for row in ordered]
    turn0 = task.eval_metric([_metric_result(row["turn0"]) for row in ordered])
    metrics = task.eval_metric(results)
    cells: dict[tuple[int, str], list[EvalResult]] = {}
    protocol_failures = 0
    truncated = 0
    for row in ordered:
        trajectory = DebateTrajectory.from_dict(row["trajectory"])
        for record in trajectory.records.values():
            cells.setdefault((record.turn, record.agent), []).append(EvalResult(
                trajectory.problem_id, record.answer, float(record.score), record.score_detail
            ))
            protocol_failures += int(record.protocol_error is not None)
            truncated += int(record.finish_reason == "length")
    metric_keys = {
        task.headline_metric, "em", "f1", "numeric_f1", "mean_explicit_score",
        "mean_requirement_coverage", "all_explicit_pass_rate", "weighted_pass_at_1",
    }
    summary = {
        **metrics,
        "mode": mode,
        "mock": config.mock,
        "seed": seed,
        "checkpoint": dict(checkpoint_identity),
        "evaluation_identity": identity,
        "turn0": turn0,
        "delta_to_turn0": {
            key: float(metrics[key]) - float(turn0[key])
            for key in sorted(metric_keys & metrics.keys() & turn0.keys())
        },
        "per_turn_agent": {
            f"turn_{turn}": {
                agent: task.eval_metric(cells[(turn, agent)]) for agent in agents
            }
            for turn in range(config.maporl.round_num)
        },
        "protocol_failed_calls": protocol_failures,
        "truncated_calls": truncated,
        "resumed": resumed,
        "seconds_this_call": time.monotonic() - started,
        "runtime": {"batch_size": batch_size, "max_concurrency": max_concurrency},
    }
    if mode == "base":
        summary["base_single_round"] = turn0
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def build_eval_callers(
    config: Config,
    fleet: Any,
    *,
    policy_names: Mapping[str, str] | None = None,
    mode: str = "maporl",
    adapter_version: str | None = None,
) -> dict[tuple[int, str], OpenAIChatCaller]:
    if mode not in {"maporl", "base"}:
        raise ValueError("evaluation mode must be maporl or base")
    if config.maporl.adapter_mode != "collaboration" or config.maporl.task_training:
        raise ValueError("serving currently requires collaboration with task_training=false")
    if mode == "maporl" and not policy_names:
        raise ValueError("MAPORL evaluation requires published collaboration adapters")
    generation = GenerationOptions(
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        max_new_tokens=config.rollout.response_length,
        enable_thinking=config.train.enable_thinking,
        top_k=config.rollout.top_k,
        force_fixed_length=config.rollout.force_fixed_length,
        guided_decoding=False,
        logprobs_mode="processed_logprobs",
        require_token_ids=True,
    )
    urls = fleet.base_urls()
    return {
        (turn, agent): OpenAIChatCaller(
            urls[agent],
            f"{agent}_base" if mode == "base" or turn == 0 else policy_names[agent],
            generation=generation,
            adapter_version=(
                "base" if mode == "base" or turn == 0 else adapter_version
            ),
        )
        for turn in range(config.maporl.round_num)
        for agent in AGENTS[:config.maporl.agent_num]
    }


def resolve_checkpoint(
    run_dir: str | Path, iteration: str | int = "last"
) -> tuple[Path, int, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    if str(iteration) == "last":
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        checkpoint = (run_dir / state["checkpoint"]).resolve()
    elif str(iteration) == "initial":
        checkpoint = run_dir / "checkpoints" / "initial"
    else:
        number = int(iteration)
        if number < 0:
            raise ValueError("checkpoint iteration must be non-negative")
        checkpoint = run_dir / "checkpoints" / f"iter_{number:04d}"
    if not checkpoint.is_relative_to(run_dir / "checkpoints"):
        raise ValueError("checkpoint path must stay inside the run checkpoints directory")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError(f"checkpoint manifest has no files: {manifest_path}")
    for relative, expected in files.items():
        path = (checkpoint / relative).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError(f"invalid checkpoint manifest path: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {path}")
    for agent in AGENTS:
        policy_dir = checkpoint / "policies" / agent
        if not (policy_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"checkpoint policy missing: {policy_dir}")
        weights = list(policy_dir.glob("adapter_model.*"))
        if not weights:
            raise FileNotFoundError(f"checkpoint adapter weights missing: {policy_dir}")
        for path in [policy_dir / "adapter_config.json", *weights]:
            if path.relative_to(checkpoint).as_posix() not in files:
                raise ValueError(f"checkpoint policy is not covered by its manifest: {path}")
    number = -1 if checkpoint.name == "initial" else int(checkpoint.name.removeprefix("iter_"))
    return checkpoint, number, {
        "checkpoint": checkpoint.name,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": manifest.get("config_sha256"),
    }
