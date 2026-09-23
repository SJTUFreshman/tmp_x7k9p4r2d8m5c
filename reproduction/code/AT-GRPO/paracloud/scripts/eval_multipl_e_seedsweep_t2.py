#!/usr/bin/env python3
"""Continuously evaluate one frozen AT-GRPO checkpoint with successive seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STOP_REQUESTED = False


class StopRequested(BaseException):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, payload, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_identity(output, identity):
    identity = json.loads(json.dumps(identity, sort_keys=True))
    manifest = output / "manifest.json"
    fingerprint = canonical_hash(identity)
    if manifest.exists():
        existing = read_json(manifest)
        if existing.get("identity") != identity or existing.get("fingerprint") != fingerprint:
            raise RuntimeError("sweep identity mismatch: use a NEW output directory; existing results are immutable")
    else:
        unexpected = [path.name for path in output.iterdir() if path.name not in {".lock", "manifest.json.tmp"}]
        if unexpected:
            raise RuntimeError(f"output contains files without an identity manifest: {unexpected[:5]}")
        write_json(manifest, {"fingerprint": fingerprint, "identity": identity, "created_at": time.time()}, immutable=True)
    return fingerprint


def validate_coverage(problems, *, full=True):
    identifiers = [problem.problem_id for problem in problems]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("eval split must be nonempty and have unique problem IDs")
    cells = Counter((problem.meta["dataset"], problem.meta["language"]) for problem in problems)
    if full and (len(cells) != 16 or len({cell[0] for cell in cells}) != 2 or len({cell[1] for cell in cells}) != 8):
        raise ValueError(f"expected all 16 dataset/language cells, found {dict(cells)}")
    return {f"{dataset}/{language}": count for (dataset, language), count in sorted(cells.items())}


def record_path(seed_directory, index):
    return seed_directory / "trajectories" / f"{index:05d}.json"


def load_records(seed_directory, problems, fingerprint, seed):
    records = {}
    for path in sorted((seed_directory / "trajectories").glob("*.json")):
        record = read_json(path)
        index = record.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(problems):
            raise RuntimeError(f"invalid trajectory index: {path}")
        if (record.get("fingerprint") != fingerprint or record.get("seed") != seed
                or record.get("problem_id") != problems[index].problem_id
                or record.get("trajectory", {}).get("problem_id") != problems[index].problem_id
                or path != record_path(seed_directory, index) or index in records):
            raise RuntimeError(f"trajectory identity mismatch: {path}")
        records[index] = record
    return records


def failure_kind(trajectory):
    if trajectory.terminated_by == "exception":
        error = str(trajectory.error or "")
        if any(marker in error.lower() for marker in (
            "maximum context length", "longer than the maximum model length",
            "prompt is too long", "max_tokens must be at least 1",
        )):
            return "model_context_limit"
        raise RuntimeError(f"generation infrastructure failure for {trajectory.problem_id}: {error}")
    if not str(trajectory.final_answer or "").strip():
        return "model_empty_or_invalid_response"
    return None


def generate_one(task, problem, callers, config, seed):
    from atgrpo.rollout import run_group

    group = run_group(task, problem, callers, group_size=1, iteration=0,
                      t_max=config.rollout.t_max, seed=seed,
                      joint_mode=config.rollout.joint_mode,
                      step_retries=config.rollout.step_retries, max_workers=1)
    if len(group) != 1:
        raise RuntimeError(f"expected one trajectory for {problem.problem_id}")
    trajectory = group[0]
    return trajectory.to_dict(), failure_kind(trajectory)


def check_stop(output):
    if STOP_REQUESTED or (output / "STOP").exists():
        raise StopRequested()


def generate_seed(config, task, callers, problems, output, seed_directory, fingerprint, seed, concurrency):
    records = load_records(seed_directory, problems, fingerprint, seed)
    pending_indices = iter(index for index in range(len(problems)) if index not in records)
    pending = {}
    pool = ThreadPoolExecutor(max_workers=concurrency)
    started = time.time()

    def submit_next():
        index = next(pending_indices, None)
        if index is not None:
            pending[pool.submit(generate_one, task, problems[index], callers, config, seed)] = index

    try:
        for unused in range(concurrency):
            submit_next()
        while pending:
            check_stop(output)
            ready, unused = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                index = pending.pop(future)
                try:
                    trajectory, failure = future.result()
                except Exception as exc:
                    write_json(seed_directory / "infrastructure_failure.json", {
                        "seed": seed, "index": index, "problem_id": problems[index].problem_id,
                        "error": f"{type(exc).__name__}: {exc}", "time": time.time(),
                    })
                    raise
                record = {"fingerprint": fingerprint, "seed": seed, "index": index,
                          "problem_id": problems[index].problem_id, "trajectory": trajectory,
                          "model_failure": failure}
                write_json(record_path(seed_directory, index), record, immutable=True)
                records[index] = record
                if len(records) % 24 == 0 or len(records) == len(problems):
                    print(f"[sweep] seed={seed} generated={len(records)}/{len(problems)} elapsed={time.time() - started:.1f}s", flush=True)
                    write_json(output / "status.json", {"state": "generating", "seed": seed,
                               "generated": len(records), "total": len(problems), "updated_at": time.time()})
                check_stop(output)
                submit_next()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return records


def summarize(problems, results):
    if len(results) != len(problems) or [row["problem_id"] for row in results] != [problem.problem_id for problem in problems]:
        raise RuntimeError("scored result coverage/order differs from eval split")
    cells = defaultdict(list)
    languages = defaultdict(list)
    datasets = defaultdict(list)
    for problem, result in zip(problems, results):
        score = result["score"]
        if score not in (0.0, 1.0):
            raise ValueError(f"non-binary execution score for {problem.problem_id}")
        cells[f"{problem.meta['dataset']}/{problem.meta['language']}"].append(score)
        languages[problem.meta["language"]].append(score)
        datasets[problem.meta["dataset"]].append(score)

    def aggregate(groups):
        return {name: {"n": len(scores), "passed": sum(scores), "pass_at_1": sum(scores) / len(scores)}
                for name, scores in sorted(groups.items())}

    per_cell = aggregate(cells)
    return {"n": len(results), "passed": sum(row["score"] for row in results),
            "weighted_pass_at_1": sum(row["score"] for row in results) / len(results),
            "macro_cell_pass_at_1": sum(cell["pass_at_1"] for cell in per_cell.values()) / len(per_cell),
            "cells": per_cell, "languages": aggregate(languages), "datasets": aggregate(datasets),
            "model_failure_count": sum(bool(row["model_failure"]) for row in results),
            "model_failure_reasons": dict(Counter(row["model_failure"] for row in results if row["model_failure"])),
            "infrastructure_failure_count": 0}


def score_seed(task, problems, records, seed_directory, fingerprint, seed):
    from atgrpo.trajectory import JointTrajectory

    task.scratch_dir = str(seed_directory / "evaluation")
    executable = [(index, problem, JointTrajectory.from_dict(records[index]["trajectory"]))
                  for index, problem in enumerate(problems)]
    try:
        scored = task.team_reward_batch([(problem, trajectory) for unused, problem, trajectory in executable])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        def decoded(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value or "")

        diagnostic = {"seed": seed, "time": time.time(), "error": str(exc),
                      "stdout": decoded(exc.stdout), "stderr": decoded(exc.stderr)}
        write_json(seed_directory / "execution_failure.json", diagnostic)
        raise RuntimeError(f"execution infrastructure failure: {exc}; stderr={diagnostic['stderr'][-4000:]}") from exc
    if len(scored) != len(executable):
        raise RuntimeError("execution scorer returned incomplete coverage")
    scored_by_index = {index: score for (index, unused, trajectory), score in zip(executable, scored)}
    results = []
    for index, problem in enumerate(problems):
        record = records[index]
        score, detail = scored_by_index[index]
        if not detail.get("executed") or detail.get("status") is None:
            raise RuntimeError(f"missing execution status for {problem.problem_id}")
        results.append({"problem_id": problem.problem_id, "score": float(score), "detail": detail,
                        "model_failure": record["model_failure"], "final_answer": record["trajectory"]["final_answer"]})
    metrics = summarize(problems, results)
    payload = {"fingerprint": fingerprint, "seed": seed, "arm": "trained", "metrics": metrics, "results": results}
    write_json(seed_directory / "results.json", payload, immutable=True)
    return payload


def build_identity(config, problems, adapters, args):
    source_paths = [Path(__file__).resolve(), ROOT / "vendor/multipl_e_completion_adapter.py"]
    source_paths.extend(sorted((ROOT / "atgrpo").rglob("*.py")))
    source_paths.extend(sorted((ROOT / "deploy").glob("multipl_e_myserver*.py")))
    adapter_files = {}
    for agent, directory in adapters.items():
        files = [directory / "adapter_config.json"] + sorted(directory.glob("adapter_model.*"))
        if len(files) < 2 or not all(path.is_file() for path in files):
            raise FileNotFoundError(f"incomplete {agent} checkpoint: {directory}")
        adapter_files[agent] = {str(path.resolve()): sha256_file(path) for path in files}
    models = {}
    for agent, directory in config.serving.models.items():
        directory = Path(directory)
        files = sorted(path for path in directory.iterdir() if path.is_file())
        models[agent] = {str(path.resolve()): (sha256_file(path) if path.suffix not in {".safetensors", ".bin"}
                         else {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}) for path in files}
    image = Path(config.task_options["image"])
    expected_sha = os.environ.get("MULTIPLE_CANONICAL_SIF_SHA256", "")
    if len(expected_sha) != 64 or sha256_file(image) != expected_sha:
        raise RuntimeError("canonical evaluator image SHA256 missing or does not match")
    return {"schema_version": 1, "config": asdict(config), "iteration": args.iteration,
            "first_seed": args.first_seed, "temperature": config.rollout.temperature, "top_p": 1.0,
            "seed_semantics": "run_group(seed=seed,iteration=0,group_size=1) independently for each problem",
            "max_concurrency": args.max_concurrency, "limit": args.limit,
            "coverage": validate_coverage(problems, full=not args.limit),
            "data_sha256": sha256_file(Path(config.task_options["data_root"]) / "test.jsonl"),
            "problem_ids_sha256": canonical_hash([problem.problem_id for problem in problems]),
            "adapters": adapter_files, "models": models,
            "evaluator_image": str(image.resolve()), "evaluator_image_sha256": expected_sha,
            "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}}


def main():
    import fcntl
    from atgrpo.config import AGENTS, load_config
    from atgrpo.serving import build_fleet
    from atgrpo.tasks import build_task
    from atgrpo.transport import GenerationOptions, build_openai_caller

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int, default=119)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.first_seed < 0 or args.max_concurrency < 1 or args.limit < 0 or args.max_seeds < 0:
        parser.error("invalid seed, concurrency or limit")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / ".lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another sweep worker owns {output}")
    config = load_config(args.config)
    if config.task != "multipl_e" or config.reward_mode != "execution" or config.mock:
        raise ValueError("this sweep requires real MultiPL-E execution evaluation")
    if (config.rollout.t_max != 3 or config.rollout.max_new_tokens != 1536
            or config.rollout.step_retries != 1 or config.train.enable_thinking):
        raise ValueError("unexpected generation protocol settings")
    adapters = {agent: Path(args.run_dir) / "adapters" / agent / f"iter_{args.iteration:04d}" for agent in AGENTS}
    task = build_task("multipl_e", reward_mode="execution", **config.task_options)
    problems = task.load("eval")
    if args.limit:
        problems = problems[:args.limit]
    fingerprint = ensure_identity(output, build_identity(config, problems, adapters, args))
    if args.preflight_only:
        print(f"Verified MultiPL-E iteration={args.iteration}, problems={len(problems)}, temperature={config.rollout.temperature}, fingerprint={fingerprint}", flush=True)
        return 0
    fleet = None

    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        raise StopRequested()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    seed = args.first_seed
    completed_seeds = 0
    try:
        check_stop(output)
        log_directory = output / "logs" / f"{int(time.time())}_{os.getpid()}"
        fleet = build_fleet(config, log_dir=log_directory)
        fleet.launch()
        fleet.wait_healthy()
        names = fleet.publish_adapters(args.iteration, adapters)
        generation = GenerationOptions(temperature=config.rollout.temperature, top_p=1.0, max_new_tokens=1536, enable_thinking=False)
        callers = {agent: build_openai_caller(fleet.base_urls()[agent], names[agent], generation=generation) for agent in AGENTS}
        while not args.max_seeds or completed_seeds < args.max_seeds:
            check_stop(output)
            directory = output / f"seed_{seed:06d}"
            directory.mkdir(exist_ok=True)
            result_path = directory / "results.json"
            if result_path.exists():
                payload = read_json(result_path)
                if payload.get("fingerprint") != fingerprint or payload.get("seed") != seed:
                    raise RuntimeError(f"completed seed identity mismatch: {result_path}")
                summarize(problems, payload["results"])
            else:
                print(f"[sweep] start seed={seed} n={len(problems)}", flush=True)
                started = time.time()
                records = generate_seed(config, task, callers, problems, output, directory, fingerprint, seed, args.max_concurrency)
                check_stop(output)
                write_json(output / "status.json", {"state": "scoring", "seed": seed, "generated": len(records), "total": len(problems), "updated_at": time.time()})
                payload = score_seed(task, problems, records, directory, fingerprint, seed)
                print(f"[sweep] complete seed={seed} weighted={payload['metrics']['weighted_pass_at_1']:.6f} macro16={payload['metrics']['macro_cell_pass_at_1']:.6f} seconds={time.time() - started:.1f}", flush=True)
            write_json(output / "status.json", {"state": "seed_complete", "seed": seed, "metrics": payload["metrics"], "updated_at": time.time()})
            completed_seeds += 1
            seed += 1
        return 0
    except StopRequested:
        write_json(output / "status.json", {"state": "stopped", "seed": seed, "resumable": True, "updated_at": time.time()})
        print(f"[sweep] stopped at seed={seed}; completed trajectories are preserved", flush=True)
        return 0
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", "seed": seed, "error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        raise
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(signum, signal.SIG_IGN)
        if fleet is not None:
            fleet.shutdown()
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
