#!/usr/bin/env python3
"""Overlap MultiPL-E generation with evaluation of completed problems."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generator while evaluating completed MultiPL-E files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-completions", type=int, required=True)
    parser.add_argument("--eval-image", required=True)
    parser.add_argument("--docker-exec", default="docker")
    parser.add_argument("--eval-shards", type=int, default=64)
    parser.add_argument("--eval-inner-workers", type=int, default=1)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Start an evaluation round after this many problems are ready.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--quiet-evaluator",
        action="store_true",
        help="Suppress per-shard evaluator output while retaining failures.",
    )
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument(
        "--timings-file",
        type=Path,
        help="JSONL file receiving one evaluation-completion timing record per task.",
    )
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=Path(__file__).with_name("evaluate_multipl_e_parallel.py"),
    )
    parser.add_argument("generator", nargs=argparse.REMAINDER)
    return parser.parse_args()


def open_json(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode + "t")
    return path.open(mode)


def load_json(path: Path) -> dict:
    with open_json(path, "r") as handle:
        return json.load(handle)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def completion_files(input_dir: Path) -> list[Path]:
    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".results.json") or path.name.endswith(
            ".results.json.gz"
        ):
            continue
        if path.name.endswith(".json") or path.name.endswith(".json.gz"):
            files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(input_dir)))


def result_path(completion_path: Path) -> Path:
    if completion_path.name.endswith(".json.gz"):
        suffix = ".json.gz"
    elif completion_path.name.endswith(".json"):
        suffix = ".json"
    else:
        raise ValueError(f"Not a completion file: {completion_path}")
    name = completion_path.name[: -len(suffix)]
    return completion_path.with_name(name + ".results" + suffix)


def json_count(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    try:
        data = load_json(path)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
        return 0
    values = data.get(key)
    return len(values) if isinstance(values, list) else 0


def ready_files(
    input_dir: Path, expected_completions: int, scheduled: set[Path]
) -> list[Path]:
    ready = []
    for path in completion_files(input_dir):
        relative = path.relative_to(input_dir)
        if relative in scheduled:
            continue
        completion_count = json_count(path, "completions")
        if completion_count < expected_completions:
            continue
        existing_results = json_count(result_path(path), "results")
        if existing_results >= completion_count:
            scheduled.add(relative)
            continue
        ready.append(path)
    return ready


def write_manifest(paths: list[Path], input_dir: Path) -> Path:
    fd, name = tempfile.mkstemp(
        prefix=".multipl_e_pipeline_", suffix=".manifest", dir=input_dir.parent
    )
    os.close(fd)
    manifest = Path(name)
    manifest.write_text(
        "".join(f"{path.relative_to(input_dir)}\n" for path in paths)
    )
    return manifest


def run_evaluation_batch(
    paths: list[Path],
    args: argparse.Namespace,
    input_dir: Path,
    pipeline_started_monotonic: float,
) -> tuple[int, float]:
    manifest = write_manifest(paths, input_dir)
    command = [
        sys.executable,
        str(args.evaluator_script.resolve()),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(input_dir),
        "--image",
        args.eval_image,
        "--docker-exec",
        args.docker_exec,
        "--shards",
        str(min(args.eval_shards, len(paths))),
        "--inner-workers",
        str(args.eval_inner_workers),
        "--manifest",
        str(manifest),
    ]
    if args.quiet_evaluator:
        command.append("--quiet")
    print(
        f"Pipeline evaluation: files={len(paths)} shards={min(args.eval_shards, len(paths))}",
        flush=True,
    )
    started = time.monotonic()
    batch_started_at = now_iso()
    try:
        completed = subprocess.run(command, check=False)
        returncode = completed.returncode
    finally:
        try:
            manifest.unlink()
        except FileNotFoundError:
            pass
    if returncode == 0:
        for path in paths:
            expected_results = json_count(path, "completions")
            if json_count(result_path(path), "results") < expected_results:
                print(f"ERROR evaluator did not finish {path}", file=sys.stderr)
                returncode = 1
                break
    batch_elapsed = time.monotonic() - started
    batch_completed_at = now_iso()
    if returncode == 0:
        for path in paths:
            relative = path.relative_to(input_dir)
            try:
                completion_data = load_json(path)
            except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
                completion_data = {}
            completion_count = json_count(path, "completions")
            result_count = json_count(result_path(path), "results")
            parts = relative.parts
            append_jsonl(
                args.timings_file,
                {
                    "event": "evaluation_completed",
                    "root_dataset": parts[0] if len(parts) > 0 else None,
                    "language": completion_data.get(
                        "language", parts[1] if len(parts) > 1 else None
                    ),
                    "problem_id": completion_data.get("name", path.stem),
                    "completion_file": str(path),
                    "result_file": str(result_path(path)),
                    "completion_count": completion_count,
                    "result_count": result_count,
                    "batch_started_at": batch_started_at,
                    "completed_at": batch_completed_at,
                    "batch_elapsed_seconds": round(batch_elapsed, 3),
                    "total_elapsed_seconds": round(
                        time.monotonic() - pipeline_started_monotonic, 3
                    ),
                    "status": "evaluated",
                },
            )
    print(
        f"Pipeline evaluation finished: files={len(paths)} rc={returncode} "
        f"seconds={batch_elapsed:.2f} completed_at={batch_completed_at}",
        flush=True,
    )
    return returncode, batch_elapsed


def terminate(process: subprocess.Popen) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def write_metrics(
    path: Path | None,
    status: int,
    generation_seconds: float,
    evaluation_seconds: float,
    elapsed_seconds: float,
    generated_files: int,
    evaluated_files: int,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"STATUS={status}\n"
        f"GENERATION_SECONDS={generation_seconds:.3f}\n"
        f"EVALUATION_WORK_SECONDS={evaluation_seconds:.3f}\n"
        f"PIPELINE_SECONDS={elapsed_seconds:.3f}\n"
        f"GENERATED_FILES={generated_files}\n"
        f"EVALUATED_FILES={evaluated_files}\n"
    )


def main() -> int:
    args = parse_args()
    generator = list(args.generator)
    if generator and generator[0] == "--":
        generator = generator[1:]
    if not generator:
        raise SystemExit("A generator command is required after --")
    if args.expected_completions <= 0:
        raise SystemExit("--expected-completions must be positive")
    if args.eval_shards <= 0 or args.eval_inner_workers <= 0:
        raise SystemExit("Evaluation worker counts must be positive")
    if args.eval_batch_size <= 0 or args.poll_interval <= 0:
        raise SystemExit("--eval-batch-size and --poll-interval must be positive")

    input_dir = args.input_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    if args.timings_file is not None:
        args.timings_file = args.timings_file.resolve()
    started = time.monotonic()
    generation_started = started
    evaluation_work_seconds = 0.0
    evaluated_files = 0
    scheduled: set[Path] = set()
    status = 1
    process = None

    print("Starting generation/evaluation pipeline", flush=True)
    print("Generator:", " ".join(generator), flush=True)
    try:
        process = subprocess.Popen(generator, start_new_session=True)
        while True:
            process_status = process.poll()
            ready = ready_files(input_dir, args.expected_completions, scheduled)
            if ready and (
                len(ready) >= args.eval_batch_size or process_status is not None
            ):
                batch = ready if process_status is not None else ready[: args.eval_batch_size]
                eval_status, eval_seconds = run_evaluation_batch(
                    batch, args, input_dir, started
                )
                evaluation_work_seconds += eval_seconds
                if eval_status != 0:
                    print("ERROR: stopping generation after evaluator failure", file=sys.stderr)
                    terminate(process)
                    status = 1
                    break
                scheduled.update(path.relative_to(input_dir) for path in batch)
                evaluated_files += len(batch)
                continue

            if process_status is not None:
                generation_seconds = time.monotonic() - generation_started
                incomplete = [
                    path
                    for path in completion_files(input_dir)
                    if json_count(path, "completions") < args.expected_completions
                ]
                if process_status != 0:
                    print(
                        f"ERROR: generator exited with status {process_status}",
                        file=sys.stderr,
                    )
                    status = 1
                elif incomplete:
                    print(
                        f"ERROR: {len(incomplete)} completion files are incomplete",
                        file=sys.stderr,
                    )
                    status = 1
                else:
                    status = 0
                break
            time.sleep(args.poll_interval)
    except BaseException:
        if process is not None:
            terminate(process)
        raise
    finally:
        if process is not None and process.poll() is None:
            terminate(process)
        generation_seconds = time.monotonic() - generation_started
        elapsed_seconds = time.monotonic() - started
        write_metrics(
            args.metrics_file,
            status,
            generation_seconds,
            evaluation_work_seconds,
            elapsed_seconds,
            len(completion_files(input_dir)),
            evaluated_files,
        )
        print(
            f"Pipeline finished: status={status} generation_seconds={generation_seconds:.2f} "
            f"evaluation_work_seconds={evaluation_work_seconds:.2f} "
            f"elapsed_seconds={elapsed_seconds:.2f}",
            flush=True,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
