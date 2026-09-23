#!/usr/bin/env python3
"""Run the canonical my-server evaluator in pinned Apptainer shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


@dataclass
class ShardResult:
    index: int
    file_count: int
    returncode: int
    elapsed: float
    output: str
    output_dir: Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def result_path(completion_path: Path) -> Path:
    if completion_path.name.endswith(".json.gz"):
        suffix = ".json.gz"
    elif completion_path.name.endswith(".json"):
        suffix = ".json"
    else:
        raise ValueError(f"Not a completion file: {completion_path}")
    name = completion_path.name[: -len(suffix)]
    return completion_path.with_name(name + ".results" + suffix)


def load_json(path: Path) -> dict:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def json_count(path: Path, key: str) -> int:
    try:
        values = load_json(path).get(key, [])
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
        return 0
    return len(values) if isinstance(values, list) else 0


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def write_timing_records(
    paths: list[Path],
    input_dir: Path,
    timing_file: Path | None,
    completed_at: str,
    batch_elapsed: float,
    total_elapsed: float,
) -> None:
    for path in paths:
        relative = path.relative_to(input_dir)
        try:
            completion_data = load_json(path)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
            completion_data = {}
        parts = relative.parts
        append_jsonl(
            timing_file,
            {
                "event": "evaluation_completed",
                "root_dataset": parts[0] if len(parts) > 0 else None,
                "language": completion_data.get(
                    "language", parts[1] if len(parts) > 1 else None
                ),
                "problem_id": completion_data.get("name", path.stem),
                "completion_file": str(path),
                "result_file": str(result_path(path)),
                "completion_count": json_count(path, "completions"),
                "result_count": json_count(result_path(path), "results"),
                "completed_at": completed_at,
                "batch_elapsed_seconds": round(batch_elapsed, 3),
                "total_elapsed_seconds": round(total_elapsed, 3),
                "status": "evaluated",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MultiPL-E completion files in parallel containers."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker-exec", default=os.environ.get("APPTAINER_BIN", "apptainer"))
    parser.add_argument("--sif-sha256", default=os.environ.get("MULTIPLE_CANONICAL_SIF_SHA256"))
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional newline-delimited list of completion paths relative to --input-dir.",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=1,
        help="Workers used by the upstream evaluator inside each container.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary shard directories for debugging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print batch-level progress and failures, but suppress per-shard output.",
    )
    parser.add_argument(
        "--timings-file",
        type=Path,
        help="JSONL file receiving one evaluation-completion timing record per task.",
    )
    return parser.parse_args()


def completion_files(input_dir: Path, manifest: Path | None = None) -> list[Path]:
    if manifest is not None:
        selected = []
        for line in manifest.read_text().splitlines():
            relative = Path(line.strip())
            if not line.strip():
                continue
            path = (relative if relative.is_absolute() else input_dir / relative).resolve()
            try:
                path.relative_to(input_dir)
            except ValueError as exc:
                raise SystemExit(f"Manifest path is outside input directory: {path}") from exc
            if (
                path.is_file()
                and not path.name.endswith(".results.json")
                and not path.name.endswith(".results.json.gz")
                and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
            ):
                selected.append(path)
        return sorted(set(selected), key=lambda path: str(path.relative_to(input_dir)))

    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".results.json") or path.name.endswith(".results.json.gz"):
            continue
        if path.name.endswith(".json") or path.name.endswith(".json.gz"):
            files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(input_dir)))


def stage_shard(
    shard_root: Path,
    shard_index: int,
    files: list[Path],
    input_dir: Path,
) -> tuple[Path, Path]:
    shard_input = shard_root / f"shard_{shard_index:03d}" / "input"
    shard_output = shard_root / f"shard_{shard_index:03d}" / "output"
    for source in files:
        relative = source.relative_to(input_dir)
        target = shard_input / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    shard_output.mkdir(parents=True, exist_ok=True)
    return shard_input, shard_output


def run_shard(
    index: int,
    files: list[Path],
    input_dir: Path,
    shard_root: Path,
    docker_exec: str,
    image: str,
    inner_workers: int,
) -> ShardResult:
    shard_input, shard_output = stage_shard(shard_root, index, files, input_dir)
    command = [
        docker_exec,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-mount",
        "hostfs,bind-paths",
        "--net",
        "--network",
        "none",
        "--pwd",
        "/code",
        "--bind",
        f"{shard_input.resolve()}:/inputs:ro",
        "--bind",
        f"{shard_output.resolve()}:/outputs:rw",
        image,
        "python3",
        "/code/main.py",
        "--dir",
        "/inputs",
        "--output-dir",
        "/outputs",
        "--recursive",
        "--max-workers",
        str(inner_workers),
    ]
    started = time.monotonic()
    child_environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("APPTAINERENV_", "SINGULARITYENV_"))
    }
    helper_directories = [
        "/data/apps/apptainer/apptainer/x86_64/libexec/apptainer/bin",
        "/data/apps/apptainer/apptainer/x86_64/utils/bin",
    ]
    child_environment["PATH"] = os.pathsep.join(helper_directories + [child_environment.get("PATH", "")])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=child_environment,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
    except OSError as exc:
        output = f"Failed to start evaluator: {exc}"
        returncode = 127
    return ShardResult(
        index=index,
        file_count=len(files),
        returncode=returncode,
        elapsed=time.monotonic() - started,
        output=output,
        output_dir=shard_output,
    )


def merge_results(shard_output: Path, output_dir: Path) -> int:
    merged = 0
    for source in shard_output.rglob("*"):
        if not source.is_file():
            continue
        if not (
            source.name.endswith(".results.json")
            or source.name.endswith(".results.json.gz")
        ):
            continue
        relative = source.relative_to(shard_output)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        merged += 1
    return merged


def main() -> int:
    args = parse_args()
    if args.shards <= 0 or args.inner_workers <= 0:
        raise SystemExit("--shards and --inner-workers must be positive")

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise SystemExit(f"Canonical SIF is missing: {image_path}")
    expected_sha256 = str(args.sif_sha256 or "").lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise SystemExit("Set MULTIPLE_CANONICAL_SIF_SHA256 to the verified canonical SIF SHA256")
    image_digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            image_digest.update(chunk)
    actual_sha256 = image_digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"Canonical SIF SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    args.image = str(image_path)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.timings_file is not None:
        args.timings_file = args.timings_file.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = completion_files(input_dir, args.manifest)
    if not files:
        raise SystemExit(f"No completion files found under {input_dir}")

    shard_count = min(args.shards, len(files))
    shards = [files[index::shard_count] for index in range(shard_count)]
    temp_root = Path(
        tempfile.mkdtemp(prefix=".multipl_e_eval_", dir=str(output_dir.parent))
    )
    started = time.monotonic()
    results: list[ShardResult] = []
    print(
        f"Parallel evaluator: files={len(files)} shards={shard_count} "
        f"inner_workers={args.inner_workers} image={args.image}",
        flush=True,
    )

    try:
        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            futures = [
                executor.submit(
                    run_shard,
                    index,
                    shard_files,
                    input_dir,
                    temp_root,
                    args.docker_exec,
                    args.image,
                    args.inner_workers,
                )
                for index, shard_files in enumerate(shards)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if not args.quiet or result.returncode != 0:
                    print(
                        f"Shard {result.index + 1}/{shard_count}: "
                        f"files={result.file_count} rc={result.returncode} "
                        f"seconds={result.elapsed:.2f}",
                        flush=True,
                    )
                if result.output and (not args.quiet or result.returncode != 0):
                    print(result.output, end="", flush=True)
                if result.returncode == 0:
                    merged = merge_results(result.output_dir, output_dir)
                    write_timing_records(
                        shards[result.index],
                        input_dir,
                        args.timings_file,
                        now_iso(),
                        result.elapsed,
                        time.monotonic() - started,
                    )
                    if not args.quiet:
                        print(
                            f"Shard {result.index + 1}: merged_results={merged}",
                            flush=True,
                        )

        failures = [result for result in results if result.returncode != 0]
        if failures:
            raise SystemExit(
                f"{len(failures)} evaluator shard(s) failed; "
                "successful shard results were preserved."
            )
        print(f"Parallel evaluation took {time.monotonic() - started:.2f} seconds")
        return 0
    finally:
        if args.keep_temp:
            print(f"Temporary evaluator shards: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
