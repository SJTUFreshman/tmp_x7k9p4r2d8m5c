#!/usr/bin/env python3
"""Generate SAS completions for fixed MultiPL-E test manifests.

The model is loaded once and reused across all datasets and languages. The
resulting files use the standard MultiPL-E completion schema so the existing
Docker evaluator can consume them directly.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

from multipl_e_completion_adapter import (
    CODE_COMPLETION_SYSTEM_PROMPT,
    PROMPT_PROTOCOL,
    build_chat_user_prompt,
    normalize_completion,
)

COMPLETION_ADAPTER = "qwen_code_continuation_v4"


@dataclass
class Task:
    root_dataset: str
    language: str
    row: dict
    data: dict
    output_path: Path
    completion_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MultiPL-E SAS completions from fixed test manifests."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--completion-limit", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Generate only this language; repeat the option to select several.",
    )
    parser.add_argument(
        "--max-problems-per-language",
        type=int,
        help="Limit each selected language for a smoke test.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled",),
        default="disabled",
        help="Qwen thinking is intentionally disabled for this benchmark.",
    )
    parser.add_argument(
        "--prompt-protocol",
        choices=(PROMPT_PROTOCOL,),
        default=PROMPT_PROTOCOL,
        help="Prompt formatting protocol used before vLLM generation.",
    )
    parser.add_argument(
        "--timings-file",
        type=Path,
        help="JSONL file receiving one generation-completion timing record per task.",
    )
    parser.add_argument("--revision", type=str)
    parser.add_argument("--tokenizer-name", type=str)
    parser.add_argument("--tokenizer-revision", type=str)
    return parser.parse_args()


def open_json(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode + "t")
    return path.open(mode, encoding="utf-8")


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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            rows.append(row)
    return rows


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with gzip.open(temporary_name, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def check_parameters(data: dict, args: argparse.Namespace, path: Path) -> None:
    checks = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking_mode": args.thinking_mode,
        "prompt_protocol": args.prompt_protocol,
        "completion_adapter": COMPLETION_ADAPTER,
        "model_path": str(args.model_path),
    }
    for key, expected in checks.items():
        if key not in data:
            raise ValueError(
                f"Existing {path} has no {key} metadata; use a fresh output directory"
            )
        if data[key] != expected:
            raise ValueError(
                f"Existing {path} has {key}={data[key]!r}, expected {expected!r}"
            )


def load_or_create_data(
    row: dict, output_path: Path, args: argparse.Namespace
) -> dict:
    if output_path.exists():
        data = load_json(output_path)
        check_parameters(data, args, output_path)
        if data.get("name") != row.get("name"):
            raise ValueError(f"Existing completion name mismatch: {output_path}")
        return data

    data = dict(row)
    data["temperature"] = args.temperature
    data["top_p"] = args.top_p
    data["max_tokens"] = args.max_tokens
    data["thinking_mode"] = args.thinking_mode
    data["prompt_protocol"] = args.prompt_protocol
    data["completion_adapter"] = COMPLETION_ADAPTER
    data["model_path"] = str(args.model_path)
    data["completions"] = []
    return data


def load_tasks(args: argparse.Namespace) -> list[list[Task]]:
    language_task_groups = []
    seen_roots = set()
    for manifest_path_arg in args.split_manifest:
        manifest_path = manifest_path_arg.resolve()
        manifest = load_json(manifest_path)
        root_dataset = manifest.get("root_dataset")
        if not isinstance(root_dataset, str) or not root_dataset:
            raise ValueError(f"Manifest has no root_dataset: {manifest_path}")
        if root_dataset in seen_roots:
            raise ValueError(f"Duplicate root_dataset manifest: {root_dataset}")
        seen_roots.add(root_dataset)

        split_root = manifest_path.parent
        for language, details in sorted(manifest["languages"].items()):
            if args.languages and language not in args.languages:
                continue
            test_path = (split_root / details["test_file"]).resolve()
            rows = read_jsonl(test_path)
            expected_ids = [item["problem_id"] for item in details["test"]]
            actual_ids = [row.get("name") for row in rows]
            if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(
                expected_ids
            ):
                raise ValueError(
                    f"Test file does not match manifest for {root_dataset}/{language}: "
                    f"{test_path}"
                )
            if len(actual_ids) != len(set(actual_ids)):
                raise ValueError(f"Duplicate problem name in {test_path}")
            if args.max_problems_per_language is not None:
                rows = rows[: args.max_problems_per_language]

            tasks = []
            for row in rows:
                problem_id = row["name"]
                output_path = (
                    args.output_dir
                    / root_dataset
                    / language
                    / f"{problem_id}.json.gz"
                )
                data = load_or_create_data(row, output_path, args)
                existing_count = len(data["completions"])
                missing = args.completion_limit - existing_count
                for completion_index in range(
                    existing_count, existing_count + max(0, missing)
                ):
                    tasks.append(
                        Task(
                            root_dataset=root_dataset,
                            language=language,
                            row=row,
                            data=data,
                            output_path=output_path,
                            completion_index=completion_index,
                        )
                    )
            language_task_groups.append(tasks)
    return language_task_groups


def generate_tasks(
    task_group: list[Task],
    model,
    args: argparse.Namespace,
    started_monotonic: float,
) -> int:
    if not task_group:
        return 0

    # Keep prompts with different language stop-token sets in separate requests.
    stop_groups: dict[tuple[str, ...], list[Task]] = {}
    for task in task_group:
        stop_tokens = list(task.row.get("stop_tokens") or [])
        stop_key = tuple(stop_tokens)
        stop_groups.setdefault(stop_key, []).append(task)

    generated = 0
    for stop_key, grouped_tasks in stop_groups.items():
        stop_tokens = list(stop_key)
        for start in range(0, len(grouped_tasks), args.batch_size):
            batch = grouped_tasks[start : start + args.batch_size]
            batch_started_monotonic = time.monotonic()
            batch_started_at = now_iso()
            print(
                f"Generating {batch[0].root_dataset}/{batch[0].language}: "
                f"{start + 1}-{start + len(batch)}/{len(grouped_tasks)}",
                flush=True,
            )
            raw_completions = model.completions(
                prompts=[
                    build_chat_user_prompt(
                        task.language,
                        task.row["prompt"],
                        task.row["tests"],
                        stop_tokens,
                    )
                    for task in batch
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=stop_tokens,
            )
            if len(raw_completions) != len(batch):
                raise RuntimeError(
                    f"Model returned {len(raw_completions)} completions for {len(batch)} prompts"
                )
            completions = [
                normalize_completion(
                    raw_completion,
                    task.row["prompt"],
                    task.row["tests"],
                    stop_tokens,
                    task.language,
                )
                for task, raw_completion in zip(batch, raw_completions)
            ]

            batch_completed_monotonic = time.monotonic()
            batch_completed_at = now_iso()
            batch_elapsed = batch_completed_monotonic - batch_started_monotonic
            total_elapsed = batch_completed_monotonic - started_monotonic

            modified = set()
            for task, completion in zip(batch, completions):
                task.data["completions"].append(completion)
                modified.add(task.output_path)
            for output_path in sorted(modified):
                write_json_atomic(
                    output_path,
                    next(task.data for task in batch if task.output_path == output_path),
                )
            for task, completion, raw_completion in zip(
                batch, completions, raw_completions
            ):
                append_jsonl(
                    args.timings_file,
                    {
                        "event": "generation_completed",
                        "root_dataset": task.root_dataset,
                        "language": task.language,
                        "problem_id": task.row["name"],
                        "completion_index": task.completion_index,
                        "completion_file": str(task.output_path),
                        "completed_at": batch_completed_at,
                        "batch_started_at": batch_started_at,
                        "batch_completed_at": batch_completed_at,
                        "batch_elapsed_seconds": round(batch_elapsed, 3),
                        "total_elapsed_seconds": round(total_elapsed, 3),
                        "generation_elapsed_seconds": round(total_elapsed, 3),
                        "raw_completion_chars": len(raw_completion),
                        "completion_chars": len(completion),
                        "temperature": args.temperature,
                        "thinking_mode": args.thinking_mode,
                        "prompt_protocol": args.prompt_protocol,
                        "completion_adapter": COMPLETION_ADAPTER,
                    },
                )
            print(
                f"Completed {batch[0].root_dataset}/{batch[0].language}: "
                f"{start + 1}-{start + len(batch)}/{len(grouped_tasks)} "
                f"batch_seconds={batch_elapsed:.2f} "
                f"total_elapsed_seconds={total_elapsed:.2f} "
                f"completed_at={batch_completed_at}",
                flush=True,
            )
            generated += len(batch)
    return generated


def main() -> int:
    args = parse_args()
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_path}")
    if args.num_gpus <= 0 or args.batch_size <= 0 or args.completion_limit <= 0:
        raise SystemExit("GPU, batch, and completion counts must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if (
        args.max_problems_per_language is not None
        and args.max_problems_per_language <= 0
    ):
        raise SystemExit("--max-problems-per-language must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if args.top_p <= 0 or args.top_p > 1:
        raise SystemExit("--top-p must be in (0, 1]")

    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.timings_file is not None:
        args.timings_file = args.timings_file.resolve()
    task_groups = load_tasks(args)
    total_tasks = sum(len(group) for group in task_groups)
    print(f"Pending completions: {total_tasks}", flush=True)
    if total_tasks == 0:
        print("All SAS completions already exist.", flush=True)
        return 0

    started_monotonic = time.monotonic()

    # This script lives below the repository root, where automodel_vllm.py is
    # kept for compatibility with the original MultiPL-E launcher.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from automodel_vllm import VLLM

    model = VLLM(
        str(args.model_path),
        args.revision,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        num_gpus=args.num_gpus,
        use_chat_template=True,
        enable_thinking=False,
        chat_system_prompt=CODE_COMPLETION_SYSTEM_PROMPT,
    )
    generated = 0
    for task_group in task_groups:
        generated += generate_tasks(task_group, model, args, started_monotonic)
    print(f"Generated completions: {generated}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
