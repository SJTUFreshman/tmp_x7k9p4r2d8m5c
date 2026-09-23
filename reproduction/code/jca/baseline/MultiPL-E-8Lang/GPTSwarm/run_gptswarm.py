#!/usr/bin/env python3
"""Run GPTSwarm-Fixed on the MultiPL-E eight-language benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_GPTSWARM_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "GPTSwarm"
MULTIPL_E_SCRIPTS = PROJECT_ROOT / "Code" / "MultiPL-E" / "scripts"
for path in (PACKAGE_PARENT, MUSIQUE_GPTSWARM_DIR, MULTIPL_E_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import nodes as nodes_core  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from multipl_e_completion_adapter import (  # noqa: E402
    PROMPT_PROTOCOL,
    build_chat_user_prompt,
    normalize_completion,
)
from swarm import build_fixed_swarm, run_swarm_on_problem  # noqa: E402


SCHEMA_VERSION = 1
TASK_NAME = "MultiPL-E-8Lang"
ROUTING_POLICY = "type_capacity"
NODE_MODELS = {
    "io_0": "A1",
    "io_1": "A1",
    "cot_0": "A2",
    "cot_1": "A2",
    "cot_2": "A2",
    "debate_0": "A3",
    "aggregator": "A3",
}


@dataclass(frozen=True)
class MultiPLEProblem:
    id: str
    root_dataset: str
    language: str
    rendered_text: str
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"GPTSwarm-Fixed on {TASK_NAME}.")
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--timings-file", type=Path)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--max-problems-per-language", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--log-raw-chars", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
    return rows


def build_problem(root_dataset: str, language: str, row: dict[str, Any]) -> MultiPLEProblem:
    rendered = build_chat_user_prompt(
        language,
        row["prompt"],
        row.get("tests", ""),
        row.get("stop_tokens") or [],
    )
    return MultiPLEProblem(
        id=row["name"],
        root_dataset=root_dataset,
        language=language,
        rendered_text=rendered,
        row=row,
    )


def load_tasks(args: argparse.Namespace) -> list[MultiPLEProblem]:
    tasks = []
    seen = set()
    for manifest_arg in args.split_manifest:
        manifest_path = manifest_arg.resolve()
        manifest = load_json(manifest_path)
        root_dataset = manifest["root_dataset"]
        for language, details in sorted(manifest["languages"].items()):
            if args.languages and language not in args.languages:
                continue
            test_path = (manifest_path.parent / details["test_file"]).resolve()
            rows = read_jsonl(test_path)
            expected = {item["problem_id"] for item in details["test"]}
            if {row.get("name") for row in rows} != expected:
                raise ValueError(f"Manifest mismatch for {root_dataset}/{language}")
            if args.max_problems_per_language is not None:
                rows = rows[: args.max_problems_per_language]
            for row in rows:
                key = (root_dataset, language, row["name"])
                if key in seen:
                    raise ValueError(f"Duplicate task: {key}")
                seen.add(key)
                tasks.append(build_problem(root_dataset, language, row))
    return tasks


def configure_prompts() -> None:
    prompt_dir = THIS_DIR / "prompts"
    nodes_core.PROMPT_IO_PATH = prompt_dir / "node_io.md"
    nodes_core.PROMPT_COT_PATH = prompt_dir / "node_cot.md"
    nodes_core.PROMPT_DEBATE_PATH = prompt_dir / "node_debate.md"
    nodes_core.PROMPT_AGGREGATOR_PATH = prompt_dir / "node_aggregator.md"
    nodes_core._parse_node_json = parse_code_node_json


def parse_code_node_json(raw: str) -> dict[str, str] | None:
    payload = nodes_core._load_json_object(raw)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str) and answer.strip():
            return {"reasoning": reasoning.strip(), "answer": answer}
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL
    )
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL
    )
    if reasoning_match and answer_match:
        try:
            reasoning = json.loads(f'"{reasoning_match.group(1)}"')
            answer = json.loads(f'"{answer_match.group(1)}"')
        except json.JSONDecodeError:
            return None
        if isinstance(answer, str) and answer.strip():
            return {"reasoning": str(reasoning).strip(), "answer": answer}
    return None


def write_gzip(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    temporary.replace(path)


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completion_payload(problem: MultiPLEProblem, completion: str, status: str) -> dict[str, Any]:
    payload = dict(problem.row)
    payload.update(
        {
            "completions": [completion],
            "baseline": "gptswarm_fixed",
            "baseline_schema_version": SCHEMA_VERSION,
            "prompt_protocol": PROMPT_PROTOCOL,
            "routing_policy": ROUTING_POLICY,
            "thinking_mode": "disabled",
            "terminated_by": status,
        }
    )
    return payload


def run_one(problem: MultiPLEProblem, dag: Any, callers: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    node_callers = {node: callers[agent] for node, agent in NODE_MODELS.items()}
    record = run_swarm_on_problem(dag, problem, node_callers)
    raw_completion = record.final_answer or ""
    final_completion = ""
    error = record.error
    status = "exception" if error else "stop"
    if not error and raw_completion:
        try:
            final_completion = normalize_completion(
                raw_completion,
                problem.row["prompt"],
                problem.row.get("tests", ""),
                problem.row.get("stop_tokens") or [],
                problem.language,
            )
        except Exception as exc:
            error = f"Final completion normalization failed: {exc}"
            status = "exception"
    elif not error:
        error = "Aggregator produced an empty completion"
        status = "exception"

    trace = {
        "schema_version": SCHEMA_VERSION,
        "root_dataset": problem.root_dataset,
        "language": problem.language,
        "problem_id": problem.id,
        "layers": record.layers,
        "node_models": NODE_MODELS,
        "node_outputs": [asdict(output) for output in record.node_outputs],
        "raw_final_completion": raw_completion,
        "final_completion": final_completion,
        "terminated_by": status,
        "error": error,
        "wall_time_s": round(time.monotonic() - started, 3),
        "sampling": {
            "prompt_protocol": PROMPT_PROTOCOL,
            "thinking_mode": "disabled",
            "routing_policy": ROUTING_POLICY,
        },
    }
    return completion_payload(problem, final_completion, status), trace


def one_line(value: Any) -> str:
    return " ".join(str(value).split())


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.max_concurrency <= 0:
        raise SystemExit("--max-new-tokens and --max-concurrency must be positive")
    if args.max_problems_per_language is not None and args.max_problems_per_language <= 0:
        raise SystemExit("--max-problems-per-language must be positive")
    configure_prompts()
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No MultiPL-E tasks selected")
    dag = build_fixed_swarm()
    print(f"GPTSwarm-Fixed tasks={len(tasks)} task={TASK_NAME}")
    print(f"layers={dag._layers}")
    print(f"node_models={json.dumps(NODE_MODELS, sort_keys=True)}")
    print(f"thinking_mode=disabled routing_policy={ROUTING_POLICY}")
    print(f"prompt_protocol={PROMPT_PROTOCOL}")
    if args.dry_run:
        first = tasks[0]
        print(f"dry-run first={first.root_dataset}/{first.language}/{first.id}")
        return 0

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    callers = {
        "A1": OpenAIChatLLMCaller(
            args.api_base_a1, args.api_model_a1, generation=generation,
            timeout=args.api_timeout, response_format={"type": "json_object"},
        ),
        "A2": OpenAIChatLLMCaller(
            args.api_base_a2, args.api_model_a2, generation=generation,
            timeout=args.api_timeout, response_format={"type": "json_object"},
        ),
        "A3": OpenAIChatLLMCaller(
            args.api_base_a3, args.api_model_a3, generation=generation,
            timeout=args.api_timeout, response_format={"type": "json_object"},
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.trajectory_output.open("w", encoding="utf-8") as trace_handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(tasks))) as pool:
            futures = {pool.submit(run_one, task, dag, callers): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                problem = futures[future]
                try:
                    payload, trace = future.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    payload = completion_payload(problem, "", "exception")
                    trace = {
                        "schema_version": SCHEMA_VERSION,
                        "root_dataset": problem.root_dataset,
                        "language": problem.language,
                        "problem_id": problem.id,
                        "layers": [],
                        "node_models": NODE_MODELS,
                        "node_outputs": [],
                        "raw_final_completion": "",
                        "final_completion": "",
                        "terminated_by": "exception",
                        "error": error,
                        "wall_time_s": 0.0,
                    }
                output_path = (
                    args.output_dir / problem.root_dataset / problem.language
                    / f"{problem.id}.json.gz"
                )
                write_gzip(output_path, payload)
                trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_handle.flush()
                elapsed = time.monotonic() - started
                append_jsonl(
                    args.timings_file,
                    {
                        "event": "generation_completed",
                        "root_dataset": problem.root_dataset,
                        "language": problem.language,
                        "problem_id": problem.id,
                        "completion_file": str(output_path.resolve()),
                        "total_elapsed_seconds": round(elapsed, 3),
                        "status": trace["terminated_by"],
                    },
                )
                print(
                    f"[{index}/{len(tasks)}] {problem.root_dataset}/{problem.language}/"
                    f"{problem.id} status={trace['terminated_by']} "
                    f"final={one_line(trace['final_completion'])[:100]}",
                    flush=True,
                )
                if trace.get("error"):
                    print(f"  error={trace['error']}", flush=True)
    print(f"GPTSwarm generation finished in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
