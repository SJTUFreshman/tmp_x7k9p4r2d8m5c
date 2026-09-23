#!/usr/bin/env python3
"""Run training-free MAD on the MultiPL-E eight-language benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_MAD_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "MAD"
MULTIPL_E_SCRIPTS = PROJECT_ROOT / "Code" / "MultiPL-E" / "scripts"
for path in (PACKAGE_PARENT, MUSIQUE_MAD_DIR, MULTIPL_E_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mad as mad_core  # noqa: E402
from jca.src.agents import AGENT_IDS  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from multipl_e_completion_adapter import (  # noqa: E402
    PROMPT_PROTOCOL,
    build_chat_user_prompt,
    normalize_completion,
)


SCHEMA_VERSION = 1
TASK_NAME = "MultiPL-E-8Lang"
TIE_BREAK_PRIORITY = ("A3", "A2", "A1")


@dataclass(frozen=True)
class MultiPLEProblem:
    id: str
    root_dataset: str
    language: str
    rendered_text: str
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"MAD baseline on {TASK_NAME}.")
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--timings-file", type=Path)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--max-problems-per-language", type=int)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature-round0", type=float, default=0.9)
    parser.add_argument("--temperature-debate", type=float, default=0.3)
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
    return MultiPLEProblem(
        id=row["name"],
        root_dataset=root_dataset,
        language=language,
        rendered_text=build_chat_user_prompt(
            language,
            row["prompt"],
            row.get("tests", ""),
            row.get("stop_tokens") or [],
        ),
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
            rows = read_jsonl((manifest_path.parent / details["test_file"]).resolve())
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


def parse_code_mad_json(raw_output: str) -> tuple[str, str] | None:
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                payload, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str) and answer.strip():
            return reasoning.strip(), answer
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
    )
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
    )
    if reasoning_match and answer_match:
        try:
            reasoning = json.loads(f'"{reasoning_match.group(1)}"')
            answer = json.loads(f'"{answer_match.group(1)}"')
        except json.JSONDecodeError:
            return None
        if isinstance(answer, str) and answer.strip():
            return str(reasoning).strip(), answer
    return None


def configure_mad_core() -> None:
    mad_core.PROMPT_ROUND0_PATH = THIS_DIR / "prompts" / "round0.md"
    mad_core.PROMPT_DEBATE_PATH = THIS_DIR / "prompts" / "debate.md"
    mad_core.format_problem_as_prompt = lambda problem: problem.rendered_text
    mad_core._parse_mad_json = parse_code_mad_json


def stable_problem_seed(problem: MultiPLEProblem) -> int:
    key = f"{problem.root_dataset}:{problem.language}:{problem.id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def normalize_candidate(problem: MultiPLEProblem, answer: str) -> str:
    return normalize_completion(
        answer,
        problem.row["prompt"],
        problem.row.get("tests", ""),
        problem.row.get("stop_tokens") or [],
        problem.language,
    )


def vote_normalized_candidates(
    candidates: dict[str, str],
) -> tuple[str | None, str | None, bool]:
    non_empty = {
        agent: candidate for agent, candidate in candidates.items() if candidate.strip()
    }
    if not non_empty:
        return None, None, False
    buckets: dict[str, list[str]] = {}
    for agent, candidate in non_empty.items():
        buckets.setdefault(candidate, []).append(agent)
    max_count = max(len(agents) for agents in buckets.values())
    winners = [candidate for candidate, agents in buckets.items() if len(agents) == max_count]
    tie_break = len(winners) > 1
    for preferred in TIE_BREAK_PRIORITY:
        for candidate in winners:
            if preferred in buckets[candidate]:
                return candidate, preferred, tie_break
    candidate = winners[0]
    return candidate, buckets[candidate][0], tie_break


def completion_payload(problem: MultiPLEProblem, completion: str, status: str) -> dict[str, Any]:
    payload = dict(problem.row)
    payload.update(
        {
            "completions": [completion],
            "baseline": "mad",
            "baseline_schema_version": SCHEMA_VERSION,
            "prompt_protocol": PROMPT_PROTOCOL,
            "thinking_mode": "disabled",
            "terminated_by": status,
        }
    )
    return payload


def run_one(
    problem: MultiPLEProblem,
    callers_round0: dict[str, Any],
    callers_debate: dict[str, Any],
    n_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    record = mad_core.run_mad_for_problem(
        problem,
        callers_round0,
        callers_debate,
        n_rounds=n_rounds,
        tie_break_priority=list(TIE_BREAK_PRIORITY),
        seed=stable_problem_seed(problem),
    )
    raw_candidates: dict[str, str] = {}
    normalized_candidates: dict[str, str] = {}
    normalization_errors: dict[str, str] = {}
    final_round = n_rounds - 1
    for agent in AGENT_IDS:
        turn = record.turn_at(final_round, agent)
        if turn is None or not turn.answer:
            continue
        raw_candidates[agent] = turn.answer
        try:
            normalized = normalize_candidate(problem, turn.answer)
        except Exception as exc:
            normalization_errors[agent] = str(exc)
            continue
        if normalized.strip():
            normalized_candidates[agent] = normalized

    final_completion, winning_agent, tie_break = vote_normalized_candidates(
        normalized_candidates
    )
    error = record.error
    if final_completion is None and error is None:
        error = "No valid final-round continuation after normalization"
    status = "exception" if error else "stop"
    final_completion = final_completion or ""
    trace = {
        "schema_version": SCHEMA_VERSION,
        "root_dataset": problem.root_dataset,
        "language": problem.language,
        "problem_id": problem.id,
        "n_rounds": n_rounds,
        "turns": [asdict(turn) for turn in record.turns],
        "raw_voter_answers": raw_candidates,
        "normalized_voter_answers": normalized_candidates,
        "normalization_errors": normalization_errors,
        "winning_agent": winning_agent,
        "tie_break": tie_break,
        "final_completion": final_completion,
        "terminated_by": status,
        "error": error,
        "wall_time_s": round(time.monotonic() - started, 3),
        "sampling": {
            "prompt_protocol": PROMPT_PROTOCOL,
            "thinking_mode": "disabled",
            "tie_break_priority": list(TIE_BREAK_PRIORITY),
        },
    }
    return completion_payload(problem, final_completion, status), trace


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


def build_callers(args: argparse.Namespace, temperature: float) -> dict[str, Any]:
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    return {
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


def one_line(value: Any) -> str:
    return " ".join(str(value).split())


def main() -> int:
    args = parse_args()
    if args.n_rounds < 1 or args.max_new_tokens <= 0 or args.max_concurrency <= 0:
        raise SystemExit("--n-rounds, --max-new-tokens and --max-concurrency must be positive")
    if args.max_problems_per_language is not None and args.max_problems_per_language <= 0:
        raise SystemExit("--max-problems-per-language must be positive")
    configure_mad_core()
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No MultiPL-E tasks selected")
    print(f"MAD tasks={len(tasks)} task={TASK_NAME} n_rounds={args.n_rounds}")
    print(f"models=A1:1.7B,A2:4B,A3:8B tie_break={','.join(TIE_BREAK_PRIORITY)}")
    print("thinking_mode=disabled aggregation=normalized_exact_majority")
    print(f"prompt_protocol={PROMPT_PROTOCOL}")
    if args.dry_run:
        first = tasks[0]
        print(f"dry-run first={first.root_dataset}/{first.language}/{first.id}")
        return 0

    callers_round0 = build_callers(args, args.temperature_round0)
    callers_debate = build_callers(args, args.temperature_debate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.trajectory_output.open("w", encoding="utf-8") as trace_handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(tasks))) as pool:
            futures = {
                pool.submit(
                    run_one, task, callers_round0, callers_debate, args.n_rounds
                ): task
                for task in tasks
            }
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
                        "n_rounds": args.n_rounds,
                        "turns": [],
                        "raw_voter_answers": {},
                        "normalized_voter_answers": {},
                        "normalization_errors": {},
                        "winning_agent": None,
                        "tie_break": False,
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
                append_jsonl(
                    args.timings_file,
                    {
                        "event": "generation_completed",
                        "root_dataset": problem.root_dataset,
                        "language": problem.language,
                        "problem_id": problem.id,
                        "completion_file": str(output_path.resolve()),
                        "total_elapsed_seconds": round(time.monotonic() - started, 3),
                        "status": trace["terminated_by"],
                        "tie_break": trace["tie_break"],
                        "winning_agent": trace["winning_agent"],
                    },
                )
                print(
                    f"[{index}/{len(tasks)}] {problem.root_dataset}/{problem.language}/"
                    f"{problem.id} status={trace['terminated_by']} "
                    f"tie={trace['tie_break']} winner={trace['winning_agent']} "
                    f"final={one_line(trace['final_completion'])[:100]}",
                    flush=True,
                )
                if trace.get("error"):
                    print(f"  error={trace['error']}", flush=True)
    print(f"MAD generation finished in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
