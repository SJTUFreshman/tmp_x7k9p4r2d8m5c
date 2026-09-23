#!/usr/bin/env python3
"""Run training-free AgentVerse on the MultiPL-E eight-language benchmark."""

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
MUSIQUE_AGENTVERSE_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "AgentVerse"
MULTIPL_E_SCRIPTS = PROJECT_ROOT / "Code" / "MultiPL-E" / "scripts"
for path in (PACKAGE_PARENT, MUSIQUE_AGENTVERSE_DIR, MULTIPL_E_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agentverse as core  # noqa: E402
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
    parser = argparse.ArgumentParser(description=f"AgentVerse on {TASK_NAME}.")
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--timings-file", type=Path)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--max-problems-per-language", type=int)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--score-threshold", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature-agent", type=float, default=0.7)
    parser.add_argument("--temperature-meta", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-base-meta", default="http://127.0.0.1:8204/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-model-meta", default="Meta_8B")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
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


def parse_code_agent_json(raw_output: str) -> tuple[str, str] | None:
    """Parse agent JSON while preserving continuation whitespace exactly."""
    payload = core._load_json_object(raw_output)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str) and answer.strip():
            return reasoning.strip(), answer
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL
    )
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL
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


def configure_core() -> None:
    core.PROMPT_RECRUITER_PATH = THIS_DIR / "prompts" / "recruiter.md"
    core.PROMPT_AGENT_PATH = THIS_DIR / "prompts" / "agent.md"
    core.PROMPT_EVALUATOR_PATH = THIS_DIR / "prompts" / "evaluator.md"
    core.format_problem_as_prompt = lambda problem: problem.rendered_text
    core._parse_agent_json = parse_code_agent_json


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


def run_protocol(
    problem: MultiPLEProblem,
    callers_agents: dict[str, Any],
    caller_meta: Any,
    max_iterations: int = 3,
    score_threshold: int = 8,
) -> dict[str, Any]:
    started = time.monotonic()
    trace: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "root_dataset": problem.root_dataset,
        "language": problem.language,
        "problem_id": problem.id,
        "max_iterations": max_iterations,
        "score_threshold": score_threshold,
        "roles": [],
        "iterations": [],
        "final_completion": "",
        "winning_agent": None,
        "tie_break": False,
        "terminated_by": "running",
        "error": None,
    }
    roles, raw, retried, raw_outputs = core._stage_recruit(caller_meta, problem)
    trace.update(
        recruit_raw_output=raw,
        recruit_raw_outputs=raw_outputs,
        recruit_retried=retried,
        recruit_parse_ok=roles is not None,
    )
    if roles is None:
        trace["terminated_by"] = "recruit_failed"
        trace["error"] = "Recruiter output could not be parsed after retry"
        trace["wall_time_s"] = round(time.monotonic() - started, 3)
        return trace
    trace["roles"] = [asdict(role) for role in roles]
    prior_answers: list[Any] = []
    prior_evaluation: Any = None
    for iteration in range(max_iterations):
        answers = core._stage_decide(
            callers_agents,
            roles,
            problem,
            iteration,
            prior_answers,
            prior_evaluation,
        )
        raw_candidates = {answer.agent_id: answer.answer for answer in answers}
        normalized_candidates: dict[str, str] = {}
        normalization_errors: dict[str, str] = {}
        for agent_id, answer in raw_candidates.items():
            if not answer.strip():
                continue
            try:
                normalized = normalize_candidate(problem, answer)
            except Exception as exc:
                normalization_errors[agent_id] = f"{type(exc).__name__}: {exc}"
                continue
            if normalized.strip():
                normalized_candidates[agent_id] = normalized
        team_answer, winning_agent, tie_break = vote_normalized_candidates(
            normalized_candidates
        )
        evaluation = core._stage_evaluate(
            caller_meta, problem, iteration, answers, team_answer
        )
        trace["iterations"].append(
            {
                "iteration": iteration,
                "answers": [asdict(answer) for answer in answers],
                "raw_candidates": raw_candidates,
                "normalized_candidates": normalized_candidates,
                "normalization_errors": normalization_errors,
                "team_answer": team_answer,
                "winning_agent": winning_agent,
                "tie_break": tie_break,
                "evaluation": asdict(evaluation),
            }
        )
        trace["final_completion"] = team_answer or ""
        trace["winning_agent"] = winning_agent
        trace["tie_break"] = tie_break
        if evaluation.parse_ok and evaluation.score >= score_threshold:
            trace["terminated_by"] = "score_threshold"
            break
        prior_answers = answers
        prior_evaluation = evaluation
    else:
        trace["terminated_by"] = "max_iterations"
    if not trace["final_completion"] and trace["error"] is None:
        trace["error"] = "No valid continuation after normalization"
    trace["wall_time_s"] = round(time.monotonic() - started, 3)
    return trace


def completion_payload(problem: MultiPLEProblem, trace: dict[str, Any]) -> dict[str, Any]:
    payload = dict(problem.row)
    payload.update(
        {
            "completions": [trace["final_completion"]],
            "baseline": "agentverse",
            "baseline_schema_version": SCHEMA_VERSION,
            "prompt_protocol": PROMPT_PROTOCOL,
            "thinking_mode": "disabled",
            "terminated_by": trace["terminated_by"],
        }
    )
    return payload


def run_one(
    problem: MultiPLEProblem,
    callers_agents: dict[str, Any],
    caller_meta: Any,
    max_iterations: int,
    score_threshold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        trace = run_protocol(
            problem, callers_agents, caller_meta, max_iterations, score_threshold
        )
    except Exception as exc:
        trace = {
            "schema_version": SCHEMA_VERSION,
            "root_dataset": problem.root_dataset,
            "language": problem.language,
            "problem_id": problem.id,
            "max_iterations": max_iterations,
            "score_threshold": score_threshold,
            "roles": [],
            "iterations": [],
            "final_completion": "",
            "winning_agent": None,
            "tie_break": False,
            "terminated_by": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": 0.0,
        }
    return completion_payload(problem, trace), trace


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


def build_callers(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    agent_generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature_agent,
        top_p=args.top_p,
        enable_thinking=False,
    )
    meta_generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature_meta,
        top_p=args.top_p,
        enable_thinking=False,
    )
    agents = {
        agent_id: OpenAIChatLLMCaller(
            getattr(args, f"api_base_{agent_id.lower()}"),
            getattr(args, f"api_model_{agent_id.lower()}"),
            generation=agent_generation,
            timeout=args.api_timeout,
            response_format={"type": "json_object"},
        )
        for agent_id in AGENT_IDS
    }
    meta = OpenAIChatLLMCaller(
        args.api_base_meta,
        args.api_model_meta,
        generation=meta_generation,
        timeout=args.api_timeout,
        response_format={"type": "json_object"},
    )
    return agents, meta


def main() -> int:
    args = parse_args()
    if args.max_iterations < 1 or args.max_new_tokens < 1 or args.max_concurrency < 1:
        raise SystemExit("iteration, token and concurrency limits must be positive")
    if not 0 <= args.score_threshold <= 10:
        raise SystemExit("--score-threshold must be in [0, 10]")
    if args.max_problems_per_language is not None and args.max_problems_per_language < 1:
        raise SystemExit("--max-problems-per-language must be positive")
    configure_core()
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No MultiPL-E tasks selected")
    print(
        f"AgentVerse tasks={len(tasks)} max_iterations={args.max_iterations} "
        f"score_threshold={args.score_threshold}"
    )
    print("models=A1:1.7B,A2:4B,A3:8B,Recruiter/Evaluator:8B")
    print("thinking_mode=disabled aggregation=per_iteration_normalized_exact_majority")
    if args.dry_run:
        first = tasks[0]
        print(f"dry-run first={first.root_dataset}/{first.language}/{first.id}")
        return 0

    callers_agents, caller_meta = build_callers(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.trajectory_output.open("w", encoding="utf-8") as trace_handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(tasks))) as pool:
            futures = {
                pool.submit(
                    run_one,
                    task,
                    callers_agents,
                    caller_meta,
                    args.max_iterations,
                    args.score_threshold,
                ): task
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                problem = futures[future]
                payload, trace = future.result()
                output_path = (
                    args.output_dir
                    / problem.root_dataset
                    / problem.language
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
                    },
                )
                print(
                    f"[{index}/{len(tasks)}] {problem.root_dataset}/{problem.language}/"
                    f"{problem.id} status={trace['terminated_by']} "
                    f"iterations={len(trace['iterations'])} winner={trace['winning_agent']}",
                    flush=True,
                )
                if trace.get("error"):
                    print(f"  error={trace['error']}", flush=True)
    print(f"AgentVerse generation finished in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
