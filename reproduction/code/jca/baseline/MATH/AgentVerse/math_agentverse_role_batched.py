#!/usr/bin/env python3
"""Durable role-batched AgentVerse evaluation on MATH."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jca.baseline.MATH.MAD import math_mad_role_batched as shared
from jca.src.math_eval import compute_math_em, format_math_problem_as_prompt, load_math_problems
from jca.src.relaxed_json import find_json_object


STATE_VERSION = 1
JOURNAL_VERSION = 1
AGENTS = ("A1", "A2", "A3")
CAPACITY_TO_AGENT = {"low": "A1", "mid": "A2", "high": "A3"}
JOURNAL_FIELDS = ("recruit", "roles", "agent_status", "iterations", "evaluation_status", "status", "exit_reason", "error", "wall_time_s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def journal_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.name}.journal.jsonl")


def replay_journal(state_path: Path, state: dict[str, Any]) -> int:
    path = journal_path(state_path)
    if not path.exists():
        return 0
    valid_bytes = 0
    applied = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                break
            record = json.loads(raw)
            if record.get("version") != JOURNAL_VERSION:
                raise ValueError("unsupported journal version")
            index = int(record["index"])
            item = state["items"][index]
            if item["problem_index"] != record["problem_index"]:
                raise ValueError(f"journal identity mismatch at item {index}")
            item.update(record["update"])
            valid_bytes = handle.tell()
            applied += 1
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return applied


class Journal:
    def __init__(self, state_path: Path, fsync_every: int) -> None:
        self.path = journal_path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.fsync_every = max(1, fsync_every)
        self.count = 0

    def append(self, index: int, item: dict[str, Any]) -> None:
        record = {
            "version": JOURNAL_VERSION, "index": index,
            "problem_index": item["problem_index"],
            "update": {field: item[field] for field in JOURNAL_FIELDS},
        }
        self.handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % self.fsync_every == 0:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    shared.atomic_json(path, state)
    journal_path(path).unlink(missing_ok=True)


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise ValueError("unsupported state version")
    replayed = replay_journal(path, state)
    if replayed:
        print(f"[resume] replayed {replayed} item updates", file=sys.stderr, flush=True)
    return state


def agent_key(iteration: int, agent: str) -> str:
    return f"i{iteration}_{agent}"


def evaluation_key(iteration: int) -> str:
    return f"i{iteration}"


def prompt_text(problem: dict[str, Any]) -> str:
    class ProblemView:
        prompt = problem["prompt"]
    return format_math_problem_as_prompt(ProblemView())


def select_problems(args: argparse.Namespace) -> list[Any]:
    problems = load_math_problems(args.data_root, split=args.split)
    if args.problem_ids_file is not None:
        requested_ids: list[str] = []
        try:
            with args.problem_ids_file.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not str(value.get("problem_id", "")).strip():
                        raise ValueError(f"missing problem_id at line {line_number}")
                    requested_ids.append(str(value["problem_id"]).strip())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid problem IDs file {args.problem_ids_file}: {exc}") from exc
        if not requested_ids:
            raise SystemExit(f"no problem IDs found in {args.problem_ids_file}")
        if len(requested_ids) != len(set(requested_ids)):
            raise SystemExit(f"duplicate problem IDs in {args.problem_ids_file}")
        problem_by_id = {problem.problem_id: problem for problem in problems}
        missing = [problem_id for problem_id in requested_ids if problem_id not in problem_by_id]
        if missing:
            raise SystemExit(f"unknown problem IDs in {args.problem_ids_file}: {missing[:5]}")
        if args.start != 0 or args.limit != len(requested_ids):
            raise SystemExit(
                "--problem-ids-file requires --start=0 and --limit equal to the file length"
            )
        return [problem_by_id[problem_id] for problem_id in requested_ids]
    selected = problems[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise SystemExit(f"requested {args.limit}, selected {len(selected)}")
    return selected


def stable_seed(problem_id: str, stage: str, retry_index: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{problem_id}:{stage}:{retry_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_recruiter(visible: str) -> tuple[list[dict[str, str]] | None, str | None]:
    if "{" not in visible:
        return None, "no JSON object"
    payload = find_json_object(visible)
    if payload is None:
        return None, "invalid JSON"
    roles = payload.get("roles")
    if not isinstance(roles, list) or len(roles) != 3:
        return None, "roles must contain exactly three entries"
    parsed = []
    capacities = []
    for role in roles:
        if not isinstance(role, dict):
            return None, "each role must be an object"
        name = role.get("name")
        capacity = role.get("capacity")
        description = role.get("description")
        if not isinstance(name, str) or not name.strip():
            return None, "role name is empty"
        if capacity not in CAPACITY_TO_AGENT:
            return None, f"invalid role capacity: {capacity}"
        if not isinstance(description, str) or not description.strip():
            return None, "role description is empty"
        capacities.append(capacity)
        parsed.append({
            "name": name.strip(), "capacity": capacity,
            "description": description.strip(), "agent": CAPACITY_TO_AGENT[capacity],
        })
    if set(capacities) != set(CAPACITY_TO_AGENT) or len(set(capacities)) != 3:
        return None, "roles must cover low, mid, and high exactly once"
    parsed.sort(key=lambda role: AGENTS.index(role["agent"]))
    return parsed, None


def parse_evaluator(visible: str) -> tuple[dict[str, Any] | None, str | None]:
    if "{" not in visible:
        return None, "no JSON object"
    payload = find_json_object(visible)
    if payload is None:
        return None, "invalid JSON"
    score = payload.get("score")
    feedback = payload.get("feedback")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 10:
        return None, "score must be a number from 0 to 10"
    if not isinstance(feedback, str) or not feedback.strip():
        return None, "feedback is empty"
    return {"score": float(score), "feedback": feedback.strip()}, None


def request_with_protocol(
    item: dict[str, Any], stage: str, messages: list[dict[str, str]], api_base: str,
    api_model: str, api_key: str, timeout: float, temperature: float,
    parser: Any, retry_label: str,
) -> tuple[list[dict[str, Any]], Any, str]:
    config = item["config"]
    attempts = []
    accepted = None
    failure_reason = "no request made"
    enable_thinking = bool(config.get("enable_thinking", False))
    require_thinking = bool(config.get("require_thinking", False))
    for retry_index in range(int(config["component_retries"]) + 1):
        attempt_messages = copy.deepcopy(messages)
        if retry_index:
            attempt_messages.append({
                "role": "user",
                "content": f"The previous {retry_label} response failed validation: {failure_reason}. Recompute and return exactly the required JSON object.",
            })
        seed = stable_seed(item["problem"]["problem_id"], stage, retry_index, int(config["generation_seed"]))
        attempt = shared.request_chat(
            api_base, api_model, api_key, attempt_messages,
            int(config["max_new_tokens"]), temperature, float(config["top_p"]), seed, timeout,
            enable_thinking,
        )
        attempt.update({"retry_index": retry_index, "messages": attempt_messages, "seed": seed})
        if not attempt["request_ok"]:
            failure_reason = attempt["exception"]
        elif not enable_thinking and attempt["thinking"].strip():
            failure_reason = "unexpected thinking output while thinking is disabled"
        elif require_thinking and not attempt["thinking"].strip():
            failure_reason = "thinking is empty"
        else:
            accepted, parse_error = parser(attempt["visible_output"])
            failure_reason = parse_error or ""
        attempt["parse_ok"] = accepted is not None
        attempt["validation_error"] = None if accepted is not None else failure_reason
        attempts.append(attempt)
        if accepted is not None:
            break
    return attempts, accepted, failure_reason


def run_recruit(
    item: dict[str, Any], prompt_dir: Path, api_base: str, api_model: str,
    api_key: str, timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    messages = [
        {"role": "system", "content": (prompt_dir / "recruiter.md").read_text(encoding="utf-8").strip()},
        {"role": "user", "content": prompt_text(item["problem"])},
    ]
    attempts, roles, failure = request_with_protocol(
        item, "recruit", messages, api_base, api_model, api_key, timeout,
        float(item["config"]["temperature_meta"]), parse_recruiter, "recruiter",
    )
    result = {
        "api_model": api_model, "model": item["config"]["model_paths"]["A3"],
        "messages": messages, "attempts": attempts, "success": roles is not None,
        "failure_reason": None if roles is not None else failure,
        "wall_time_s": round(time.monotonic() - started, 6),
    }
    item["recruit"] = result
    item["wall_time_s"] = round(item["wall_time_s"] + result["wall_time_s"], 6)
    if roles is None:
        item["status"] = "failed"
        item["exit_reason"] = "recruit_failed"
        item["error"] = failure
    else:
        item["roles"] = roles
        item["status"] = "active"
    return item


def role_for_agent(item: dict[str, Any], agent: str) -> dict[str, str]:
    for role in item["roles"]:
        if role["agent"] == agent:
            return role
    raise KeyError(agent)


def previous_iteration_block(item: dict[str, Any], iteration: int) -> str:
    previous = item["iterations"][str(iteration - 1)]
    blocks = []
    for agent in AGENTS:
        answer = previous["answers"].get(agent)
        if answer and answer["success"]:
            body = f"Reasoning: {answer['reasoning']}\nAnswer: {answer['answer']}"
        else:
            body = f"STATUS: FAILED\nReason: {(answer or {}).get('failure_reason', 'missing output')}"
        blocks.append(f"## {agent}\n{body}")
    evaluation = previous.get("evaluation") or {}
    feedback = evaluation.get("feedback") if evaluation.get("success") else "Evaluator output unavailable; independently recheck all mathematics."
    return (
        "# Previous Agent Attempts\n" + "\n\n".join(blocks)
        + f"\n\n# Previous Team Answer\n{previous.get('team_answer') or '<empty>'}"
        + f"\n\n# Evaluator Feedback\n{feedback}"
    )


def build_agent_messages(
    item: dict[str, Any], iteration: int, agent: str, prompt_dir: Path,
) -> list[dict[str, str]]:
    role = role_for_agent(item, agent)
    template = (prompt_dir / "agent.md").read_text(encoding="utf-8").strip()
    system = template.format(
        ROLE_NAME=role["name"], ROLE_CAPACITY=role["capacity"],
        ROLE_DESCRIPTION=role["description"], AGENT_ID=agent,
    )
    user = prompt_text(item["problem"])
    if iteration:
        user += f"\n\n{previous_iteration_block(item, iteration)}\n\nProduce your revised JSON answer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def aggregate_iteration(item: dict[str, Any], iteration: int) -> dict[str, Any]:
    answers = {
        agent: item["iterations"][str(iteration)]["answers"].get(agent, {}).get("answer", "")
        for agent in AGENTS
    }
    proxy = {
        "config": {"n_rounds": 1},
        "turns": {shared.turn_key(0, agent): {"answer": answer} for agent, answer in answers.items()},
    }
    vote = shared.aggregate_answers(proxy)
    return {
        "raw_answers": vote["raw_answers"], "normalized_answers": vote["normalized_answers"],
        "buckets": vote["buckets"], "winning_normalized_answer": vote["winning_normalized_answer"],
        "winning_agent": vote["winning_agent"], "team_answer": vote["final_answer"],
        "vote_type": vote["vote_type"], "tie_break": vote["tie_break"],
    }


def run_agent(
    item: dict[str, Any], iteration: int, agent: str, prompt_dir: Path,
    api_base: str, api_model: str, api_key: str, timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    messages = build_agent_messages(item, iteration, agent, prompt_dir)
    attempts, accepted, failure = request_with_protocol(
        item, f"iteration{iteration}_{agent}", messages, api_base, api_model, api_key, timeout,
        float(item["config"]["temperature_agent"]), shared.parse_visible_json, "solver",
    )
    role = role_for_agent(item, agent)
    result = {
        "iteration": iteration, "agent": agent, "role": role,
        "api_model": api_model, "model": item["config"]["model_paths"][agent],
        "messages": messages, "attempts": attempts, "success": accepted is not None,
        "reasoning": accepted["reasoning"] if accepted else "",
        "answer": accepted["answer"] if accepted else "",
        "failure_reason": None if accepted else failure,
        "wall_time_s": round(time.monotonic() - started, 6),
    }
    iteration_state = item["iterations"].setdefault(str(iteration), {"answers": {}})
    iteration_state["answers"][agent] = result
    item["agent_status"][agent_key(iteration, agent)] = "success" if accepted else "failed"
    item["wall_time_s"] = round(item["wall_time_s"] + result["wall_time_s"], 6)
    if all(item["agent_status"][agent_key(iteration, candidate)] != "pending" for candidate in AGENTS):
        vote = aggregate_iteration(item, iteration)
        iteration_state.update(vote)
    return item


def build_evaluator_messages(
    item: dict[str, Any], iteration: int, prompt_dir: Path,
) -> list[dict[str, str]]:
    state = item["iterations"][str(iteration)]
    blocks = []
    for agent in AGENTS:
        answer = state["answers"].get(agent)
        if answer and answer["success"]:
            body = f"Role: {answer['role']['name']} ({answer['role']['capacity']})\nReasoning: {answer['reasoning']}\nAnswer: {answer['answer']}"
        else:
            body = f"STATUS: FAILED\nReason: {(answer or {}).get('failure_reason', 'missing output')}"
        blocks.append(f"## {agent}\n{body}")
    user = (
        f"{prompt_text(item['problem'])}\n\n# Current Agent Attempts\n" + "\n\n".join(blocks)
        + f"\n\n# Deterministic Team Answer\n{state.get('team_answer') or '<empty>'}"
        + "\n\nEvaluate this iteration without access to any reference answer."
    )
    return [
        {"role": "system", "content": (prompt_dir / "evaluator.md").read_text(encoding="utf-8").strip()},
        {"role": "user", "content": user},
    ]


def run_evaluator(
    item: dict[str, Any], iteration: int, prompt_dir: Path,
    api_base: str, api_model: str, api_key: str, timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    messages = build_evaluator_messages(item, iteration, prompt_dir)
    attempts, accepted, failure = request_with_protocol(
        item, f"iteration{iteration}_evaluate", messages, api_base, api_model, api_key, timeout,
        float(item["config"]["temperature_meta"]), parse_evaluator, "evaluator",
    )
    result = {
        "iteration": iteration, "api_model": api_model,
        "model": item["config"]["model_paths"]["A3"], "messages": messages,
        "attempts": attempts, "success": accepted is not None,
        "score": accepted["score"] if accepted else None,
        "feedback": accepted["feedback"] if accepted else "",
        "failure_reason": None if accepted else failure,
        "wall_time_s": round(time.monotonic() - started, 6),
    }
    item["iterations"][str(iteration)]["evaluation"] = result
    item["evaluation_status"][evaluation_key(iteration)] = "success" if accepted else "failed"
    item["wall_time_s"] = round(item["wall_time_s"] + result["wall_time_s"], 6)
    threshold = float(item["config"]["score_threshold"])
    last_iteration = int(item["config"]["max_iterations"]) - 1
    if accepted is not None and accepted["score"] >= threshold:
        item["status"] = "done"
        item["exit_reason"] = "score_threshold"
    elif iteration == last_iteration:
        item["status"] = "done"
        item["exit_reason"] = "max_iterations"
    return item


def recruit_pending(state: dict[str, Any]) -> list[int]:
    return [index for index, item in enumerate(state["items"]) if item["recruit"] is None and item["status"] == "pending_recruit"]


def iteration_ready(item: dict[str, Any], iteration: int) -> bool:
    if item["status"] != "active":
        return False
    if iteration == 0:
        return item["recruit"] is not None and item["recruit"]["success"]
    return item["evaluation_status"][evaluation_key(iteration - 1)] in {"success", "failed"}


def agent_pending(state: dict[str, Any], iteration: int, agent: str) -> list[int]:
    key = agent_key(iteration, agent)
    return [index for index, item in enumerate(state["items"]) if item["agent_status"][key] == "pending" and iteration_ready(item, iteration)]


def evaluator_pending(state: dict[str, Any], iteration: int) -> list[int]:
    key = evaluation_key(iteration)
    return [
        index for index, item in enumerate(state["items"])
        if item["evaluation_status"][key] == "pending" and item["status"] == "active"
        and all(item["agent_status"][agent_key(iteration, agent)] != "pending" for agent in AGENTS)
        and "team_answer" in item["iterations"].get(str(iteration), {})
    ]


def total_usage(item: dict[str, Any]) -> dict[str, int | None]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = False
    components = []
    if item["recruit"]:
        components.append(item["recruit"])
    for iteration in item["iterations"].values():
        components.extend(iteration.get("answers", {}).values())
        if iteration.get("evaluation"):
            components.append(iteration["evaluation"])
    for component in components:
        for attempt in component["attempts"]:
            usage = attempt.get("usage")
            if not isinstance(usage, dict):
                continue
            seen = True
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals if seen else {key: None for key in totals}


def finalized_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in state["items"]:
        if item["status"] not in {"done", "failed"}:
            raise ValueError(f"unfinished problem: {item['problem']['problem_id']}")
        completed_iterations = sorted(int(key) for key in item["iterations"] if "team_answer" in item["iterations"][key])
        final_iteration = completed_iterations[-1] if completed_iterations else None
        final_answer = item["iterations"][str(final_iteration)].get("team_answer", "") if final_iteration is not None else ""
        problem = item["problem"]
        components = ([item["recruit"]] if item["recruit"] else [])
        for iteration in item["iterations"].values():
            components.extend(iteration.get("answers", {}).values())
            if iteration.get("evaluation"):
                components.append(iteration["evaluation"])
        rows.append({
            "schema_version": 1, "baseline": "agentverse", "training_free": True,
            "problem_index": item["problem_index"], "problem_id": problem["problem_id"],
            "subject": problem["subject"], "level": problem["level"], "prompt": problem["prompt"],
            "models": state["config"]["model_paths"], "roles": item["roles"],
            "recruit": item["recruit"], "iterations": item["iterations"],
            "final_iteration": final_iteration, "final_answer": final_answer,
            "gold_answer": problem["gold_answer"], "em": compute_math_em(final_answer, problem["gold_answer"]),
            "terminated_by": item["exit_reason"], "error": item["error"],
            "total_calls": sum(len(component["attempts"]) for component in components),
            "token_usage": total_usage(item), "wall_time_s": item["wall_time_s"],
            "sampling": state["config"],
        })
    return rows


def build_summary(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    ids = [row["problem_id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError(f"coverage failure: rows={len(rows)} unique={len(set(ids))} expected={expected}")
    subjects = {}
    for subject in sorted({row["subject"] for row in rows}):
        selected = [row for row in rows if row["subject"] == subject]
        correct = sum(row["em"] for row in selected)
        subjects[subject] = {"count": len(selected), "correct": int(correct), "em": correct / len(selected)}
    votes = Counter()
    tie_breaks = 0
    solver_failures = Counter()
    evaluator_failures = Counter()
    retry_calls = Counter()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        components: list[tuple[str, dict[str, Any]]] = []
        if row["recruit"]:
            components.append(("recruit", row["recruit"]))
        for iteration_key, iteration in row["iterations"].items():
            if "vote_type" in iteration:
                votes[iteration["vote_type"]] += 1
                tie_breaks += int(iteration["tie_break"])
            for agent, answer in iteration.get("answers", {}).items():
                components.append((f"iteration{iteration_key}_{agent}", answer))
                if not answer["success"]:
                    solver_failures[f"i{iteration_key}_{agent}"] += 1
            evaluation = iteration.get("evaluation")
            if evaluation:
                components.append((f"iteration{iteration_key}_evaluate", evaluation))
                if not evaluation["success"]:
                    evaluator_failures[f"i{iteration_key}"] += 1
        for stage, component in components:
            bucket = grouped.setdefault(stage, {
                "model": component["model"], "components": 0, "successes": 0,
                "calls": 0, "wall_time_s": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
            })
            bucket["components"] += 1
            bucket["successes"] += int(component["success"])
            bucket["calls"] += len(component["attempts"])
            bucket["wall_time_s"] = round(bucket["wall_time_s"] + component["wall_time_s"], 6)
            for attempt in component["attempts"]:
                retry_calls[int(attempt["retry_index"])] += 1
                usage = attempt.get("usage") or {}
                for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    bucket[token_key] += int(usage.get(token_key) or 0)
    correct = sum(row["em"] for row in rows)
    completed_iterations = [row["final_iteration"] + 1 for row in rows if row["final_iteration"] is not None]
    return {
        "expected": expected, "count": len(rows), "unique_problem_ids": len(set(ids)),
        "correct": int(correct), "em": correct / len(rows), "subjects": subjects,
        "exit_reasons": dict(Counter(row["terminated_by"] for row in rows)),
        "recruit_failures": sum(not row["recruit"] or not row["recruit"]["success"] for row in rows),
        "solver_failures": dict(solver_failures), "evaluator_failures": dict(evaluator_failures),
        "vote_types": dict(votes), "tie_breaks": tie_breaks,
        "average_iterations": sum(completed_iterations) / len(rows),
        "calls_by_retry_index": dict(sorted(retry_calls.items())),
        "total_calls": sum(row["total_calls"] for row in rows),
        "token_usage": {
            key: sum((row["token_usage"].get(key) or 0) for row in rows)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "by_stage_model": grouped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--data-root", type=Path, required=True)
    init.add_argument("--split", choices=("train", "test"), default="test")
    init.add_argument("--problem-ids-file", type=Path)
    init.add_argument("--start", type=int, default=0)
    init.add_argument("--limit", type=int, default=5000)
    init.add_argument("--max-iterations", type=int, default=3)
    init.add_argument("--score-threshold", type=float, default=8)
    init.add_argument("--max-new-tokens", type=int, default=8192)
    init.add_argument("--temperature-agent", type=float, default=0.7)
    init.add_argument("--temperature-meta", type=float, default=0.0)
    init.add_argument("--top-p", type=float, default=0.95)
    init.add_argument("--generation-seed", type=int, default=42)
    init.add_argument("--component-retries", type=int, default=4)
    init.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--model-a1", required=True)
    init.add_argument("--model-a2", required=True)
    init.add_argument("--model-a3", required=True)
    status = commands.add_parser("status")
    status.add_argument("--state", type=Path, required=True)
    pending = commands.add_parser("pending")
    pending.add_argument("--state", type=Path, required=True)
    pending.add_argument("--stage", choices=("recruit", "agent", "evaluate"), required=True)
    pending.add_argument("--iteration", type=int, choices=(0, 1, 2))
    pending.add_argument("--agent", choices=AGENTS)
    recruit = commands.add_parser("run-recruit")
    agent = commands.add_parser("run-agents")
    evaluate = commands.add_parser("run-evaluator")
    for sub in (recruit, agent, evaluate):
        sub.add_argument("--state", type=Path, required=True)
        sub.add_argument("--prompt-dir", type=Path, required=True)
        sub.add_argument("--api-base", required=True)
        sub.add_argument("--api-model", required=True)
        sub.add_argument("--api-key", default="EMPTY")
        sub.add_argument("--api-timeout", type=float, default=900)
        sub.add_argument("--max-concurrency", type=int, default=128)
        sub.add_argument("--journal-fsync-every", type=int, default=25)
    agent.add_argument("--iteration", type=int, choices=(0, 1, 2), required=True)
    agent.add_argument("--agent", choices=AGENTS, required=True)
    evaluate.add_argument("--iteration", type=int, choices=(0, 1, 2), required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--state", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--summary", type=Path, required=True)
    final.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_phase(
    state_path: Path, state: dict[str, Any], indices: list[int], operation: Any,
    max_concurrency: int, fsync_every: int, label: str,
) -> None:
    started = time.monotonic()
    with Journal(state_path, fsync_every) as journal:
        with ThreadPoolExecutor(max_workers=max(1, min(max_concurrency, len(indices) or 1))) as pool:
            futures = [pool.submit(operation, index) for index in indices]
            for completed, future in enumerate(as_completed(futures), 1):
                index, item = future.result()
                state["items"][index] = item
                journal.append(index, item)
                if completed == 1 or completed % 250 == 0 or completed == len(indices):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(f"progress={completed}/{len(indices)} stage={label} rate={completed / elapsed:.2f}/s", flush=True)
    save_state(state_path, state)


def main() -> None:
    args = parse_args()
    if args.command == "init":
        if args.state.exists():
            raise SystemExit(f"state exists: {args.state}")
        if args.max_iterations != 3:
            raise SystemExit("MATH AgentVerse requires exactly 3 iterations")
        if args.require_thinking and not args.enable_thinking:
            raise SystemExit("--require-thinking requires --enable-thinking")
        selected = select_problems(args)
        config = {
            "split": args.split, "start": args.start, "limit": args.limit,
            "problem_ids_file": str(args.problem_ids_file.resolve()) if args.problem_ids_file else None,
            "max_iterations": args.max_iterations, "score_threshold": args.score_threshold,
            "max_new_tokens": args.max_new_tokens, "max_model_len": 40960,
            "temperature_agent": args.temperature_agent, "temperature_meta": args.temperature_meta,
            "top_p": args.top_p, "generation_seed": args.generation_seed,
            "component_retries": args.component_retries,
            "enable_thinking": args.enable_thinking,
            "require_thinking": args.require_thinking,
            "model_paths": {"A1": args.model_a1, "A2": args.model_a2, "A3": args.model_a3},
            "tie_break_priority": ["A3", "A2", "A1"],
        }
        agent_status = {agent_key(iteration, agent): "pending" for iteration in range(3) for agent in AGENTS}
        evaluation_status = {evaluation_key(iteration): "pending" for iteration in range(3)}
        items = []
        for offset, problem in enumerate(selected):
            items.append({
                "problem_index": args.start + offset,
                "problem": {
                    "problem_id": problem.problem_id, "subject": problem.subject,
                    "level": problem.level, "prompt": problem.prompt,
                    "solution": problem.solution, "gold_answer": problem.gold_answer,
                },
                "config": config, "recruit": None, "roles": [],
                "agent_status": dict(agent_status), "iterations": {},
                "evaluation_status": dict(evaluation_status), "status": "pending_recruit",
                "exit_reason": None, "error": None, "wall_time_s": 0.0,
            })
        save_state(args.state, {
            "version": STATE_VERSION, "created_at": utc_now(), "updated_at": utc_now(),
            "config": config, "items": items,
        })
        print(f"initialized={len(items)} state={args.state}")
        return
    state = load_state(args.state)
    if args.command == "pending":
        if args.stage == "recruit":
            indices = recruit_pending(state)
        elif args.stage == "agent":
            if args.iteration is None or args.agent is None:
                raise SystemExit("agent pending requires --iteration and --agent")
            indices = agent_pending(state, args.iteration, args.agent)
        else:
            if args.iteration is None:
                raise SystemExit("evaluate pending requires --iteration")
            indices = evaluator_pending(state, args.iteration)
        print(len(indices))
        return
    if args.command == "status":
        print(json.dumps({
            "items": len(state["items"]), "pending_recruit": len(recruit_pending(state)),
            "active": sum(item["status"] == "active" for item in state["items"]),
            "done": sum(item["status"] == "done" for item in state["items"]),
            "failed": sum(item["status"] == "failed" for item in state["items"]),
            "pending_agents": {
                agent_key(iteration, agent): len(agent_pending(state, iteration, agent))
                for iteration in range(3) for agent in AGENTS
            },
            "pending_evaluators": {evaluation_key(iteration): len(evaluator_pending(state, iteration)) for iteration in range(3)},
        }, sort_keys=True))
        return
    if args.command == "run-recruit":
        common = (args.prompt_dir, args.api_base, args.api_model, args.api_key, args.api_timeout)
        if not args.api_model.startswith("A3"):
            raise SystemExit("Recruiter requires A3 model")
        indices = recruit_pending(state)
        def operation(index: int) -> tuple[int, dict[str, Any]]:
            return index, run_recruit(copy.deepcopy(state["items"][index]), *common)
        run_phase(args.state, state, indices, operation, args.max_concurrency, args.journal_fsync_every, "recruit")
        return
    if args.command == "run-agents":
        common = (args.prompt_dir, args.api_base, args.api_model, args.api_key, args.api_timeout)
        if not args.api_model.startswith(args.agent):
            raise SystemExit(f"agent {args.agent} got wrong model {args.api_model}")
        indices = agent_pending(state, args.iteration, args.agent)
        def operation(index: int) -> tuple[int, dict[str, Any]]:
            return index, run_agent(copy.deepcopy(state["items"][index]), args.iteration, args.agent, *common)
        run_phase(args.state, state, indices, operation, args.max_concurrency, args.journal_fsync_every, f"i{args.iteration}_{args.agent}")
        return
    if args.command == "run-evaluator":
        common = (args.prompt_dir, args.api_base, args.api_model, args.api_key, args.api_timeout)
        if not args.api_model.startswith("A3"):
            raise SystemExit("Evaluator requires A3 model")
        indices = evaluator_pending(state, args.iteration)
        def operation(index: int) -> tuple[int, dict[str, Any]]:
            return index, run_evaluator(copy.deepcopy(state["items"][index]), args.iteration, *common)
        run_phase(args.state, state, indices, operation, args.max_concurrency, args.journal_fsync_every, f"i{args.iteration}_evaluate")
        return
    if args.command == "finalize":
        rows = finalized_rows(state)
        summary = build_summary(rows, int(state["config"]["limit"]))
        if args.output.exists() and not args.resume:
            raise SystemExit(f"output exists: {args.output}")
        shared.atomic_jsonl(args.output, rows)
        shared.atomic_json(args.summary, summary)
        print(json.dumps(summary, sort_keys=True))
        return


if __name__ == "__main__":
    main()
