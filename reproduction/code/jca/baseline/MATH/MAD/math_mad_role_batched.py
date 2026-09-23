#!/usr/bin/env python3
"""Durable role-batched three-round MAD evaluation on MATH."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jca.src.math_eval import (
    compute_math_em,
    format_math_problem_as_prompt,
    load_math_problems,
    normalize_math_answer,
)
from jca.src.relaxed_json import find_json_object


STATE_VERSION = 1
JOURNAL_VERSION = 1
AGENTS = ("A1", "A2", "A3")
TIE_BREAK_PRIORITY = ("A3", "A2", "A1")
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)
JOURNAL_FIELDS = ("turn_status", "turns", "status", "error", "wall_time_s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
            "version": JOURNAL_VERSION,
            "index": index,
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
    atomic_json(path, state)
    journal_path(path).unlink(missing_ok=True)


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise ValueError("unsupported state version")
    replayed = replay_journal(path, state)
    if replayed:
        print(f"[resume] replayed {replayed} turn updates", file=sys.stderr, flush=True)
    return state


def turn_key(round_idx: int, agent: str) -> str:
    return f"r{round_idx}_{agent}"


def split_thinking(raw: str) -> tuple[str, str]:
    matches = list(THINK_RE.finditer(raw))
    thinking = "\n\n".join(match.group(1).strip() for match in matches if match.group(1).strip())
    return thinking, THINK_RE.sub("", raw).strip()


def parse_visible_json(visible: str) -> tuple[dict[str, str] | None, str | None]:
    if "{" not in visible:
        return None, "no JSON object"
    payload = find_json_object(visible)
    if payload is None:
        return None, "invalid JSON"
    reasoning = payload.get("reasoning")
    answer = payload.get("answer")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "reasoning is empty"
    if not isinstance(answer, str) or not answer.strip():
        return None, "answer is empty"
    return {"reasoning": reasoning.strip(), "answer": answer.strip()}, None


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


def stable_peer_order(problem_id: str, round_idx: int, agent: str, peers: list[str], seed: int) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{problem_id}:{round_idx}:{agent}:peers".encode()).digest()
    shuffled = list(peers)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(shuffled)
    return shuffled


def request_seed(problem_id: str, round_idx: int, agent: str, retry_index: int, seed: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{problem_id}:{round_idx}:{agent}:{retry_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def previous_turn_block(item: dict[str, Any], round_idx: int, agent: str) -> tuple[str, dict[str, str]]:
    own_key = turn_key(round_idx - 1, agent)
    own = item["turns"].get(own_key)
    if own and own["success"]:
        own_text = f"Reasoning: {own['reasoning']}\nAnswer: {own['answer']}"
    else:
        own_text = f"STATUS: FAILED\nReason: {(own or {}).get('failure_reason', 'missing output')}"
    peers = [candidate for candidate in AGENTS if candidate != agent]
    ordered = stable_peer_order(
        item["problem"]["problem_id"], round_idx, agent, peers,
        int(item["config"]["generation_seed"]),
    )
    mapping = {label: peer for label, peer in zip(("Peer A", "Peer B"), ordered)}
    blocks = []
    for label, peer in mapping.items():
        turn = item["turns"].get(turn_key(round_idx - 1, peer))
        if turn and turn["success"]:
            body = f"Reasoning: {turn['reasoning']}\nAnswer: {turn['answer']}"
        else:
            body = f"STATUS: FAILED\nReason: {(turn or {}).get('failure_reason', 'missing output')}"
        blocks.append(f"## {label}\n{body}")
    return (
        f"# Your Previous Answer\n{own_text}\n\n# Other Agents' Answers\n" + "\n\n".join(blocks),
        mapping,
    )


def build_messages(
    item: dict[str, Any], round_idx: int, agent: str, prompt_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if round_idx == 0:
        system = (prompt_dir / "round0.md").read_text(encoding="utf-8").strip()
        user = prompt_text(item["problem"])
        peer_mapping: dict[str, str] = {}
    else:
        system = (prompt_dir / "debate.md").read_text(encoding="utf-8").strip()
        previous, peer_mapping = previous_turn_block(item, round_idx, agent)
        user = f"{prompt_text(item['problem'])}\n\n{previous}\n\nProduce your updated JSON answer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], peer_mapping


def request_chat(
    api_base: str, api_model: str, api_key: str, messages: list[dict[str, str]],
    max_tokens: int, temperature: float, top_p: float, seed: int, timeout: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": api_model, "messages": messages, "temperature": temperature,
        "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_http = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw_http)
        choice = parsed["choices"][0]
        message = choice["message"]
        visible = str(message.get("content") or "").strip()
        thinking = str(message.get("reasoning_content") or "").strip()
        raw_output = f"<think>\n{thinking}\n</think>\n{visible}" if thinking else visible
        if not thinking:
            thinking, visible = split_thinking(visible)
        return {
            "request_ok": True, "raw_http_response": raw_http, "raw_output": raw_output,
            "thinking": thinking, "visible_output": visible,
            "finish_reason": choice.get("finish_reason"), "usage": parsed.get("usage"),
            "wall_time_s": round(time.monotonic() - started, 6), "exception": None,
        }
    except Exception as exc:
        body = exc.read().decode("utf-8", errors="replace") if isinstance(exc, urllib.error.HTTPError) else None
        return {
            "request_ok": False, "raw_http_response": body, "raw_output": "",
            "thinking": "", "visible_output": "", "finish_reason": None, "usage": None,
            "wall_time_s": round(time.monotonic() - started, 6),
            "exception": f"{type(exc).__name__}: {exc}",
        }


def run_turn(
    item: dict[str, Any], round_idx: int, agent: str, prompt_dir: Path,
    api_base: str, api_model: str, api_key: str, timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    messages, peer_mapping = build_messages(item, round_idx, agent, prompt_dir)
    config = item["config"]
    temperature = config["temperature_round0"] if round_idx == 0 else config["temperature_debate"]
    attempts = []
    accepted = None
    failure_reason = "no request made"
    enable_thinking = bool(config.get("enable_thinking", False))
    require_thinking = bool(config.get("require_thinking", False))
    for retry_index in range(int(config["turn_retries"]) + 1):
        attempt_messages = copy.deepcopy(messages)
        if retry_index:
            attempt_messages.append({
                "role": "user",
                "content": (
                    f"The previous response failed validation: {failure_reason}. Recompute and return "
                    'exactly one JSON object: {"reasoning":"...","answer":"..."}. Both fields '
                    "must be non-empty."
                ),
            })
        seed = request_seed(item["problem"]["problem_id"], round_idx, agent, retry_index, int(config["generation_seed"]))
        attempt = request_chat(
            api_base, api_model, api_key, attempt_messages, int(config["max_new_tokens"]),
            float(temperature), float(config["top_p"]), seed, timeout,
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
            accepted, parse_error = parse_visible_json(attempt["visible_output"])
            failure_reason = parse_error or ""
        attempt["parse_ok"] = accepted is not None
        attempt["validation_error"] = None if accepted is not None else failure_reason
        attempts.append(attempt)
        if accepted is not None:
            break
    result = {
        "round_idx": round_idx, "agent": agent, "api_model": api_model,
        "model": config["model_paths"][agent], "messages": messages,
        "peer_mapping": peer_mapping, "attempts": attempts,
        "success": accepted is not None,
        "reasoning": accepted["reasoning"] if accepted else "",
        "answer": accepted["answer"] if accepted else "",
        "failure_reason": None if accepted else failure_reason,
        "wall_time_s": round(time.monotonic() - started, 6),
    }
    key = turn_key(round_idx, agent)
    item["turns"][key] = result
    item["turn_status"][key] = "success" if accepted else "failed"
    item["wall_time_s"] = round(float(item["wall_time_s"]) + result["wall_time_s"], 6)
    if round_idx == int(config["n_rounds"]) - 1 and all(
        item["turn_status"][turn_key(round_idx, candidate)] != "pending" for candidate in AGENTS
    ):
        item["status"] = "ready_to_finalize"
    return item


def prior_round_complete(item: dict[str, Any], round_idx: int) -> bool:
    return round_idx == 0 or all(
        item["turn_status"][turn_key(round_idx - 1, agent)] != "pending" for agent in AGENTS
    )


def pending_indices(state: dict[str, Any], round_idx: int, agent: str) -> list[int]:
    key = turn_key(round_idx, agent)
    return [
        index for index, item in enumerate(state["items"])
        if item["turn_status"][key] == "pending" and prior_round_complete(item, round_idx)
    ]


def aggregate_answers(item: dict[str, Any]) -> dict[str, Any]:
    final_round = int(item["config"]["n_rounds"]) - 1
    raw_answers = {
        agent: item["turns"].get(turn_key(final_round, agent), {}).get("answer", "")
        for agent in AGENTS
    }
    normalized = {
        agent: normalize_math_answer(answer)
        for agent, answer in raw_answers.items() if answer and normalize_math_answer(answer)
    }
    buckets: dict[str, list[str]] = {}
    for agent, answer in normalized.items():
        buckets.setdefault(answer, []).append(agent)
    if not buckets:
        return {
            "raw_answers": raw_answers, "normalized_answers": normalized, "buckets": {},
            "winning_normalized_answer": None, "winning_agent": None,
            "final_answer": "", "vote_type": "all_empty", "tie_break": False,
        }
    max_votes = max(len(agents) for agents in buckets.values())
    winning_buckets = [answer for answer, agents in buckets.items() if len(agents) == max_votes]
    tie_break = len(winning_buckets) > 1
    winning_answer = None
    winning_agent = None
    for preferred in TIE_BREAK_PRIORITY:
        for candidate in winning_buckets:
            if preferred in buckets[candidate]:
                winning_answer, winning_agent = candidate, preferred
                break
        if winning_answer is not None:
            break
    assert winning_answer is not None and winning_agent is not None
    if max_votes == 3:
        vote_type = "unanimous"
    elif max_votes == 2:
        vote_type = "majority_2_1"
    else:
        vote_type = "three_way_tie" if len(normalized) == 3 else "partial_tie"
    return {
        "raw_answers": raw_answers, "normalized_answers": normalized, "buckets": buckets,
        "winning_normalized_answer": winning_answer, "winning_agent": winning_agent,
        "final_answer": raw_answers[winning_agent], "vote_type": vote_type, "tie_break": tie_break,
    }


def total_usage(item: dict[str, Any]) -> dict[str, int | None]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = False
    for turn in item["turns"].values():
        for attempt in turn["attempts"]:
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
        if any(status == "pending" for status in item["turn_status"].values()):
            raise ValueError(f"unfinished problem: {item['problem']['problem_id']}")
        aggregation = aggregate_answers(item)
        final_answer = aggregation["final_answer"]
        problem = item["problem"]
        status = "done" if final_answer else "failed"
        rows.append({
            "schema_version": 1, "baseline": "mad", "training_free": True,
            "problem_index": item["problem_index"], "problem_id": problem["problem_id"],
            "subject": problem["subject"], "level": problem["level"], "prompt": problem["prompt"],
            "models": state["config"]["model_paths"], "turns": item["turns"],
            "aggregation": aggregation, "final_answer": final_answer,
            "gold_answer": problem["gold_answer"], "em": compute_math_em(final_answer, problem["gold_answer"]),
            "terminated_by": status, "error": None if final_answer else "no non-empty final-round answer",
            "total_calls": sum(len(turn["attempts"]) for turn in item["turns"].values()),
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
    grouped: dict[str, dict[str, Any]] = {}
    retry_calls: Counter[int] = Counter()
    turn_failures: Counter[str] = Counter()
    for row in rows:
        for key, turn in row["turns"].items():
            group = grouped.setdefault(key, {
                "round_idx": turn["round_idx"], "agent": turn["agent"], "model": turn["model"],
                "turns": 0, "successes": 0, "calls": 0, "wall_time_s": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            })
            group["turns"] += 1
            group["successes"] += int(turn["success"])
            group["calls"] += len(turn["attempts"])
            group["wall_time_s"] = round(group["wall_time_s"] + turn["wall_time_s"], 6)
            if not turn["success"]:
                turn_failures[key] += 1
            for attempt in turn["attempts"]:
                retry_calls[int(attempt["retry_index"])] += 1
                usage = attempt.get("usage") or {}
                for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    group[token_key] += int(usage.get(token_key) or 0)
    correct = sum(row["em"] for row in rows)
    total_usage = {
        key: sum((row["token_usage"].get(key) or 0) for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "expected": expected, "count": len(rows), "unique_problem_ids": len(set(ids)),
        "correct": int(correct), "em": correct / len(rows), "subjects": subjects,
        "terminal_status": dict(Counter(row["terminated_by"] for row in rows)),
        "vote_types": dict(Counter(row["aggregation"]["vote_type"] for row in rows)),
        "tie_breaks": sum(row["aggregation"]["tie_break"] for row in rows),
        "turn_failures": dict(turn_failures), "calls_by_retry_index": dict(sorted(retry_calls.items())),
        "total_calls": sum(row["total_calls"] for row in rows), "token_usage": total_usage,
        "by_round_agent_model": grouped,
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
    init.add_argument("--n-rounds", type=int, default=3)
    init.add_argument("--max-new-tokens", type=int, default=8192)
    init.add_argument("--temperature-round0", type=float, default=0.9)
    init.add_argument("--temperature-debate", type=float, default=0.3)
    init.add_argument("--top-p", type=float, default=0.95)
    init.add_argument("--generation-seed", type=int, default=42)
    init.add_argument("--turn-retries", type=int, default=4)
    init.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--model-a1", required=True)
    init.add_argument("--model-a2", required=True)
    init.add_argument("--model-a3", required=True)
    for command in ("pending", "status"):
        sub = commands.add_parser(command)
        sub.add_argument("--state", type=Path, required=True)
        if command == "pending":
            sub.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
            sub.add_argument("--agent", choices=AGENTS, required=True)
    run = commands.add_parser("run-turns")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--agent", choices=AGENTS, required=True)
    run.add_argument("--prompt-dir", type=Path, required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--api-model", required=True)
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--api-timeout", type=float, default=900)
    run.add_argument("--max-concurrency", type=int, default=128)
    run.add_argument("--journal-fsync-every", type=int, default=25)
    final = commands.add_parser("finalize")
    final.add_argument("--state", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--summary", type=Path, required=True)
    final.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        if args.state.exists():
            raise SystemExit(f"state exists: {args.state}")
        if args.n_rounds != 3:
            raise SystemExit("MATH MAD requires exactly 3 rounds")
        if args.require_thinking and not args.enable_thinking:
            raise SystemExit("--require-thinking requires --enable-thinking")
        selected = select_problems(args)
        config = {
            "split": args.split, "start": args.start, "limit": args.limit,
            "problem_ids_file": str(args.problem_ids_file.resolve()) if args.problem_ids_file else None,
            "n_rounds": args.n_rounds, "max_new_tokens": args.max_new_tokens,
            "max_model_len": 40960, "temperature_round0": args.temperature_round0,
            "temperature_debate": args.temperature_debate, "top_p": args.top_p,
            "generation_seed": args.generation_seed, "turn_retries": args.turn_retries,
            "enable_thinking": args.enable_thinking,
            "require_thinking": args.require_thinking,
            "model_paths": {"A1": args.model_a1, "A2": args.model_a2, "A3": args.model_a3},
            "tie_break_priority": list(TIE_BREAK_PRIORITY),
        }
        statuses = {turn_key(round_idx, agent): "pending" for round_idx in range(3) for agent in AGENTS}
        items = []
        for offset, problem in enumerate(selected):
            items.append({
                "problem_index": args.start + offset,
                "problem": {
                    "problem_id": problem.problem_id, "subject": problem.subject,
                    "level": problem.level, "prompt": problem.prompt,
                    "solution": problem.solution, "gold_answer": problem.gold_answer,
                },
                "config": config, "turn_status": dict(statuses), "turns": {},
                "status": "pending", "error": None, "wall_time_s": 0.0,
            })
        save_state(args.state, {
            "version": STATE_VERSION, "created_at": utc_now(), "updated_at": utc_now(),
            "config": config, "items": items,
        })
        print(f"initialized={len(items)} state={args.state}")
        return
    state = load_state(args.state)
    if args.command == "pending":
        print(len(pending_indices(state, args.round, args.agent)))
        return
    if args.command == "status":
        print(json.dumps({
            "items": len(state["items"]),
            "pending": {
                turn_key(round_idx, agent): len(pending_indices(state, round_idx, agent))
                for round_idx in range(3) for agent in AGENTS
            },
            "completed": {
                key: sum(item["turn_status"][key] != "pending" for item in state["items"])
                for key in state["items"][0]["turn_status"]
            } if state["items"] else {},
        }, sort_keys=True))
        return
    if args.command == "run-turns":
        if not args.api_model.startswith(args.agent):
            raise SystemExit(f"agent {args.agent} got wrong model {args.api_model}")
        indices = pending_indices(state, args.round, args.agent)
        started = time.monotonic()
        def work(index: int) -> tuple[int, dict[str, Any]]:
            return index, run_turn(
                copy.deepcopy(state["items"][index]), args.round, args.agent, args.prompt_dir,
                args.api_base, args.api_model, args.api_key, args.api_timeout,
            )
        with Journal(args.state, args.journal_fsync_every) as journal:
            with ThreadPoolExecutor(max_workers=max(1, min(args.max_concurrency, len(indices) or 1))) as pool:
                futures = [pool.submit(work, index) for index in indices]
                for completed, future in enumerate(as_completed(futures), 1):
                    index, item = future.result()
                    state["items"][index] = item
                    journal.append(index, item)
                    if completed == 1 or completed % 250 == 0 or completed == len(indices):
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(f"progress={completed}/{len(indices)} round={args.round} agent={args.agent} rate={completed / elapsed:.2f}/s", flush=True)
        save_state(args.state, state)
        print(f"round={args.round} agent={args.agent} completed={len(indices)}")
        return
    if args.command == "finalize":
        rows = finalized_rows(state)
        summary = build_summary(rows, int(state["config"]["limit"]))
        if args.output.exists() and not args.resume:
            raise SystemExit(f"output exists: {args.output}")
        atomic_jsonl(args.output, rows)
        atomic_json(args.summary, summary)
        print(json.dumps(summary, sort_keys=True))
        return


if __name__ == "__main__":
    main()
