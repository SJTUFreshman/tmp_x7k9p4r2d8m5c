#!/usr/bin/env python3
"""Durable role-batched GPTSwarm-Fixed evaluation on MATH."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jca.src.math_eval import compute_math_em, format_math_problem_as_prompt, load_math_problems
from jca.src.relaxed_json import find_json_object


STATE_VERSION = 1
JOURNAL_VERSION = 1
NODES = ("io_0", "cot_0", "cot_1", "debate_0", "io_1", "cot_2", "aggregator")
NODE_TYPES = {
    "io_0": "IO", "io_1": "IO",
    "cot_0": "CoT", "cot_1": "CoT", "cot_2": "CoT",
    "debate_0": "Debate", "aggregator": "Aggregator",
}
NODE_AGENTS = {
    "io_0": "A1", "io_1": "A1",
    "cot_0": "A2", "cot_1": "A2", "cot_2": "A2",
    "debate_0": "A3", "aggregator": "A3",
}
DEPENDENCIES = {
    "io_0": (), "cot_0": (), "cot_1": (),
    "debate_0": ("io_0", "cot_0", "cot_1"),
    "io_1": ("debate_0",), "cot_2": ("debate_0",),
    "aggregator": ("debate_0", "cot_2", "io_1"),
}
PROMPT_FILES = {
    "IO": "node_io.md", "CoT": "node_cot.md",
    "Debate": "node_debate.md", "Aggregator": "node_aggregator.md",
}
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)
JOURNAL_FIELDS = ("node_status", "node_outputs", "status", "error", "wall_time_s")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
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
    applied = 0
    valid_bytes = 0
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
                raise ValueError(f"journal identity mismatch at {index}")
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
            "update": {key: item[key] for key in JOURNAL_FIELDS},
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
        print(f"[resume] replayed {replayed} node updates", file=sys.stderr, flush=True)
    return state


def split_thinking(raw: str) -> tuple[str, str]:
    matches = list(THINK_RE.finditer(raw))
    thinking = "\n\n".join(match.group(1).strip() for match in matches if match.group(1).strip())
    visible = THINK_RE.sub("", raw).strip()
    return thinking, visible


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


def predecessor_block(item: dict[str, Any], node: str) -> str:
    blocks = []
    for predecessor in DEPENDENCIES[node]:
        output = item["node_outputs"].get(predecessor)
        if output and output.get("success"):
            blocks.append(
                f"## Predecessor: {predecessor}\n"
                f"Reasoning: {output['reasoning']}\nAnswer: {output['answer']}"
            )
        else:
            reason = (output or {}).get("failure_reason", "missing predecessor output")
            blocks.append(f"## Predecessor: {predecessor}\nSTATUS: FAILED\nReason: {reason}")
    return "\n\n".join(blocks)


def build_messages(item: dict[str, Any], node: str, prompt_dir: Path) -> list[dict[str, str]]:
    system = (prompt_dir / PROMPT_FILES[NODE_TYPES[node]]).read_text(encoding="utf-8").strip()
    user = format_math_problem_as_prompt_obj(item["problem"])
    predecessors = predecessor_block(item, node)
    if predecessors:
        user += f"\n\n# Direct Predecessor Outputs\n{predecessors}\n\nProduce your JSON output now."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_math_problem_as_prompt_obj(problem: dict[str, Any]) -> str:
    class PromptView:
        prompt = problem["prompt"]
    return format_math_problem_as_prompt(PromptView())


def request_chat(
    api_base: str, api_model: str, api_key: str, messages: list[dict[str, str]],
    config: dict[str, Any], timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_new_tokens"],
        "seed": config["generation_seed"],
        "chat_template_kwargs": {"enable_thinking": bool(config.get("enable_thinking", False))},
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
        body = None
        if isinstance(exc, urllib.error.HTTPError):
            body = exc.read().decode("utf-8", errors="replace")
        return {
            "request_ok": False, "raw_http_response": body, "raw_output": "",
            "thinking": "", "visible_output": "", "finish_reason": None, "usage": None,
            "wall_time_s": round(time.monotonic() - started, 6),
            "exception": f"{type(exc).__name__}: {exc}",
        }


def run_one_node(
    item: dict[str, Any], node: str, prompt_dir: Path, api_base: str,
    api_model: str, api_key: str, timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    messages = build_messages(item, node, prompt_dir)
    attempts = []
    accepted = None
    failure_reason = "no request made"
    enable_thinking = bool(item["config"].get("enable_thinking", False))
    require_thinking = bool(item["config"].get("require_thinking", False))
    for retry_index in range(int(item["config"]["node_retries"]) + 1):
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
        attempt = request_chat(
            api_base, api_model, api_key, attempt_messages, item["config"], timeout,
        )
        attempt.update({"retry_index": retry_index, "messages": attempt_messages, "seed": item["config"]["generation_seed"]})
        if not attempt["request_ok"]:
            failure_reason = attempt["exception"]
        elif not enable_thinking and attempt["thinking"].strip():
            failure_reason = "unexpected thinking output while thinking is disabled"
        elif require_thinking and not attempt["thinking"].strip():
            failure_reason = "thinking is empty"
        else:
            parsed, parse_error = parse_visible_json(attempt["visible_output"])
            failure_reason = parse_error or ""
            if parsed is not None:
                accepted = parsed
        attempt["parse_ok"] = accepted is not None
        attempt["validation_error"] = None if accepted is not None else failure_reason
        attempts.append(attempt)
        if accepted is not None:
            break
    output = {
        "node_name": node, "node_type": NODE_TYPES[node], "agent": NODE_AGENTS[node],
        "api_model": api_model, "model": item["config"]["model_paths"][NODE_AGENTS[node]],
        "predecessors": list(DEPENDENCIES[node]), "messages": messages,
        "attempts": attempts, "success": accepted is not None,
        "reasoning": accepted["reasoning"] if accepted else "",
        "answer": accepted["answer"] if accepted else "",
        "failure_reason": None if accepted else failure_reason,
        "wall_time_s": round(time.monotonic() - started, 6),
    }
    item["node_outputs"][node] = output
    item["node_status"][node] = "success" if accepted else "failed"
    item["wall_time_s"] = round(float(item.get("wall_time_s", 0)) + output["wall_time_s"], 6)
    if node == "aggregator":
        item["status"] = "done" if accepted else "failed"
        item["error"] = None if accepted else f"aggregator: {failure_reason}"
    return item


def pending_indices(state: dict[str, Any], node: str) -> list[int]:
    indices = []
    for index, item in enumerate(state["items"]):
        if item["node_status"][node] != "pending":
            continue
        if all(item["node_status"][dep] in {"success", "failed"} for dep in DEPENDENCIES[node]):
            indices.append(index)
    return indices


def compact_state(state_path: Path, state: dict[str, Any]) -> None:
    save_state(state_path, state)


def total_usage(item: dict[str, Any]) -> dict[str, int | None]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = False
    for output in item["node_outputs"].values():
        for attempt in output["attempts"]:
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
        if any(status == "pending" for status in item["node_status"].values()):
            raise ValueError(f"unfinished problem: {item['problem']['problem_id']}")
        aggregator = item["node_outputs"].get("aggregator", {})
        answer = aggregator.get("answer", "") if aggregator.get("success") else ""
        problem = item["problem"]
        rows.append({
            "schema_version": 1, "baseline": "gptswarm_fixed", "training_free": True,
            "problem_index": item["problem_index"], "problem_id": problem["problem_id"],
            "subject": problem["subject"], "level": problem["level"], "prompt": problem["prompt"],
            "node_agents": NODE_AGENTS,
            "node_models": {node: state["config"]["model_paths"][NODE_AGENTS[node]] for node in NODES},
            "node_outputs": item["node_outputs"],
            "final_answer": answer, "gold_answer": problem["gold_answer"],
            "em": compute_math_em(answer, problem["gold_answer"]),
            "terminated_by": item["status"], "error": item["error"],
            "total_calls": sum(len(output["attempts"]) for output in item["node_outputs"].values()),
            "token_usage": total_usage(item), "wall_time_s": item["wall_time_s"],
            "sampling": state["config"],
        })
    return rows


def build_summary(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    ids = [row["problem_id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError(f"coverage failure: rows={len(rows)} unique={len(set(ids))} expected={expected}")
    subjects: dict[str, dict[str, Any]] = {}
    for subject in sorted({row["subject"] for row in rows}):
        subset = [row for row in rows if row["subject"] == subject]
        correct = sum(row["em"] for row in subset)
        subjects[subject] = {"count": len(subset), "correct": int(correct), "em": correct / len(subset)}
    correct = sum(row["em"] for row in rows)
    failures = Counter(row["terminated_by"] for row in rows)
    node_failures = Counter(
        name for row in rows for name, output in row["node_outputs"].items() if not output["success"]
    )
    usage = {key: sum((row["token_usage"].get(key) or 0) for row in rows) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    grouped: dict[str, dict[str, Any]] = {}
    retry_calls: Counter[int] = Counter()
    request_failures: Counter[str] = Counter()
    for row in rows:
        for name, output in row["node_outputs"].items():
            key = f"{name}|{output['node_type']}|{output['model']}"
            bucket = grouped.setdefault(key, {
                "node_name": name, "node_type": output["node_type"], "model": output["model"],
                "problems": 0, "successes": 0, "calls": 0, "wall_time_s": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            })
            bucket["problems"] += 1
            bucket["successes"] += int(output["success"])
            bucket["calls"] += len(output["attempts"])
            bucket["wall_time_s"] = round(bucket["wall_time_s"] + float(output["wall_time_s"]), 6)
            for attempt in output["attempts"]:
                retry_calls[int(attempt["retry_index"])] += 1
                if attempt.get("validation_error"):
                    request_failures[str(attempt["validation_error"])] += 1
                attempt_usage = attempt.get("usage") or {}
                for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    bucket[token_key] += int(attempt_usage.get(token_key) or 0)
    return {
        "expected": expected, "count": len(rows), "unique_problem_ids": len(set(ids)),
        "correct": int(correct), "em": correct / len(rows), "subjects": subjects,
        "terminal_status": dict(failures), "node_failures": dict(node_failures),
        "total_calls": sum(row["total_calls"] for row in rows), "token_usage": usage,
        "average_calls_per_problem": sum(row["total_calls"] for row in rows) / len(rows),
        "average_wall_time_s_per_problem": sum(float(row.get("wall_time_s", 0)) for row in rows) / len(rows),
        "by_node_type_model": grouped, "calls_by_retry_index": dict(sorted(retry_calls.items())),
        "request_failures": dict(request_failures),
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
    init.add_argument("--max-new-tokens", type=int, default=8192)
    init.add_argument("--temperature", type=float, default=0.0)
    init.add_argument("--top-p", type=float, default=0.95)
    init.add_argument("--generation-seed", type=int, default=42)
    init.add_argument("--node-retries", type=int, default=4)
    init.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--model-a1", required=True)
    init.add_argument("--model-a2", required=True)
    init.add_argument("--model-a3", required=True)
    for command in ("pending", "status"):
        sub = commands.add_parser(command)
        sub.add_argument("--state", type=Path, required=True)
        if command == "pending":
            sub.add_argument("--node", choices=NODES, required=True)
    run = commands.add_parser("run-node")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--node", choices=NODES, required=True)
    run.add_argument("--prompt-dir", type=Path, required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--api-model", required=True)
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--api-timeout", type=float, default=900)
    run.add_argument("--max-concurrency", type=int, default=128)
    run.add_argument("--journal-fsync-every", type=int, default=25)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--state", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        if args.state.exists():
            raise SystemExit(f"state exists: {args.state}")
        if args.require_thinking and not args.enable_thinking:
            raise SystemExit("--require-thinking requires --enable-thinking")
        selected = select_problems(args)
        config = {
            "split": args.split, "start": args.start, "limit": args.limit,
            "problem_ids_file": str(args.problem_ids_file.resolve()) if args.problem_ids_file else None,
            "max_new_tokens": args.max_new_tokens, "max_model_len": 40960,
            "temperature": args.temperature, "top_p": args.top_p,
            "generation_seed": args.generation_seed, "node_retries": args.node_retries,
            "enable_thinking": args.enable_thinking,
            "require_thinking": args.require_thinking,
            "model_paths": {"A1": args.model_a1, "A2": args.model_a2, "A3": args.model_a3},
        }
        items = []
        for offset, problem in enumerate(selected):
            items.append({
                "problem_index": args.start + offset,
                "problem": {
                    "problem_id": problem.problem_id, "subject": problem.subject,
                    "level": problem.level, "prompt": problem.prompt,
                    "solution": problem.solution, "gold_answer": problem.gold_answer,
                },
                "config": config, "node_status": {node: "pending" for node in NODES},
                "node_outputs": {}, "status": "pending", "error": None, "wall_time_s": 0.0,
            })
        state = {"version": STATE_VERSION, "created_at": utc_now(), "updated_at": utc_now(), "config": config, "items": items}
        save_state(args.state, state)
        print(f"initialized={len(items)} state={args.state}")
        return
    state = load_state(args.state)
    if args.command == "pending":
        print(len(pending_indices(state, args.node)))
        return
    if args.command == "status":
        print(json.dumps({
            "items": len(state["items"]), "done": sum(item["status"] == "done" for item in state["items"]),
            "failed": sum(item["status"] == "failed" for item in state["items"]),
            "pending_by_node": {node: len(pending_indices(state, node)) for node in NODES},
            "completed_by_node": {node: sum(item["node_status"][node] != "pending" for item in state["items"]) for node in NODES},
        }, sort_keys=True))
        return
    if args.command == "run-node":
        expected_agent = NODE_AGENTS[args.node]
        if not args.api_model.startswith(expected_agent):
            raise SystemExit(f"node {args.node} requires {expected_agent}, got model {args.api_model}")
        indices = pending_indices(state, args.node)
        started = time.monotonic()
        def work(index: int) -> tuple[int, dict[str, Any]]:
            return index, run_one_node(copy.deepcopy(state["items"][index]), args.node, args.prompt_dir, args.api_base, args.api_model, args.api_key, args.api_timeout)
        with Journal(args.state, args.journal_fsync_every) as journal:
            with ThreadPoolExecutor(max_workers=max(1, min(args.max_concurrency, len(indices) or 1))) as pool:
                futures = [pool.submit(work, index) for index in indices]
                for completed, future in enumerate(as_completed(futures), 1):
                    index, item = future.result()
                    state["items"][index] = item
                    journal.append(index, item)
                    if completed == 1 or completed % 250 == 0 or completed == len(indices):
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(f"progress={completed}/{len(indices)} node={args.node} rate={completed / elapsed:.2f}/s", flush=True)
        compact_state(args.state, state)
        print(f"node={args.node} completed={len(indices)}")
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
