#!/usr/bin/env python3
"""Concurrent, resumable single-agent evaluation on the MATH benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jca.src.math_eval import (
    compute_math_em,
    extract_last_boxed,
    fallback_answer,
    format_math_problem_as_prompt,
    load_math_problems,
)


SYSTEM = (
    "You are an expert mathematical problem solver. Solve the problem carefully. "
    "Return a concise derivation and put the final answer in \\boxed{...}."
)


def split_thinking(content: str, reasoning_content: str = "") -> tuple[str, str]:
    matches = re.findall(r"<think\b[^>]*>(.*?)</think>", content, flags=re.I | re.S)
    thinking = "\n\n".join(x.strip() for x in matches if x.strip())
    visible = re.sub(r"<think\b[^>]*>.*?</think>", "", content, flags=re.I | re.S).strip()
    return (reasoning_content.strip() or thinking), visible


def request_seed(base_seed: int, problem_index: int, attempt: int) -> int:
    text = f"{base_seed}:{problem_index}:0:0:{attempt}:0"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0x7FFFFFFF


def request_one(args, problem_index, problem):
    last = None
    for attempt in range(args.retries + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": format_math_problem_as_prompt(problem)},
            ]
            if attempt:
                thinking_instruction = (
                    "include a non-empty thinking trace, and "
                    if args.require_thinking
                    else ""
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous response violated the protocol. Recompute the mathematics, "
                        f"{thinking_instruction}end with one final \\boxed{{...}} answer."
                    ),
                })
            max_tokens = args.max_new_tokens
            length_attempt = 0
            while True:
                payload = {
                    "model": args.api_model,
                    "messages": messages,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": max_tokens,
                    "seed": request_seed(args.seed, problem_index, attempt),
                    "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
                }
                request = urllib.request.Request(
                    args.api_base.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                )
                with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
                    body = json.loads(response.read())
                if body["choices"][0].get("finish_reason") != "length" or length_attempt >= args.length_retries:
                    break
                length_attempt += 1
                max_tokens = max(max_tokens + 1, math.ceil(max_tokens * 1.5))
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            thinking, visible = split_thinking(content, message.get("reasoning_content") or "")
            if args.require_thinking and not thinking:
                raise RuntimeError("missing thinking trace")
            final_answer = extract_last_boxed(visible)
            if not final_answer:
                raise RuntimeError("missing final boxed answer")
            return {
                "problem": problem, "response": visible, "thinking": thinking,
                "final_answer": final_answer, "error": None, "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
    return {
        "problem": problem, "response": "", "thinking": "", "final_answer": "",
        "error": last, "attempt": args.retries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--problem-ids-file", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--api-base", default="http://127.0.0.1:8501/v1")
    parser.add_argument("--api-model", default="single_agent")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--length-retries", type=int, default=1)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")

    problems = load_math_problems(args.data_root, split=args.split)
    source_index = {problem.problem_id: index for index, problem in enumerate(problems)}
    if args.problem_ids_file is not None:
        requested_ids = []
        with args.problem_ids_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    parser.error(f"invalid JSON in {args.problem_ids_file}:{line_number}: {exc}")
                if not isinstance(value, dict) or not str(value.get("problem_id", "")).strip():
                    parser.error(f"missing problem_id in {args.problem_ids_file}:{line_number}")
                requested_ids.append(str(value["problem_id"]).strip())
        if not requested_ids:
            parser.error(f"no problem IDs found in {args.problem_ids_file}")
        if len(requested_ids) != len(set(requested_ids)):
            parser.error(f"duplicate problem IDs in {args.problem_ids_file}")
        missing = [problem_id for problem_id in requested_ids if problem_id not in source_index]
        if missing:
            parser.error(f"unknown problem IDs in {args.problem_ids_file}: {missing[:5]}")
        ordered = [problems[source_index[problem_id]] for problem_id in requested_ids]
        selected = ordered[args.start : args.start + args.limit]
    else:
        selected = problems[args.start : args.start + args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if output.exists():
        for line in output.open():
            try:
                item = json.loads(line)
                done[item["problem_id"]] = item
            except (ValueError, KeyError):
                continue
    pending = [
        (source_index[p.problem_id], p)
        for p in selected
        if p.problem_id not in done
    ]
    print(f"MATH single-agent: split={args.split} selected={len(selected)} done={len(done)} pending={len(pending)}", flush=True)
    with output.open("a") as stream, ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = {
            pool.submit(request_one, args, problem_index, problem): problem
            for problem_index, problem in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            problem = result["problem"]
            response = result["response"]
            final_answer = result.get("final_answer") or fallback_answer(response) or ""
            item = {
                "schema_version": 1,
                "baseline": "math_single_agent",
                "training_free": True,
                "problem_id": problem.problem_id,
                "subject": problem.subject,
                "level": problem.level,
                "prompt": problem.prompt,
                "gold_answer": problem.gold_answer,
                "response": response,
                "thinking": result.get("thinking", ""),
                "final_answer": final_answer,
                "em": compute_math_em(final_answer, problem.gold_answer),
                "error": result["error"],
                "attempt": result["attempt"],
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            done[problem.problem_id] = item
            print(f"[{len(done)}/{len(selected)}] em={item['em']:.0f} pending={len(pending)-index}", flush=True)

    rows = [done[p.problem_id] for p in selected if p.problem_id in done]
    correct = sum(float(row.get("em", 0)) for row in rows)
    summary = {
        "baseline": "math_single_agent", "training_free": True, "split": args.split,
        "expected": len(selected), "count": len(rows), "correct": int(correct),
        "em": correct / len(selected) if selected else 0.0,
        "missing": len(selected) - len(rows), "errors": sum(bool(row.get("error")) for row in rows),
        "model": args.model or args.api_model, "api_model": args.api_model,
        "max_concurrency": args.max_concurrency,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "generation_seed": args.seed, "retries": args.retries,
        "length_retries": args.length_retries,
        "problem_ids_file": str(args.problem_ids_file.resolve()) if args.problem_ids_file else None,
        "enable_thinking": args.enable_thinking, "require_thinking": args.require_thinking,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
