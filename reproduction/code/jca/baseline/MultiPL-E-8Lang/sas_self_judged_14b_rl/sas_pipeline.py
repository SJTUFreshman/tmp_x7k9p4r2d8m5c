#!/usr/bin/env python3
"""One-shot MultiPL-E SAS rollout and Qwen14B self-judge record builder."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are a strict one-shot MultiPL-E code completion agent.
Return exactly one JSON object with string fields:
{"reasoning":"brief implementation reasoning", "final_completion":"code appended at the cursor"}
The final_completion must contain only the missing source continuation. Do not repeat the source
prefix or function declaration, do not include markdown, explanations, imports, or tests, and do
not emit thinking blocks."""

RETRY_PROMPT = "Your previous response was invalid. Return only the required JSON object with a non-empty final_completion. Do not add markdown or commentary."

JUDGE_SYSTEM_PROMPT = """You are an expert judge for a one-shot MultiPL-E code-completion trajectory.
You will receive the target language, source prefix, execution pass/fail, model reasoning, and
submitted continuation. Judge PROCESS QUALITY, not only execution correctness.

reasoning_score must be one of {-1.0,-0.5,0.0,0.5,1.0}:
+1.0 correct implementation, edge cases, target-language details, and completion boundary;
+0.5 mostly correct and useful with minor omissions;
 0.0 generic, shallow, empty, or not auditable;
-0.5 materially confused or missing an important issue;
-1.0 fundamentally wrong or strongly misleading.

code_finalization_score must be one of {-1.0,0.0,1.0} and must not judge semantic correctness:
+1.0 one clear, complete continuation respecting the cursor boundary and suitable in form for evaluation;
 0.0 intended continuation is identifiable but has a minor completeness, formatting, or boundary problem;
-1.0 empty, severely malformed, conflicting, or not usable as a stable continuation.

Code must contain only text appended at the cursor. Repeating the source prefix or function
declaration is usually bad. Respect the evaluator's final-delimiter convention. Execution
pass/fail is context, not the scoring rule; do not copy it into both scores.
Output only one JSON object with reasoning_score, code_finalization_score, and comment."""


def call(base: str, model: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode())
    return str(data["choices"][0]["message"].get("content", ""))


def parse_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def policy_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    language = row["language"]
    user = (f"Target language: {language}\n\n"
            "Generate only the source text inserted after the cursor.\n"
            "SOURCE PREFIX:\n" + str(row.get("prompt", "")))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def policy_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_responses = [call(args.policy_api, args.policy_model, policy_messages(row), max_tokens=args.max_new_tokens, temperature=args.temperature)]
    parsed = parse_object(raw_responses[-1])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("final_completion"), str) or not parsed.get("final_completion", "").strip():
        retry_messages = policy_messages(row) + [{"role": "user", "content": RETRY_PROMPT}]
        raw_responses.append(call(args.policy_api, args.policy_model, retry_messages, max_tokens=args.max_new_tokens, temperature=args.temperature))
        parsed = parse_object(raw_responses[-1])
    policy_invalid = not isinstance(parsed, dict) or not str(parsed.get("final_completion", "")).strip()
    candidate = parsed if isinstance(parsed, dict) else {}
    parsed = {
        "reasoning": str(candidate.get("reasoning", "")),
        "final_completion": str(candidate.get("final_completion", "")) or raw_responses[-1],
    }
    return {
        "dataset": row.get("dataset"), "problem_id": row.get("name"), "language": row.get("language"),
        "row": row, "messages": policy_messages(row), "response": raw_responses[-1] if policy_invalid else json.dumps(parsed, ensure_ascii=False),
        "raw_response": raw_responses[-1], "raw_responses": raw_responses,
        "reasoning": parsed["reasoning"], "final_completion": parsed["final_completion"],
        "policy_model": args.policy_model, "policy_invalid": policy_invalid, "enable_thinking": False,
    }


def reward_record(
    row: dict[str, Any],
    raw_responses: list[str],
    parsed: dict[str, Any],
    outcome: bool,
    judge: dict[str, Any] | None,
    judge_raw_responses: list[str],
    policy_model: str,
    judge_model: str,
) -> dict[str, Any]:
    task_reward = 1.0 if outcome else -1.0
    if judge is None:
        judge_score = 0.0
        turn_scores: dict[str, Any] = {}
        reward = task_reward
    else:
        reasoning = float(judge["reasoning_score"])
        finalization = float(judge["code_finalization_score"])
        judge_score = 0.5 * (reasoning + finalization)
        turn_scores = {
            "reasoning_score": reasoning,
            "code_finalization_score": finalization,
            "judge_score": judge_score,
            "comment": str(judge.get("comment", "")),
        }
        reward = max(-1.0, min(1.0, 0.3 * task_reward + 0.7 * judge_score))
    return {
        "dataset": row.get("dataset"),
        "problem_id": row.get("name"),
        "language": row.get("language"),
        "row": row,
        "turn": 0,
        "agent_id": "SAS",
        "messages": policy_messages(row),
        "response": json.dumps(parsed, ensure_ascii=False),
        "raw_response": raw_responses[-1] if raw_responses else "",
        "raw_responses": raw_responses,
        "judge_raw_response": judge_raw_responses[-1] if judge_raw_responses else "",
        "judge_raw_responses": judge_raw_responses,
        "reasoning": str(parsed.get("reasoning", "")),
        "final_completion": str(parsed.get("final_completion", "")),
        "passed": bool(outcome),
        "task_reward": round(task_reward, 4),
        "judge_score": round(judge_score, 4),
        "turn_scores": turn_scores,
        "reward": round(reward, 4),
        "reward_alpha": 0.3,
        "reward_source": "qwen14b_sas_judge",
        "reward_semantics": "multipl_e_sas_execution_judge_v1",
        "judge_failed": judge is None,
        "policy_model": policy_model,
        "judge_model": judge_model,
        "enable_thinking": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, help="JSONL keyed by dataset/name/language with passed boolean")
    parser.add_argument("--policy-api", required=True)
    parser.add_argument("--policy-model", default="sas_policy")
    parser.add_argument("--judge-api")
    parser.add_argument("--judge-model", default="qwen14b_multipl_e_judge")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--policy-only", action="store_true", help="Write policy responses without execution/judge scoring")
    parser.add_argument("--limit", type=int, help="Only process the first N non-empty input rows")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Append after a validated existing output prefix")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if not args.policy_only and (args.outcomes is None or args.judge_api is None):
        parser.error("--outcomes and --judge-api are required unless --policy-only is set")
    outcomes = {}
    if not args.policy_only:
        assert args.outcomes is not None
        for line in args.outcomes.open(encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                key = (item.get("dataset"), item.get("name", item.get("problem_id")), item.get("language"))
                outcomes[key] = bool(item.get("passed"))
    existing_records: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as existing_source:
            existing_records = [json.loads(line) for line in existing_source if line.strip()]
        print(f"resuming_after_records={len(existing_records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.policy_only:
        rows = []
        with args.input.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append(json.loads(line))
                    if args.limit is not None and len(rows) >= args.limit:
                        break
        existing = 0
        if args.resume and args.output.is_file():
            existing = sum(1 for line in args.output.open(encoding="utf-8") if line.strip())
        pending_rows = rows[existing:]
        records: list[dict[str, Any] | None] = [None] * len(pending_rows)
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, max(1, len(pending_rows)))) as pool:
            futures = {pool.submit(policy_record, row, args): index for index, row in enumerate(pending_rows)}
            for future in futures:
                records[futures[future]] = future.result()
        with args.output.open("a" if args.resume else "w", encoding="utf-8") as sink:
            for record in records:
                assert record is not None
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"policy_rollout_records={len(records)} max_concurrency={args.max_concurrency}")
        return 0
    output_mode = "a" if args.resume else "w"
    with args.input.open(encoding="utf-8") as source, args.output.open(output_mode, encoding="utf-8") as sink:
        processed = 0
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            row = json.loads(line); key = (row.get("dataset"), row.get("name"), row.get("language"))
            if processed <= len(existing_records):
                existing = existing_records[processed - 1]
                existing_key = (existing.get("dataset"), existing.get("problem_id"), existing.get("language"))
                if existing_key != key:
                    raise RuntimeError(
                        f"resume prefix mismatch at record {processed}: input={key}, output={existing_key}"
                    )
                continue
            raw_responses = [call(args.policy_api, args.policy_model, policy_messages(row), max_tokens=args.max_new_tokens, temperature=args.temperature)]
            parsed = parse_object(raw_responses[-1])
            retry_used = False
            if not isinstance(parsed, dict) or not isinstance(parsed.get("final_completion"), str) or not parsed.get("final_completion", "").strip():
                retry_used = True
                retry_messages = policy_messages(row) + [{"role": "user", "content": RETRY_PROMPT}]
                raw_responses.append(call(args.policy_api, args.policy_model, retry_messages, max_tokens=args.max_new_tokens, temperature=args.temperature))
                parsed = parse_object(raw_responses[-1])
            policy_invalid = not isinstance(parsed, dict) or not str(parsed.get("final_completion", "")).strip()
            if policy_invalid:
                candidate = parsed if isinstance(parsed, dict) else {}
                parsed = {
                    "reasoning": str(candidate.get("reasoning", "")),
                    "final_completion": str(candidate.get("final_completion", "")) or raw_responses[-1],
                }
            response = raw_responses[-1] if policy_invalid else json.dumps(parsed, ensure_ascii=False)
            if args.policy_only:
                sink.write(json.dumps({"dataset": row.get("dataset"), "problem_id": row.get("name"), "language": row.get("language"), "row": row, "messages": policy_messages(row), "response": response, "raw_response": raw_responses[-1], "raw_responses": raw_responses, "reasoning": str(parsed.get("reasoning", "")), "final_completion": str(parsed.get("final_completion", "")), "policy_model": args.policy_model, "policy_invalid": policy_invalid, "enable_thinking": False}, ensure_ascii=False) + "\n")
                sink.flush()
                continue
            outcome = outcomes[key]
            judge_user = (f"language: {row['language']}\nexecution_passed: {'YES' if outcome else 'NO'}\n\nSOURCE PREFIX:\n{row.get('prompt','')}\n\nREASONING:\n{parsed.get('reasoning','')}\n\nSUBMITTED CONTINUATION:\n{parsed['final_completion']}")
            judge_messages = [{"role":"system","content":JUDGE_SYSTEM_PROMPT},{"role":"user","content":judge_user}]
            judge = None
            judge_raw_responses: list[str] = []
            for _ in range(3):
                judge_raw_responses.append(call(args.judge_api, args.judge_model, judge_messages, max_tokens=args.judge_max_tokens, temperature=0.0))
                candidate = parse_object(judge_raw_responses[-1])
                try:
                    valid = (candidate is not None
                             and float(candidate.get("reasoning_score")) in {-1.0,-0.5,0.0,0.5,1.0}
                             and float(candidate.get("code_finalization_score")) in {-1.0,0.0,1.0})
                except (TypeError, ValueError):
                    valid = False
                if valid:
                    judge = candidate
                    break
            if judge is None:
                print(f"warning: invalid judge response after 2 retries at line {line_no}")
            record = reward_record(row, raw_responses, parsed, outcome, judge, judge_raw_responses, args.policy_model, args.judge_model)
            record["response"] = response
            record["policy_invalid"] = policy_invalid
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
