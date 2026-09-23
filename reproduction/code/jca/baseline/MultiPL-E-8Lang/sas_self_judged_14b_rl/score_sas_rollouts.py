#!/usr/bin/env python3
"""Judge policy-only SAS records after standard MultiPL-E execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from sas_pipeline import JUDGE_SYSTEM_PROMPT, call, parse_object, reward_record


def score_one(record: dict, outcomes: dict, args: argparse.Namespace) -> tuple[str, dict | None, str | None]:
    row = record["row"]
    key = (record["dataset"], record["problem_id"], record["language"])
    outcome = outcomes[key]
    if record.get("policy_invalid"):
        return "rejected", {**record, "passed": outcome, "rejection_reason": "policy_invalid"}, None
    user = (
        f"language: {row['language']}\nexecution_passed: {'YES' if outcome else 'NO'}\n\n"
        f"SOURCE PREFIX:\n{row.get('prompt','')}\n\nREASONING:\n{record.get('reasoning','')}\n\n"
        f"SUBMITTED CONTINUATION:\n{record['final_completion']}"
    )
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    judge = None
    judge_raw: list[str] = []
    for _ in range(3):
        judge_raw.append(call(args.judge_api, args.judge_model, messages, max_tokens=args.judge_max_tokens, temperature=0.0))
        candidate = parse_object(judge_raw[-1])
        try:
            valid = candidate is not None and float(candidate.get("reasoning_score")) in {-1., -.5, 0., .5, 1.} and float(candidate.get("code_finalization_score")) in {-1., 0., 1.}
        except (TypeError, ValueError):
            valid = False
        if valid:
            judge = candidate
            break
    if judge is None:
        return "rejected", {**record, "passed": outcome, "judge_raw_responses": judge_raw, "rejection_reason": "judge_failed"}, str(key)
    scored = reward_record(row, record["raw_responses"], {"reasoning": record.get("reasoning", ""), "final_completion": record["final_completion"]}, outcome, judge, judge_raw, record.get("policy_model", "sas_policy"), args.judge_model)
    scored["response"] = record.get("response", scored["response"])
    scored["policy_invalid"] = bool(record.get("policy_invalid", False))
    return "scored", scored, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-api", required=True)
    parser.add_argument("--judge-model", default="qwen14b_judge")
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--rejected-output", type=Path)
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    outcomes = {}
    for line in args.outcomes.open(encoding="utf-8"):
        if line.strip():
            item = json.loads(line)
            outcomes[(item["dataset"], item["name"], item["language"])] = bool(item["passed"])
    records = [json.loads(line) for line in args.rollout.open(encoding="utf-8") if line.strip()]
    with ThreadPoolExecutor(max_workers=min(args.max_concurrency, max(1, len(records)))) as pool:
        futures = [pool.submit(score_one, record, outcomes, args) for record in records]
        results = [future.result() for future in futures]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored = [record for status, record, _ in results if status == "scored" and record is not None]
    rejected = [record for status, record, _ in results if status == "rejected" and record is not None]
    with args.output.open("w", encoding="utf-8") as sink:
        for record in scored:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.rejected_output is not None:
        args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rejected_output.open("w", encoding="utf-8") as sink:
            for record in rejected:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    judge_failed = sum(error is not None for _, _, error in results)
    print(f"scored_records={len(scored)} rejected_records={len(rejected)} judge_failed={judge_failed} max_concurrency={args.max_concurrency}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
