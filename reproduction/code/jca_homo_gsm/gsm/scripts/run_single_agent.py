"""GSM-HARD single-agent inference script.

Runs a single Qwen3-8B model (no MAS) to establish baseline performance.

Usage:
    python jca/gsm/scripts/run_single_agent.py \
        --data-path jca/MATH-data/GSM-HARD/gsmhardv2.jsonl \
        --start 0 --limit 100 \
        --api-base http://127.0.0.1:8203/v1 \
        --api-model A3 \
        --output results/gsm_single_a3.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jca.gsm.src.data import GSMProblem, load_gsm_hard, format_problem_as_prompt, extract_number
from jca.gsm.src.grader import compute_em_f1, is_correct
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller

SINGLE_AGENT_SYSTEM = """You are an expert math solver. Solve the problem step by step.
Show all arithmetic clearly. Your final answer must be a single number.

Output format:
  Reasoning: <step-by-step calculation>
  Answer: <number only, e.g. 42 or -9867630>
"""


_REQUEST_SEED_MODULUS = 2**31 - 1


def derive_generation_request_seed(
    generation_seed: Optional[int],
    problem_id: str,
) -> Optional[int]:
    if generation_seed is None:
        return None
    payload = json.dumps(
        [generation_seed, problem_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _REQUEST_SEED_MODULUS


def solve_single(
    problem: GSMProblem,
    caller: OpenAIChatLLMCaller,
    *,
    generation_seed: Optional[int] = None,
) -> tuple[Optional[str], str]:
    """Returns (final_answer, raw_output)."""
    messages = [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM},
        {"role": "user",   "content": format_problem_as_prompt(problem)},
    ]
    try:
        request_seed = derive_generation_request_seed(generation_seed, problem.id)
        raw = (
            caller(messages)
            if request_seed is None
            else caller.generate(messages, seed=request_seed)
        )
        # Extract answer from output
        import re
        # Try "Answer: X"
        m = re.search(r'[Aa]nswer\s*:\s*([-\d,\.]+)', raw)
        if m:
            ans = m.group(1).replace(',', '')
            return ans, raw
        # Fallback: extract_number
        num = extract_number(raw)
        if num is not None:
            if num == int(num):
                return str(int(num)), raw
            return str(num), raw
        return None, raw
    except Exception as e:
        return None, str(e)


def format_duration(sec: float) -> str:
    if sec < 60: return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    if m < 60: return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/data/wangyuheng/jca/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl")
    p.add_argument("--start",  type=int, default=0)
    p.add_argument("--limit",  type=int, default=100)
    p.add_argument("--t-max",  type=int, default=1)   # unused, kept for consistency
    p.add_argument("--api-base",   default="http://127.0.0.1:8203/v1")
    p.add_argument("--api-model",  default="A3")
    p.add_argument("--api-key",    default="EMPTY")
    p.add_argument("--api-timeout",type=int, default=120)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--top-p",          type=float, default=0.95)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking through the chat template.",
    )
    p.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Optional base seed; each problem gets a stable derived request seed.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of problems to request in parallel.",
    )
    p.add_argument("--output", type=Path, default=Path("results/gsm_single.jsonl"))
    p.add_argument("--log-raw-chars", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")

    problems = load_gsm_hard(args.data_path)
    selected = problems[args.start: args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    caller = OpenAIChatLLMCaller(
        args.api_base, args.api_model,
        generation=generation, timeout=args.api_timeout, api_key=args.api_key,
    )

    print(f"GSM-HARD Single Agent ({args.api_model})")
    print(f"  n={len(selected)}  start={args.start}")
    print(f"  max_concurrency={args.max_concurrency}")
    print(f"  enable_thinking={args.enable_thinking}")
    if args.generation_seed is not None:
        print(f"  generation_seed={args.generation_seed}")
    print(f"  output={args.output}")

    run_start = time.monotonic()

    def process_one(problem: GSMProblem) -> tuple[GSMProblem, Optional[str], dict[str, Any]]:
        final_answer, raw = solve_single(
            problem,
            caller,
            generation_seed=args.generation_seed,
        )
        em, f1 = compute_em_f1(final_answer or "", problem)
        record: dict[str, Any] = {
            "problem": {
                "id":       problem.id,
                "question": problem.question,
                "answer":   problem.answer_str,
            },
            "final_answer":  final_answer,
            "raw_output":    raw[:args.log_raw_chars],
            "terminated_by": "stop" if final_answer else "exception",
            "em":  em,
            "f1":  f1,
        }
        if args.generation_seed is not None:
            record["generation_seed"] = args.generation_seed
        return problem, final_answer, record

    records: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        if args.max_concurrency <= 1:
            for idx, problem in enumerate(selected, start=1):
                t0 = time.monotonic()
                problem, final_answer, record = process_one(problem)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)

                elapsed = time.monotonic() - run_start
                avg = elapsed / idx
                eta  = avg * (len(selected) - idx)
                print(
                    f"[{idx}/{len(selected)}  elapsed={format_duration(elapsed)}"
                    f"  eta={format_duration(eta)}] {problem.id}"
                )
                print(f"  gold={problem.answer_str}  pred={final_answer}  em={record['em']:.3f}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            done = 0
            next_record_index = 0
            pending_records: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(
                max_workers=min(args.max_concurrency, len(selected)),
            ) as pool:
                futures = {
                    pool.submit(process_one, problem): index
                    for index, problem in enumerate(selected)
                }
                for future in as_completed(futures):
                    problem, final_answer, record = future.result()
                    pending_records[futures[future]] = record
                    done += 1

                    elapsed = time.monotonic() - run_start
                    rate = done / max(elapsed, 1e-6)
                    print(
                        f"[{done}/{len(selected)}  elapsed={format_duration(elapsed)}"
                        f"  rate={rate:.2f}/s] {problem.id}"
                    )
                    print(
                        f"  gold={problem.answer_str}  pred={final_answer}"
                        f"  em={record['em']:.3f}"
                    )
                    while next_record_index in pending_records:
                        ordered_record = pending_records.pop(next_record_index)
                        handle.write(json.dumps(ordered_record, ensure_ascii=False) + "\n")
                        handle.flush()
                        records.append(ordered_record)
                        next_record_index += 1

    print(f"\nDone.")
    print(f"  EM:  {mean(r['em'] for r in records):.4f}")
    print(f"  F1:  {mean(r['f1'] for r in records):.4f}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
