"""Multi-Agent Debate (MAD) baseline for MuSiQue.

Runs 3 heterogeneous Qwen3 base models (A1=1.7B, A2=4B, A3=8B) with no
LoRA adapters through the standard Du et al. 2023 MAD protocol. See
the sibling `mad.py` for the protocol implementation.

Expected setup: 3 vLLM OpenAI-compatible servers already running and
serving base-model IDs (e.g. "A1_base", "A2_base", "A3_base"). Launch
them via `baseline/MuSiQue/MAD/run_vllm_8gpu.sh`.

Concurrency: problems run in parallel via a ThreadPoolExecutor.
Within each problem, the 3 agents in a debate round run concurrently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Dict, List

# Add the Handoff-learning root to sys.path so `jca.src.*` imports work.
# This file lives at jca/baseline/MuSiQue/MAD/run_mad.py, so parents[3] is
# Handoff-learning/.
PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
# Also expose this directory itself so we can import the sibling `mad.py`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from jca.src.agents import AGENT_IDS  # noqa: E402
from jca.src.data import MuSiQueProblem, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from mad import mad_record_to_dict, run_mad_for_problem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate (MAD) baseline on MuSiQue."
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="MAD rounds (>=1). Round 0 is independent; rest are debate.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature-round0", type=float, default=0.9,
                        help="Temperature for the independent round-0 turn.")
    parser.add_argument("--temperature-debate", type=float, default=0.3,
                        help="Temperature for debate rounds (round >= 1).")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=64,
                        help="Max problems processed concurrently.")
    parser.add_argument("--tie-break-order", default="A3,A2,A1",
                        help="Comma-separated agent priority for majority-vote ties.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-raw-chars", type=int, default=800)
    return parser.parse_args()


def _build_callers(
    args: argparse.Namespace,
    *,
    temperature: float,
) -> Dict[str, OpenAIChatLLMCaller]:
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    return {
        "A1": OpenAIChatLLMCaller(
            args.api_base_a1, args.api_model_a1,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
        ),
        "A2": OpenAIChatLLMCaller(
            args.api_base_a2, args.api_model_a2,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
        ),
        "A3": OpenAIChatLLMCaller(
            args.api_base_a3, args.api_model_a3,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.n_rounds < 1:
        raise SystemExit("--n-rounds must be >= 1")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")

    tie_break_priority = [t.strip() for t in args.tie_break_order.split(",") if t.strip()]
    for agent_id in tie_break_priority:
        if agent_id not in AGENT_IDS:
            raise SystemExit(f"Unknown agent in --tie-break-order: {agent_id}")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    output_path = args.output or (
        Path("outputs")
        / "mad"
        / f"{args.split}_start{args.start}_n{len(selected)}_r{args.n_rounds}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callers_round0 = _build_callers(args, temperature=args.temperature_round0)
    callers_debate = _build_callers(args, temperature=args.temperature_debate)

    print("MAD MuSiQue inference")
    print(f"  split: {args.split}")
    print(f"  selected: start={args.start}, n={len(selected)}")
    print(f"  n_rounds: {args.n_rounds}")
    print(f"  temperature_round0: {args.temperature_round0}")
    print(f"  temperature_debate: {args.temperature_debate}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  tie_break_order: {tie_break_priority}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  output: {output_path}")
    for agent_id in AGENT_IDS:
        c0 = callers_round0[agent_id]
        cd = callers_debate[agent_id]
        print(f"  {agent_id} api: {c0.base_url} model={c0.model_name} "
              f"temps=(r0={c0.generation.temperature}, deb={cd.generation.temperature})")

    if args.dry_run:
        first = selected[0]
        print("\nDry run only. First selected problem:")
        print(f"  id: {first.id}")
        print(f"  question: {first.question}")
        print(f"  gold: {first.answer}")
        return

    run_started = time.monotonic()
    file_lock = Lock()
    records: List[Dict[str, Any]] = []

    def _process(problem: MuSiQueProblem) -> Dict[str, Any]:
        item_started = time.monotonic()
        rec = run_mad_for_problem(
            problem,
            callers_round0,
            callers_debate,
            n_rounds=args.n_rounds,
            tie_break_priority=tie_break_priority,
            seed=hash(problem.id) & 0xFFFFFFFF,
        )
        pred = rec.final_answer or ""
        em, f1 = compute_em_f1(pred, problem)
        correct = is_correct(pred, problem)
        record = {
            "problem": {
                "id": problem.id,
                "question": problem.question,
                "answer": problem.answer,
                "answer_aliases": problem.answer_aliases,
                "hop": problem.hop,
            },
            "mad": mad_record_to_dict(rec),
            "final_answer": rec.final_answer,
            "correct": correct,
            "em": em,
            "f1": f1,
            "wall_time_s": round(time.monotonic() - item_started, 3),
        }
        return record

    with output_path.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = {executor.submit(_process, p): p for p in selected}
            for future in as_completed(futures):
                problem = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    print(f"[error] problem={problem.id}: {type(exc).__name__}: {exc}")
                    continue

                with file_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    records.append(record)

                idx = _increment_progress()
                _log_progress(record, run_started, len(selected), idx)

    print("\nDone.")
    if records:
        print(f"  accuracy_em: {mean(r['em'] for r in records):.3f}")
        print(f"  avg_f1:      {mean(r['f1'] for r in records):.3f}")
        n_tie = sum(1 for r in records if r["mad"].get("tie_break"))
        n_err = sum(1 for r in records if r["mad"].get("error"))
        print(f"  tie_break_used: {n_tie}/{len(records)}")
        print(f"  errors:         {n_err}/{len(records)}")
    print(f"  output: {output_path}")
    print(f"  total_time: {_format_duration(time.monotonic() - run_started)}")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


_PROGRESS_COUNTER = [0]
_PROGRESS_LOCK = Lock()


def _increment_progress() -> int:
    with _PROGRESS_LOCK:
        _PROGRESS_COUNTER[0] += 1
        return _PROGRESS_COUNTER[0]


def _log_progress(record: Dict[str, Any], started_at: float, total: int, idx: int) -> None:
    elapsed = time.monotonic() - started_at
    avg = elapsed / idx if idx > 0 else 0.0
    eta = avg * max(total - idx, 0)
    percent = 100.0 * idx / total if total else 100.0
    mad = record["mad"]
    voters = mad.get("voter_answers", {}) or {}
    voters_str = ", ".join(f"{k}={_one_line(str(v))[:40]}" for k, v in voters.items())
    print(
        f"[{idx}/{total} {percent:5.1f}% elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}] "
        f"id={record['problem']['id']} em={record['em']:.0f} f1={record['f1']:.2f} "
        f"tie={mad.get('tie_break')} err={mad.get('error') is not None} "
        f"voters=({voters_str}) "
        f"final={_one_line(str(record.get('final_answer')))[:60]}"
    )


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


if __name__ == "__main__":
    main()
