"""AgentVerse baseline for MuSiQue.

Runs the AgentVerse pipeline (recruit → decide → action → evaluate)
with 3 heterogeneous Qwen3 base agents (A1=1.7B, A2=4B, A3=8B) and a
Qwen3-8B meta model that plays both recruiter and evaluator. See the
sibling `agentverse.py` for the pipeline implementation.

Expected setup: 4 vLLM OpenAI-compatible servers already running:
    - A1_base  (Qwen3-1.7B)  on port 8201
    - A2_base  (Qwen3-4B)    on port 8202
    - A3_base  (Qwen3-8B)    on port 8203
    - Meta_8B  (Qwen3-8B)    on port 8204

Launch them via `baseline/MuSiQue/AgentVerse/run_vllm_8gpu.sh`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Dict, List

# Add Handoff-learning root and this directory to sys.path.
PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from jca.src.agents import AGENT_IDS  # noqa: E402
from jca.src.data import MuSiQueProblem, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from agentverse import (  # noqa: E402
    agentverse_record_to_dict,
    run_agentverse_for_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentVerse baseline on MuSiQue."
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Hard cap on decide/evaluate iterations.")
    parser.add_argument("--score-threshold", type=int, default=8,
                        help="Exit early once evaluator score >= this.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature-agent", type=float, default=0.7,
                        help="Sampling temperature for the 3 agents.")
    parser.add_argument("--temperature-meta", type=float, default=0.0,
                        help="Sampling temperature for recruiter/evaluator (8B).")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-base-meta", default="http://127.0.0.1:8204/v1")

    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-model-meta", default="Meta_8B")
    parser.add_argument(
        "--meta-agent",
        choices=["meta", "A1", "A2", "A3", "random"],
        default="meta",
        help="Which model plays Recruiter+Evaluator. 'random' deterministically samples A1/A2/A3 per problem.",
    )
    parser.add_argument("--meta-agent-seed", type=int, default=42)

    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=32,
                        help="Max problems processed concurrently.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-raw-chars", type=int, default=400)
    return parser.parse_args()


def _build_agent_callers(args: argparse.Namespace) -> Dict[str, OpenAIChatLLMCaller]:
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature_agent,
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


def _build_meta_caller(args: argparse.Namespace) -> OpenAIChatLLMCaller:
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature_meta,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    return OpenAIChatLLMCaller(
        args.api_base_meta, args.api_model_meta,
        generation=generation, timeout=args.api_timeout, api_key=args.api_key,
    )


def _build_agent_meta_callers(args: argparse.Namespace) -> Dict[str, OpenAIChatLLMCaller]:
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature_meta,
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


def _select_meta_agent(problem_index: int, configured: str, seed: int) -> str:
    if configured == "meta":
        return "META"
    if configured == "random":
        digest = hashlib.sha256(f"{seed}:{problem_index}".encode("utf-8")).digest()
        return AGENT_IDS[int.from_bytes(digest[:8], "big") % len(AGENT_IDS)]
    return configured


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    output_path = args.output or (
        Path("baseline/MuSiQue/AgentVerse/outputs")
        / f"{args.split}_start{args.start}_n{len(selected)}"
        f"_iters{args.max_iterations}_thr{args.score_threshold}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callers_agents = _build_agent_callers(args)
    meta_callers = {"META": _build_meta_caller(args), **_build_agent_meta_callers(args)}
    problem_indices = {problem.id: args.start + i for i, problem in enumerate(selected)}

    print("AgentVerse MuSiQue inference")
    print(f"  split: {args.split}")
    print(f"  selected: start={args.start}, n={len(selected)}")
    print(f"  max_iterations: {args.max_iterations}")
    print(f"  score_threshold: {args.score_threshold}")
    print(f"  temperature_agent: {args.temperature_agent}")
    print(f"  temperature_meta: {args.temperature_meta}")
    print(f"  meta_agent: {args.meta_agent} seed={args.meta_agent_seed}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  output: {output_path}")
    for agent_id in AGENT_IDS:
        c = callers_agents[agent_id]
        print(f"  {agent_id} api: {c.base_url} model={c.model_name} "
              f"temp={c.generation.temperature}")
    for meta_id, caller in meta_callers.items():
        print(f"  META[{meta_id}] api: {caller.base_url} model={caller.model_name} "
              f"temp={caller.generation.temperature}")

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
        problem_index = problem_indices[problem.id]
        meta_agent = _select_meta_agent(
            problem_index, args.meta_agent, args.meta_agent_seed
        )
        caller_meta = meta_callers[meta_agent]
        rec = run_agentverse_for_problem(
            problem,
            callers_agents,
            caller_meta,
            max_iterations=args.max_iterations,
            score_threshold=args.score_threshold,
        )
        pred = rec.final_answer or ""
        em, f1 = compute_em_f1(pred, problem)
        correct = is_correct(pred, problem)
        return {
            "problem": {
                "id": problem.id,
                "question": problem.question,
                "answer": problem.answer,
                "answer_aliases": problem.answer_aliases,
                "hop": problem.hop,
            },
            "agentverse": {
                **agentverse_record_to_dict(rec),
                "meta_agent": meta_agent,
                "meta_model": caller_meta.model_name,
            },
            "final_answer": rec.final_answer,
            "correct": correct,
            "em": em,
            "f1": f1,
            "wall_time_s": round(time.monotonic() - item_started, 3),
        }

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
        exits = {}
        for r in records:
            reason = r["agentverse"].get("exit_reason", "unknown")
            exits[reason] = exits.get(reason, 0) + 1
        for reason, count in exits.items():
            print(f"  exit_reason[{reason}]: {count}/{len(records)}")
        n_recruit_retry = sum(
            1 for r in records if r["agentverse"].get("recruit_retried")
        )
        n_recruit_fail = sum(
            1 for r in records if not r["agentverse"].get("recruit_parse_ok")
        )
        print(f"  recruit_retried: {n_recruit_retry}/{len(records)}")
        print(f"  recruit_failed:  {n_recruit_fail}/{len(records)}")
        n_iters = [len(r["agentverse"].get("iterations", [])) for r in records]
        if n_iters:
            print(f"  avg_iterations:  {mean(n_iters):.2f}")
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
    av = record["agentverse"]
    exit_reason = av.get("exit_reason", "?")
    n_iters = len(av.get("iterations", []))
    last_score = None
    if n_iters:
        last_eval = av["iterations"][-1].get("evaluation", {})
        last_score = last_eval.get("score")
    print(
        f"[{idx}/{total} {percent:5.1f}% elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}] "
        f"id={record['problem']['id']} em={record['em']:.0f} f1={record['f1']:.2f} "
        f"iters={n_iters} score={last_score} exit={exit_reason} "
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
