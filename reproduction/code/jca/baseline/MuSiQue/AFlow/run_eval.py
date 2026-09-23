"""AFlow eval runner: apply one workflow file to a MuSiQue split, output EM/F1.

Two typical uses:
  1. AFlow-Initial main-table number:
       python run_eval.py --workflow-file workflows/round_00_initial.py --limit 2417
  2. AFlow-Search main-table number (after run_search.py picks a best node):
       python run_eval.py --workflow-file search_runs/<RUN_ID>/workflows/round_NN_*.py --limit 2417
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.data import MuSiQueProblem, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from workflow import ExecutorCallers, load_workflow_from_file, run_workflow_on_problem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AFlow workflow on MuSiQue.")
    parser.add_argument("--workflow-file", type=Path, required=True,
                        help="Path to a workflow .py file to evaluate.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument(
        "--judge-agent",
        choices=["A1", "A2", "A3", "random"],
        default="A3",
        help="Which solver model plays ensemble+answer_generate judge. 'random' samples A1/A2/A3 per problem.",
    )
    parser.add_argument("--judge-agent-seed", type=int, default=42)
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_judge_agent(problem_index: int, configured: str, seed: int) -> str:
    if configured == "random":
        digest = hashlib.sha256(f"{seed}:{problem_index}".encode("utf-8")).digest()
        return ["A1", "A2", "A3"][int.from_bytes(digest[:8], "big") % 3]
    return configured


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not args.workflow_file.exists():
        raise SystemExit(f"Workflow file not found: {args.workflow_file}")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    workflow = load_workflow_from_file(args.workflow_file, round_id=0)

    output_path = args.output
    if output_path is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = (
            _HERE / "outputs"
            / f"eval_{args.workflow_file.stem}_{args.split}_start{args.start}_n{len(selected)}_{run_timestamp}.jsonl"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    a1 = OpenAIChatLLMCaller(args.api_base_a1, args.api_model_a1, generation=generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    a2 = OpenAIChatLLMCaller(args.api_base_a2, args.api_model_a2, generation=generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    a3 = OpenAIChatLLMCaller(args.api_base_a3, args.api_model_a3, generation=generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    solve_callers = [a1, a2, a3]
    solve_caller_ids = ["A1_1.7B", "A2_4B", "A3_8B"]
    judge_callers = {"A1": a1, "A2": a2, "A3": a3}
    judge_caller_ids = {"A1": "A1_1.7B", "A2": "A2_4B", "A3": "A3_8B"}
    problem_indices = {problem.id: args.start + i for i, problem in enumerate(selected)}

    print("AFlow eval")
    print(f"  workflow_file:   {args.workflow_file}")
    print(f"  split:           {args.split}")
    print(f"  selected:        start={args.start}, n={len(selected)}")
    print("  solve pool:      A1_1.7B, A2_4B, A3_8B (round-robin)")
    print(f"  judge_agent:     {args.judge_agent} seed={args.judge_agent_seed}")
    for judge_id, caller in judge_callers.items():
        print(f"  judge[{judge_id}]:    {caller.base_url} model={caller.model_name}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  enable_thinking: {args.enable_thinking}")
    print(f"  output:          {output_path}")

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
        judge_agent = _select_judge_agent(
            problem_index, args.judge_agent, args.judge_agent_seed
        )
        callers = ExecutorCallers(
            solve_callers,
            judge_callers[judge_agent],
            solve_caller_ids,
            judge_caller_ids[judge_agent],
        )
        rec = run_workflow_on_problem(workflow, problem, callers)
        pred = rec.final_answer or ""
        pred_wrapped = pred if "\\boxed" in pred else (f"\\boxed{{{pred}}}" if pred else "")
        em, f1 = compute_em_f1(pred_wrapped, problem)
        correct = is_correct(pred_wrapped, problem)
        return {
            "problem": {
                "id": problem.id,
                "question": problem.question,
                "answer": problem.answer,
                "answer_aliases": problem.answer_aliases,
                "hop": problem.hop,
            },
            "final_answer": pred,
            "correct": correct,
            "em": em,
            "f1": f1,
            "n_ops": len(rec.op_calls),
            "op_kinds": [c.op for c in rec.op_calls],
            "op_records": [asdict(c) for c in rec.op_calls],
            "judge_agent": judge_agent,
            "judge_model": judge_callers[judge_agent].model_name,
            "error": rec.error,
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
        print(f"  accuracy_em:  {mean(r['em'] for r in records):.3f}")
        print(f"  avg_f1:       {mean(r['f1'] for r in records):.3f}")
        avg_ops = mean(r['n_ops'] for r in records)
        n_err = sum(1 for r in records if r.get('error'))
        print(f"  avg_ops/prob: {avg_ops:.2f}")
        print(f"  errors:       {n_err}/{len(records)}")
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
    print(
        f"[{idx}/{total} {percent:5.1f}% elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}] "
        f"id={record['problem']['id']} em={record['em']:.0f} f1={record['f1']:.2f} "
        f"n_ops={record['n_ops']} err={record.get('error') is not None} "
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
