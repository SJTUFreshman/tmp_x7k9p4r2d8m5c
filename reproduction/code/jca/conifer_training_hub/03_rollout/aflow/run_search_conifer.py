#!/usr/bin/env python3
"""Search an AFlow workflow on a fixed shuffled Conifer train-20 subset.

Mirrors `baseline/MuSiQue/AFlow/run_search.py`; the task-specific seams are
installed by `conifer_aflow_adapter.configure()`.  The search signal is
Conifer's reported `hard_score`, exactly as the other datasets search on EM.

Never reads test.jsonl: the reported evaluation set must stay unseen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent

from conifer_aflow_adapter import (  # noqa: E402  (installs sys.path)
    INITIAL_WORKFLOW,
    METRIC_NAME,
    ConiferProblem,
    configure,
)

_MODULES = configure()
optimizer = _MODULES["optimizer"]

from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from workflow import ExecutorCallers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Conifer train jsonl. Passing a test split is refused.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-workflow", type=Path, default=INITIAL_WORKFLOW)
    parser.add_argument("--search-size", type=int, default=20)
    parser.add_argument("--search-seed", type=int, default=20260810)
    parser.add_argument("--rng-seed", type=int, default=20260810)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=20)

    parser.add_argument("--max-new-tokens-executor", type=int, default=1024)
    parser.add_argument("--max-new-tokens-optimizer", type=int, default=2048)
    parser.add_argument("--temperature-executor", type=float, default=0.7)
    parser.add_argument("--temperature-optimizer", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--api-base-a1", required=True)
    parser.add_argument("--api-base-a2", required=True)
    parser.add_argument("--api-base-a3", required=True)
    parser.add_argument("--api-base-optimizer", required=True)
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-model-optimizer", default="Optimizer_14B")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config and dev subset, then exit.")
    return parser.parse_args()


def load_dev(path: Path, size: int, seed: int) -> list[ConiferProblem]:
    if "test" in path.name:
        raise SystemExit(
            f"refusing to search on what looks like a test split: {path}. "
            "AFlow search must run on train."
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < size:
        raise SystemExit(f"need at least {size} rows, found {len(rows)} in {path}")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return [ConiferProblem.from_row(row) for row in shuffled[:size]]


def write_search_metrics(run_dir: Path, nodes: list, best) -> dict[str, Any]:
    """Per-round metric table plus 'did the search actually do anything'.

    Conifer's hard_score saturates on ~64% of problems, so a plausible outcome
    is that no proposal beats round 0.  That is a valid result, but it has to
    be legible afterwards instead of hiding behind a single number.
    """
    metric_keys = (
        "hard_score", "explicit_score", "requirement_coverage",
        "reference_lexical_f1", "word_count",
    )
    rounds = []
    for node in nodes:
        records = node.dev_records or []
        entry: dict[str, Any] = {
            "round_id": node.round_id,
            "name": node.name,
            "parent_round": node.parent_round,
            "proposed_by": node.proposed_by,
            "parse_ok": node.parse_ok,
            "reject_reason": node.reject_reason,
            "visits": node.visits,
            "dev_hard_score": node.dev_em,
            "dev_reference_lexical_f1": node.dev_f1,
            "n_dev_records": len(records),
            "mean_ops_per_problem": (
                round(fmean([r.get("n_ops", 0) for r in records]), 3) if records else None
            ),
            "n_errors": sum(1 for r in records if r.get("error")),
        }
        for key in metric_keys:
            values = [
                float(r["hard_checks"][key])
                for r in records
                if isinstance((r.get("hard_checks") or {}).get(key), (int, float))
            ]
            entry[f"mean_{key}"] = round(fmean(values), 5) if values else None
        rounds.append(entry)

    valid = [node for node in nodes if node.parse_ok]
    baseline = next((node for node in nodes if node.round_id == 0), None)
    baseline_score = baseline.dev_em if baseline is not None else None
    beat = [n.round_id for n in valid if baseline_score is not None and n.dev_em > baseline_score]
    tied = [
        n.round_id for n in valid
        if baseline_score is not None and abs(n.dev_em - baseline_score) < 1e-9
    ]
    metrics = {
        "metric": METRIC_NAME,
        "search_signal": "dev_em = hard_score (UCB exploit); dev_f1 = reference_lexical_f1 (tie-break only)",
        "total_nodes": len(nodes),
        "parse_ok_nodes": len(valid),
        "rejected_nodes": len(nodes) - len(valid),
        "round_0_hard_score": baseline_score,
        "rounds_beating_round_0": beat,
        "rounds_tied_with_round_0": tied,
        "search_improved_over_initial": bool(beat),
        "selected_round": best.round_id if best is not None else None,
        "selected_is_initial": bool(best is not None and best.round_id == 0),
        "selected_hard_score": best.dev_em if best is not None else None,
        "selected_source": best.source_file if best is not None else None,
        "rounds": rounds,
    }
    (run_dir / "search_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.search_size < 1:
        raise SystemExit("--search-size must be >= 1")
    if not args.initial_workflow.exists():
        raise SystemExit(f"initial workflow not found: {args.initial_workflow}")

    dev = load_dev(args.data_path, args.search_size, args.search_seed)

    executor_generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens_executor,
        temperature=args.temperature_executor,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )

    def executor(base: str, model: str) -> OpenAIChatLLMCaller:
        # response_format is deliberately unset: the AFlow operator schemas
        # differ from the Conifer protocol schema, so Conifer's guided decoding
        # cannot be reused here (see AFLOW_SEARCH_MIGRATION.md 4.2).
        return OpenAIChatLLMCaller(
            base, model, generation=executor_generation,
            timeout=args.api_timeout, api_key=args.api_key,
            max_model_len=args.max_model_len,
        )

    a1 = executor(args.api_base_a1, args.api_model_a1)
    a2 = executor(args.api_base_a2, args.api_model_a2)
    a3 = executor(args.api_base_a3, args.api_model_a3)
    executor_callers = ExecutorCallers(
        [a1, a2, a3], a3, ["A1_1.7B", "A2_4B", "A3_8B"], "A3_8B"
    )
    optimizer_caller = OpenAIChatLLMCaller(
        args.api_base_optimizer,
        args.api_model_optimizer,
        generation=GenerationOptions(
            max_new_tokens=args.max_new_tokens_optimizer,
            temperature=args.temperature_optimizer,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        ),
        timeout=args.api_timeout,
        api_key=args.api_key,
        max_model_len=args.max_model_len,
    )

    print("Conifer AFlow MCTS search")
    print(f"  run_dir:         {args.run_dir}")
    print(f"  initial:         {args.initial_workflow}")
    print(f"  data:            {args.data_path}")
    print(f"  search_size:     {args.search_size} (shuffle seed {args.search_seed})")
    print(f"  max_iterations:  {args.max_iterations}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  enable_thinking: {args.enable_thinking}")
    print(f"  metric:          {METRIC_NAME}")
    print(f"  exec tokens:     {args.max_new_tokens_executor} @ T={args.temperature_executor}")
    print(f"  opt tokens:      {args.max_new_tokens_optimizer} @ T={args.temperature_optimizer}")
    print("  solve pool:      A1_1.7B, A2_4B, A3_8B (round-robin)")
    print(f"  judge:           {a3.base_url} model={a3.model_name}")
    print(f"  optimizer:       {optimizer_caller.base_url} model={optimizer_caller.model_name}")
    print(f"  dev problem ids: {[problem.id for problem in dev]}")

    if args.dry_run:
        print("\n[dry-run] configuration resolved; no server was contacted")
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    nodes = optimizer.run_mcts(
        root=args.run_dir,
        initial_source_path=args.initial_workflow,
        dev_problems=dev,
        executor_callers=executor_callers,
        optimizer_caller=optimizer_caller,
        max_iterations=args.max_iterations,
        max_concurrency=args.max_concurrency,
        rng_seed=args.rng_seed,
        optimizer_caller_id=args.api_model_optimizer,
    )

    best = optimizer.best_node(nodes)
    metrics = write_search_metrics(args.run_dir, nodes, best)

    print("\n================ Search Complete ================")
    print(f"  total_nodes:     {metrics['total_nodes']} parse_ok={metrics['parse_ok_nodes']}")
    print(f"  round 0 hard:    {metrics['round_0_hard_score']}")
    print(f"  beat round 0:    {metrics['rounds_beating_round_0'] or 'none'}")
    print(f"  tied w/ round 0: {metrics['rounds_tied_with_round_0'] or 'none'}")
    if best is not None:
        print(f"  selected:        round={best.round_id} name={best.name} "
              f"hard={best.dev_em:.4f} lex_f1={best.dev_f1:.4f}")
        print(f"  selected source: {best.source_file}")
    if not metrics["search_improved_over_initial"]:
        print("  NOTE: no proposal beat the hand-written initial workflow.")

    summary_lines = [
        f"RUN_DIR={args.run_dir}",
        f"METRIC={METRIC_NAME}",
        f"MAX_ITERATIONS={args.max_iterations}",
        f"SEARCH_SIZE={args.search_size}",
        f"TOTAL_NODES={metrics['total_nodes']}",
        f"PARSE_OK_NODES={metrics['parse_ok_nodes']}",
        f"SEARCH_IMPROVED_OVER_INITIAL={metrics['search_improved_over_initial']}",
    ]
    if best is not None:
        summary_lines += [
            f"BEST_ROUND={best.round_id}",
            f"BEST_HARD_SCORE={best.dev_em:.4f}",
            f"BEST_REFERENCE_LEXICAL_F1={best.dev_f1:.4f}",
            f"BEST_SOURCE={best.source_file}",
        ]
    summary_lines.append(f"WALL_TIME_S={time.monotonic() - started:.1f}")
    (args.run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
