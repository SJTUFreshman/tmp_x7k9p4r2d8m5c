"""AFlow search runner: run MCTS-lite optimizer to search for a good workflow.

Expects four vLLM OpenAI-compatible servers up:
  - Qwen3-1.7B / 4B / 8B solvers (A1/A2/A3 at ports 8201/8202/8203)
  - Qwen3-14B optimizer (at port 8204)

Outputs go into: PROJECT_ROOT/baseline/MuSiQue/AFlow/search_runs/<RUN_ID>/
  - workflows/round_NN_*.py       (each iteration's candidate source)
  - state.json                    (metadata, executor/optimizer raw outputs; resume-able)
  - summary.txt                   (best node at end)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

# sys.path bootstrap.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.data import load_musique  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from optimizer import (  # noqa: E402
    best_node,
    run_mcts,
)
from workflow import ExecutorCallers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFlow MCTS search on MuSiQue.")
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--dev-start", type=int, default=0)
    parser.add_argument("--dev-size", type=int, default=20,
                        help="How many dev problems each iteration evaluates on.")
    parser.add_argument("--dev-shuffle-seed", type=int, default=20260810)

    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=20260810)

    parser.add_argument("--initial-workflow", type=Path,
                        default=_HERE / "workflows" / "round_00_initial.py")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Where to store search state. Defaults to search_runs/<timestamp>.")

    parser.add_argument("--max-new-tokens-executor", type=int, default=1024)
    parser.add_argument("--max-new-tokens-optimizer", type=int, default=2048)
    parser.add_argument("--temperature-executor", type=float, default=0.7)
    parser.add_argument("--temperature-optimizer", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-base-optimizer", default="http://127.0.0.1:8204/v1")
    parser.add_argument("--api-model-a1", default="A1_base")
    parser.add_argument("--api-model-a2", default="A2_base")
    parser.add_argument("--api-model-a3", default="A3_base")
    parser.add_argument("--api-model-optimizer", default="Optimizer_14B")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--max-concurrency", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.dev_size < 1:
        raise SystemExit("--dev-size must be >= 1")

    # Load & subsample dev.
    all_problems = load_musique(args.dev_split, data_dir=args.data_dir)
    if not all_problems:
        raise SystemExit("No MuSiQue problems loaded.")
    rng = random.Random(args.dev_shuffle_seed)
    problems = list(all_problems[args.dev_start:])
    rng.shuffle(problems)
    dev = problems[: args.dev_size]

    # Callers.
    executor_generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens_executor,
        temperature=args.temperature_executor,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    a1 = OpenAIChatLLMCaller(args.api_base_a1, args.api_model_a1, generation=executor_generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    a2 = OpenAIChatLLMCaller(args.api_base_a2, args.api_model_a2, generation=executor_generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    a3 = OpenAIChatLLMCaller(args.api_base_a3, args.api_model_a3, generation=executor_generation,
                             timeout=args.api_timeout, api_key=args.api_key)
    executor_callers = ExecutorCallers([a1, a2, a3], a3, ["A1_1.7B", "A2_4B", "A3_8B"], "A3_8B")
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
    )

    # Run dir.
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = _HERE / "search_runs" / f"{stamp}_mcts_iters{args.max_iterations}_dev{args.dev_size}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("AFlow MCTS search")
    print(f"  run_dir:         {run_dir}")
    print(f"  initial:         {args.initial_workflow}")
    print(f"  dev_split:       {args.dev_split}")
    print(f"  dev_size:        {args.dev_size} (shuffle seed {args.dev_shuffle_seed})")
    print(f"  max_iterations:  {args.max_iterations}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  enable_thinking: {args.enable_thinking}")
    print("  solve pool:      A1_1.7B, A2_4B, A3_8B (round-robin)")
    print(f"  judge:           {a3.base_url} model={a3.model_name}")
    print(f"  optimizer:       {optimizer_caller.base_url} model={optimizer_caller.model_name}")

    started = time.monotonic()
    nodes = run_mcts(
        root=run_dir,
        initial_source_path=args.initial_workflow,
        dev_problems=dev,
        executor_callers=executor_callers,
        optimizer_caller=optimizer_caller,
        max_iterations=args.max_iterations,
        max_concurrency=args.max_concurrency,
        rng_seed=args.rng_seed,
        optimizer_caller_id=args.api_model_optimizer,
    )

    best = best_node(nodes)
    print("\n================ Search Complete ================")
    print(f"  total_nodes:   {len(nodes)}  parse_ok={sum(1 for n in nodes if n.parse_ok)}")
    if best is not None:
        print(f"  best node:     round={best.round_id} name={best.name} "
              f"EM={best.dev_em:.3f} F1={best.dev_f1:.3f}")
        print(f"  best source:   {best.source_file}")

    summary_lines = [
        f"RUN_DIR={run_dir}",
        f"MAX_ITERATIONS={args.max_iterations}",
        f"DEV_SIZE={args.dev_size}",
        f"TOTAL_NODES={len(nodes)}",
    ]
    if best is not None:
        summary_lines += [
            f"BEST_ROUND={best.round_id}",
            f"BEST_EM={best.dev_em:.4f}",
            f"BEST_F1={best.dev_f1:.4f}",
            f"BEST_SOURCE={best.source_file}",
        ]
    summary_lines.append(f"WALL_TIME_S={time.monotonic() - started:.1f}")
    (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
