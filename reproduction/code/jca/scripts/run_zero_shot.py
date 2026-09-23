"""Run zero-shot multi-agent inference with local Qwen models.

This script does not train and does not call the judge. It runs the current
agent scheduler with real local models, then grades the final answer by EM/F1.

Usage:
    PYTHONPATH=/data/wangyuheng python scripts/run_zero_shot.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.data import load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.agents import AGENT_IDS  # noqa: E402
from jca.src.inference import (  # noqa: E402
    DEFAULT_LOCAL_MODEL_PATHS,
    GenerationOptions,
    build_qwen_llm_callers,
)
from jca.src.scheduler import Trajectory, run_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot multi-agent Qwen inference on MuSiQue."
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--start", type=int, default=0, help="Start offset in the split.")
    parser.add_argument("--limit", type=int, default=1, help="Number of problems to run.")
    parser.add_argument("--t-max", type=int, default=6, help="Max scheduler turns.")
    parser.add_argument(
        "--start-agent",
        default="A1",
        choices=AGENT_IDS,
        help="Agent that takes the first turn. Default: A1.",
    )
    parser.add_argument(
        "--single-agent",
        action="store_true",
        help="Disable handoffs and run only --start-agent.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode if the tokenizer supports it.",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "transformers", "openai"],
        help="Inference backend. Default: vllm. Use openai for vLLM API servers.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.30,
        help="vLLM GPU memory fraction per engine. Default: 0.30.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running when CUDA is unavailable. This is usually too slow for 3 models.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="auto, bfloat16/bf16, float16/fp16, or float32/fp32.",
    )
    parser.add_argument("--model-a1", type=Path, default=DEFAULT_LOCAL_MODEL_PATHS["A1"])
    parser.add_argument("--model-a2", type=Path, default=DEFAULT_LOCAL_MODEL_PATHS["A2"])
    parser.add_argument("--model-a3", type=Path, default=DEFAULT_LOCAL_MODEL_PATHS["A3"])
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8101/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8102/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8103/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Default: outputs/zero_shot/<split>_<start>_<limit>.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected problems/model paths without loading models.",
    )
    parser.add_argument(
        "--log-raw-chars",
        type=int,
        default=1200,
        help="Max raw model-output characters to print per turn. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit(
            f"No problems selected for split={args.split!r}, start={args.start}, "
            f"limit={args.limit}."
        )

    model_paths = {
        "A1": args.model_a1,
        "A2": args.model_a2,
        "A3": args.model_a3,
    }
    api_base_urls = {
        "A1": args.api_base_a1,
        "A2": args.api_base_a2,
        "A3": args.api_base_a3,
    }
    api_model_names = {
        "A1": args.api_model_a1,
        "A2": args.api_model_a2,
        "A3": args.api_model_a3,
    }
    required_agent_ids = [args.start_agent] if args.single_agent else list(AGENT_IDS)
    for agent_id in required_agent_ids:
        path = model_paths[agent_id]
        if not path.exists():
            raise SystemExit(f"{agent_id} model path does not exist: {path}")

    output_path = args.output or (
        Path("outputs")
        / "zero_shot"
        / f"{args.split}_start{args.start}_n{len(selected)}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Zero-shot MuSiQue inference")
    print(f"  split: {args.split}")
    print(f"  selected: start={args.start}, n={len(selected)}")
    print(f"  t_max: {args.t_max}")
    print(f"  start_agent: {args.start_agent}")
    print(f"  single_agent: {args.single_agent}")
    print(f"  backend: {args.backend}")
    if args.backend == "vllm":
        print(f"  tensor_parallel_size: {args.tensor_parallel_size}")
        print(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
        if args.max_model_len is not None:
            print(f"  max_model_len: {args.max_model_len}")
    if args.backend == "openai":
        for agent_id in required_agent_ids:
            print(
                f"  {agent_id} api: {api_base_urls[agent_id]} "
                f"model={api_model_names[agent_id]}"
            )
    print(f"  output: {output_path}")
    for agent_id in required_agent_ids:
        path = model_paths[agent_id]
        print(f"  {agent_id}: {path}")

    if args.dry_run:
        print("\nDry run only. First selected problem:")
        first = selected[0]
        print(f"  id: {first.id}")
        print(f"  question: {first.question}")
        print(f"  answer: {first.answer}")
        print(f"  paragraphs: {first.n_paragraphs}")
        return

    ensure_runtime_device(args)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    llm_callers = build_qwen_llm_callers(
        model_paths,
        agent_ids=required_agent_ids,
        backend=args.backend,
        generation=generation,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        api_base_urls=api_base_urls,
        api_model_names=api_model_names,
        api_timeout=args.api_timeout,
        api_key=args.api_key,
    )

    records: List[Dict[str, Any]] = []
    run_started = time.monotonic()
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, problem in enumerate(selected, start=1):
            item_started = time.monotonic()
            print(f"\n{format_progress_prefix(idx, len(selected), run_started)} {problem.id}")
            print(f"  question={problem.question}")
            print(f"  gold={problem.answer}")
            traj = run_trajectory(
                problem,
                llm_callers,
                t_max=args.t_max,
                start_agent=args.start_agent,
                allow_handoff=not args.single_agent,
            )
            em, f1 = compute_em_f1(traj.final_answer or "", problem)
            correct = is_correct(traj.final_answer or "", problem)

            record = {
                "problem": {
                    "id": problem.id,
                    "question": problem.question,
                    "answer": problem.answer,
                    "answer_aliases": problem.answer_aliases,
                    "hop": problem.hop,
                },
                "trajectory": trajectory_to_dict(traj),
                "final_answer": traj.final_answer,
                "terminated_by": traj.terminated_by,
                "correct": correct,
                "em": em,
                "f1": f1,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)

            print(
                "  "
                f"terminated={traj.terminated_by} "
                f"agents={traj.active_agents} "
                f"handoffs={traj.n_handoffs} "
                f"em={em:.3f} f1={f1:.3f}"
            )
            if traj.error:
                print(f"  error={traj.error}")
            print(f"  final={traj.final_answer}")
            print_trajectory_log(traj, raw_chars=args.log_raw_chars)
            print(
                "  progress="
                f"{format_progress_line(idx, len(selected), run_started)} "
                f"last={format_duration(time.monotonic() - item_started)}"
            )

    print("\nDone.")
    print(f"  output: {output_path}")
    print(f"  accuracy: {mean(r['em'] for r in records):.3f}")
    print(f"  avg_f1: {mean(r['f1'] for r in records):.3f}")
    print(f"  total_time: {format_duration(time.monotonic() - run_started)}")


def trajectory_to_dict(traj: Trajectory) -> Dict[str, Any]:
    """Serialize a trajectory for JSONL output."""
    return {
        "problem_id": traj.problem_id,
        "steps": [asdict(step) for step in traj.steps],
        "final_answer": traj.final_answer,
        "terminated_by": traj.terminated_by,
        "error": traj.error,
        "active_agents": traj.active_agents,
        "n_handoffs": traj.n_handoffs,
    }


def print_trajectory_log(traj: Trajectory, *, raw_chars: int) -> None:
    """Print detailed per-turn trajectory information to stdout."""
    for step in traj.steps:
        print(f"  turn={step.turn} agent={step.active_agent}")
        if step.reasoning:
            print(f"    reasoning={_one_line(step.reasoning)}")
        if step.handoff_target:
            note = f" note={_one_line(step.handoff_note or '')}" if step.handoff_note else ""
            print(f"    handoff={step.handoff_target}{note}")
        if step.final_answer:
            print(f"    stop={_one_line(step.final_answer)}")
        if raw_chars > 0:
            raw = _one_line(step.raw_output)
            if len(raw) > raw_chars:
                raw = raw[:raw_chars] + "...<truncated>"
            print(f"    raw={raw}")


def _one_line(text: str) -> str:
    """Collapse multiline text for readable logs."""
    return " ".join(str(text).split())


def format_progress_prefix(idx: int, total: int, started_at: float) -> str:
    """Return a compact progress prefix for the current item."""
    percent = 100.0 * (idx - 1) / total if total else 100.0
    return (
        f"[{idx}/{total} {percent:5.1f}% "
        f"elapsed={format_duration(time.monotonic() - started_at)}]"
    )


def format_progress_line(idx: int, total: int, started_at: float) -> str:
    """Return progress stats after completing idx items."""
    elapsed = time.monotonic() - started_at
    avg = elapsed / idx if idx > 0 else 0.0
    remaining = max(total - idx, 0)
    eta = avg * remaining
    percent = 100.0 * idx / total if total else 100.0
    return (
        f"{idx}/{total} {percent:5.1f}% "
        f"elapsed={format_duration(elapsed)} "
        f"avg={format_duration(avg)}/item "
        f"eta={format_duration(eta)}"
    )


def format_duration(seconds: float) -> str:
    """Format seconds as a short human-readable duration."""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"

    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def ensure_runtime_device(args: argparse.Namespace) -> None:
    """Avoid accidentally loading all local models on CPU."""
    if args.allow_cpu or args.device_map == "cpu":
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available, and loading all three Qwen models on CPU is "
            "not recommended. Run on a GPU node, or pass --allow-cpu if you "
            "intentionally want CPU inference."
        )


if __name__ == "__main__":
    main()
