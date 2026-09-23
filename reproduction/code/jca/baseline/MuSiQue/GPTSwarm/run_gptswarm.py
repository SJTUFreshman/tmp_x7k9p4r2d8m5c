"""GPTSwarm-Fixed baseline for MuSiQue.

Runs a fixed 7-node DAG (IO/CoT layer -> Debate -> IO/CoT layer -> Aggregator)
backed by heterogeneous Qwen3-1.7B/4B/8B executors. See `swarm.py` for
the DAG structure. The default route assigns models by node type; the
balanced route rotates every concrete node across all three models by
problem index.

Expected setup: 3 vLLM OpenAI-compatible servers hosting Qwen3-1.7B,
Qwen3-4B, and Qwen3-8B. Node types are routed by capacity: IO -> A1,
CoT -> A2, and Debate/Aggregator -> A3. Launch them via
`baseline/MuSiQue/GPTSwarm/run_vllm_8gpu.sh`.
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
from typing import Any, Dict, List, Mapping

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
PACKAGE_PARENT = Path(__file__).resolve().parents[4]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.data import MuSiQueProblem, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from swarm import (  # noqa: E402
    SwarmProblem,
    build_fixed_swarm,
    run_swarm_on_problem,
    swarm_record_to_dict,
)


TASK_NAME = "MuSiQue"
MODEL_KEYS = ("A1", "A2", "A3")
NODE_GROUPS = {
    "IO": ("io_0", "io_1"),
    "CoT": ("cot_0", "cot_1", "cot_2"),
    "Debate+Aggregator": ("debate_0", "aggregator"),
}
TYPE_CAPACITY_ROUTING = {
    "io_0": "A1",
    "io_1": "A1",
    "cot_0": "A2",
    "cot_1": "A2",
    "cot_2": "A2",
    "debate_0": "A3",
    "aggregator": "A3",
}
BALANCED_NODE_OFFSETS = {
    "io_0": 0,
    "io_1": 1,
    "cot_0": 0,
    "cot_1": 1,
    "cot_2": 2,
    "debate_0": 1,
    "aggregator": 2,
}


def build_node_routing(policy: str, global_problem_index: int) -> Dict[str, str]:
    """Map every concrete DAG node to A1/A2/A3 deterministically."""
    if policy == "type_capacity":
        return dict(TYPE_CAPACITY_ROUTING)
    if policy != "balanced_per_problem":
        raise ValueError(f"unsupported routing policy: {policy}")
    base = global_problem_index % len(MODEL_KEYS)
    return {
        node_name: MODEL_KEYS[(base + offset) % len(MODEL_KEYS)]
        for node_name, offset in BALANCED_NODE_OFFSETS.items()
    }


def summarize_routing(policy: str, start: int, count: int) -> Dict[str, Dict[str, int]]:
    """Count assignments per semantic node group for audit logging."""
    summary = {
        group: {model: 0 for model in MODEL_KEYS}
        for group in NODE_GROUPS
    }
    for global_index in range(start, start + count):
        route = build_node_routing(policy, global_index)
        for group, node_names in NODE_GROUPS.items():
            for node_name in node_names:
                summary[group][route[node_name]] += 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"GPTSwarm-Fixed baseline on {TASK_NAME}."
    )
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
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--routing-policy",
        choices=("type_capacity", "balanced_per_problem"),
        default="type_capacity",
        help=(
            "type_capacity keeps IO/CoT/Debate+Aggregator on A1/A2/A3; "
            "balanced_per_problem rotates every node across A1/A2/A3."
        ),
    )

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=32,
                        help="Max problems processed concurrently.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-raw-chars", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    output_path = args.output or (
        _HERE / "outputs"
        / f"{args.split}_start{args.start}_n{len(selected)}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    caller_a1 = OpenAIChatLLMCaller(
        args.api_base_a1, args.api_model_a1, generation=generation,
        timeout=args.api_timeout, api_key=args.api_key,
    )
    caller_a2 = OpenAIChatLLMCaller(
        args.api_base_a2, args.api_model_a2, generation=generation,
        timeout=args.api_timeout, api_key=args.api_key,
    )
    caller_a3 = OpenAIChatLLMCaller(
        args.api_base_a3, args.api_model_a3, generation=generation,
        timeout=args.api_timeout, api_key=args.api_key,
    )
    model_callers: Mapping[str, Any] = {
        "A1": caller_a1,
        "A2": caller_a2,
        "A3": caller_a3,
    }

    dag = build_fixed_swarm()

    print(f"GPTSwarm-Fixed {TASK_NAME} inference")
    print(f"  split:           {args.split}")
    print(f"  selected:        start={args.start}, n={len(selected)}")
    print(f"  temperature:     {args.temperature}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  routing_policy:  {args.routing_policy}")
    print(
        "  routing_counts:  "
        + json.dumps(
            summarize_routing(args.routing_policy, args.start, len(selected)),
            sort_keys=True,
        )
    )
    print(f"  output:          {output_path}")
    print(f"  A1 server:       {caller_a1.base_url} model={caller_a1.model_name}")
    print(f"  A2 server:       {caller_a2.base_url} model={caller_a2.model_name}")
    print(f"  A3 server:       {caller_a3.base_url} model={caller_a3.model_name}")
    print(f"  swarm nodes:     {list(dag.nodes.keys())}")
    print(f"  swarm layers:    {dag._layers}")

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

    def _process(problem: MuSiQueProblem, selected_index: int) -> Dict[str, Any]:
        item_started = time.monotonic()
        global_index = args.start + selected_index
        node_models = build_node_routing(args.routing_policy, global_index)
        node_callers = {
            node_name: model_callers[model_key]
            for node_name, model_key in node_models.items()
        }
        swarm_problem = SwarmProblem.from_musique(problem)
        rec = run_swarm_on_problem(dag, swarm_problem, node_callers)
        pred = rec.final_answer or ""
        pred_wrapped = f"\\boxed{{{pred}}}" if pred and "\\boxed" not in pred else pred
        em, f1 = compute_em_f1(pred_wrapped, problem)
        correct = is_correct(pred_wrapped, problem)
        swarm_payload = swarm_record_to_dict(rec)
        swarm_payload["routing_policy"] = args.routing_policy
        swarm_payload["global_problem_index"] = global_index
        swarm_payload["node_models"] = node_models
        return {
            "problem": {
                "id": problem.id,
                "question": problem.question,
                "answer": problem.answer,
                "answer_aliases": problem.answer_aliases,
                "hop": problem.hop,
            },
            "swarm": swarm_payload,
            "final_answer": pred_wrapped or None,
            "correct": correct,
            "em": em,
            "f1": f1,
            "wall_time_s": round(time.monotonic() - item_started, 3),
        }

    with output_path.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = {
                executor.submit(_process, problem, selected_index): problem
                for selected_index, problem in enumerate(selected)
            }
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
        n_err = sum(1 for r in records if r["swarm"].get("error"))
        avg_wall = mean(r["wall_time_s"] for r in records)
        print(f"  errors:      {n_err}/{len(records)}")
        print(f"  avg_wall/prob: {avg_wall:.1f}s")
        # Node-level parse-ok breakdown
        node_stats: Dict[str, Dict[str, int]] = {}
        for r in records:
            for out in r["swarm"].get("node_outputs", []):
                key = f"{out['node_type']}/{out['node_name']}"
                stats = node_stats.setdefault(key, {"total": 0, "ok": 0, "retried": 0})
                stats["total"] += 1
                if out.get("parse_ok"):
                    stats["ok"] += 1
                if out.get("retried"):
                    stats["retried"] += 1
        print("  node parse OK:")
        for key in sorted(node_stats):
            s = node_stats[key]
            print(f"    {key:20s} ok={s['ok']}/{s['total']} retried={s['retried']}")
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
    n_ok = sum(1 for o in record["swarm"].get("node_outputs", []) if o.get("parse_ok"))
    n_tot = len(record["swarm"].get("node_outputs", []))
    err = record["swarm"].get("error")
    print(
        f"[{idx}/{total} {percent:5.1f}% elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}] "
        f"id={record['problem']['id']} em={record['em']:.0f} f1={record['f1']:.2f} "
        f"nodes_ok={n_ok}/{n_tot} err={err is not None} "
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
