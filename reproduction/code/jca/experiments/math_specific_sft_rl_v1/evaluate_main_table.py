#!/usr/bin/env python3
"""Run the final MATH CL protocol or summarize its stored evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO


EXPERIMENT_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = EXPERIMENT_ROOT / "vendor"
RUNNER_ROOT = VENDOR_ROOT / "jca/experiments/math_rl_mas_thinking"
SHARD_SHA256 = "3d4b0649a1a4f6198ed22b138fb82b31d7339be65997af4175bf9bdcc6343183"
SCORER_SHA256 = "1c462d6aa67eb79a90ccf04e283133b0b7c4207b2740a6fe28c5c6069afa5908"
EXPECTED_COUNT = 500


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scoring():
    sys.path.insert(0, str(VENDOR_ROOT))
    from jca.src import math_eval

    if sha256(Path(math_eval.__file__)) != SCORER_SHA256:
        raise ValueError("MATH scorer does not match the selected evaluation version")
    module_path = EXPERIMENT_ROOT.parents[1] / "analysis/math/common.py"
    spec = importlib.util.spec_from_file_location("math_main_table_analysis", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load answer projection: {module_path}")
    analysis = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analysis
    spec.loader.exec_module(analysis)
    if analysis.MATH_JCA_TURN_BUDGET != 3:
        raise ValueError("MATH main-table scoring requires a three-turn budget")
    return math_eval, analysis.math_jca_answer_projection


def score_results(input_path: Path, expected_count: int = EXPECTED_COUNT) -> dict[str, Any]:
    scorer, project_answer = load_scoring()
    digest = hashlib.sha256()
    identities: set[str] = set()
    scores = []
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"expected a result object at line {line_number}")
            trajectory = row.get("trajectory")
            problem = row.get("problem")
            if not isinstance(trajectory, dict) or not isinstance(problem, dict):
                raise ValueError(f"missing problem or trajectory at line {line_number}")
            problem_id = trajectory.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id.strip():
                raise ValueError(f"invalid problem ID at line {line_number}")
            if problem_id in identities:
                raise ValueError(f"duplicate problem ID: {problem_id}")
            identities.add(problem_id)
            rollout = trajectory.get("rollout_idx", 0)
            if type(rollout) is not int or rollout != 0:
                raise ValueError(f"expected one rollout per problem: {problem_id}")
            gold = problem.get("gold_answer")
            if not isinstance(gold, str):
                raise ValueError(f"missing gold answer: {problem_id}")
            steps = trajectory.get("steps")
            if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
                raise ValueError(f"invalid trajectory steps: {problem_id}")
            if row.get("math_eval_fingerprint", SCORER_SHA256) != SCORER_SHA256:
                raise ValueError(f"scorer fingerprint mismatch: {problem_id}")
            projection = project_answer(row)
            raw_em = int(scorer.math_answers_equivalent(projection["raw_final_answer"], gold))
            if type(row.get("em")) not in (int, float) or row["em"] != raw_em:
                raise ValueError(f"stored raw EM disagrees with scorer: {problem_id}")
            prediction = projection["projected_answer"]
            soft_f1, _, _ = scorer.math_soft_f1(prediction, gold)
            scores.append({
                "problem_id": problem_id,
                "answer": prediction,
                "gold_answer": gold,
                "answer_source": projection["source"],
                "em": int(scorer.math_answers_equivalent(prediction, gold)),
                "soft_f1": soft_f1,
            })
    if not scores or len(scores) != expected_count:
        raise ValueError(f"expected {expected_count} unique problems, found {len(scores)}")
    correct = sum(score["em"] for score in scores)
    return {
        "source_name": input_path.name,
        "source_sha256": digest.hexdigest(),
        "count": len(scores),
        "correct": correct,
        "em": correct / len(scores),
        "soft_f1": sum(score["soft_f1"] for score in scores) / len(scores),
        "math_eval_version": scorer.MATH_EVAL_VERSION,
        "math_soft_f1_version": scorer.MATH_SOFT_F1_VERSION,
        "scores": scores,
    }


def write_summary(handle: TextIO, metrics: dict[str, Any]) -> None:
    handle.write("MATH CL main-table evaluation summary\n")
    handle.write("Derived from result records; this section is not an execution transcript.\n")
    handle.write("Scoring: retain the final answer within three recorded protocol turns;\n")
    handle.write("if more than three turns were recorded, or the third turn did not stop,\n")
    handle.write("use the first tentative answer. Source records are unchanged.\n")
    for key in ("source_name", "source_sha256", "math_eval_version", "math_soft_f1_version"):
        handle.write(f"{key}: {metrics[key]}\n")
    handle.write(f"scorer_sha256: {SCORER_SHA256}\n")
    handle.write(f"problems: {metrics['count']}\n")
    handle.write(f"correct: {metrics['correct']}\n")
    handle.write(f"EM: {100 * metrics['em']:.2f}%\n")
    handle.write(f"F1: {100 * metrics['soft_f1']:.2f}%\n")
    handle.write(f"soft_f1_fraction: {metrics['soft_f1']:.12f}\n")
    handle.write("\nPer-problem scores under the main-table rule (JSONL):\n")
    for score in metrics["scores"]:
        handle.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + "\n")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output = args.output_root.resolve()
    state = output / "trajectory_state.json"
    data_root = output / "data"
    runner = [sys.executable, "-B", "-u", str(RUNNER_ROOT / "math_role_batched.py")]
    commands = [[
        sys.executable, "-B", "-u", str(RUNNER_ROOT / "build_eval_shard_root.py"),
        "--source-root", str(args.source_root.resolve()),
        "--shard-file", str(args.shard_file.resolve()),
        "--output-root", str(data_root), "--manifest", str(data_root / "manifest.json"),
    ], runner + [
        "init", "--state", str(state), "--data-root", str(data_root),
        "--split", "test", "--start", "0", "--limit", str(EXPECTED_COUNT),
        "--num-rollouts", "1", "--t-max", "3", "--start-agent", "A3",
        "--bootstrap-agent", "A3", "--bootstrap-handoff-target", "A3",
        "--preserve-reasonable-incumbent", "--lock-upstream-answer-agents", "",
        "--start-agent-seed", "43", "--min-agents-before-stop", "1",
        "--no-allow-first-turn-stop", "--generation-seed", "42",
        "--max-new-tokens", "8192", "--protocol-thinking-max-tokens", "2048",
        "--protocol-max-tokens", "1024", "--temperature", "0", "--top-p", "0.95",
        "--no-enable-thinking", "--no-require-thinking", "--api-timeout", "1800",
        "--group-retries", "0", "--no-retry-failed-groups", "--step-retries", "2",
        "--json-transport", "json_schema", "--output-mode", "trajectories",
    ]]
    concurrent = runner + [
        "run-concurrent", "--state", str(state), "--max-inflight", "128",
        "--rebalance-after-advanced", "0", "--max-advanced-per-pass", "0",
        "--api-key", args.api_key,
    ]
    for agent in ("a1", "a2", "a3"):
        concurrent.extend([
            f"--api-base-{agent}", getattr(args, f"api_base_{agent}"),
            f"--api-model-{agent}", getattr(args, f"api_model_{agent}"),
            f"--max-concurrency-{agent}", str(getattr(args, f"max_concurrency_{agent}")),
        ])
    commands.append(concurrent)
    commands.append(runner + [
        "finalize", "--state", str(state), "--output", str(output / "results.jsonl"),
        "--no-resume",
    ])
    return commands


def run_evaluation(args: argparse.Namespace) -> None:
    load_scoring()
    if not args.source_root.is_dir():
        raise ValueError(f"MATH source root is missing: {args.source_root}")
    if sha256(args.shard_file) != SHARD_SHA256:
        raise ValueError("shard file does not match the main-table 500-problem selection")
    for agent in ("a1", "a2", "a3"):
        if getattr(args, f"max_concurrency_{agent}") < 1:
            raise ValueError("endpoint concurrency must be positive")
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError("output root already exists; select a new directory")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(VENDOR_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    log_path = output / "evaluation.log"
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("MATH CL evaluation: fresh state, final rules enabled at initialization.\n")
        handle.write("Bootstrap A3 -> A3; preserve reasonable incumbent; no answer locks.\n")
        handle.write("Three recorded protocol turns; one rollout; no thinking.\n")
        for stage, command in zip(("shard", "init", "evaluate", "finalize"), build_commands(args)):
            print(f"[{stage}] {log_path}", flush=True)
            handle.write(f"\n[{stage}]\n")
            handle.flush()
            completed = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                handle.write(f"\nStage failed with exit code {completed.returncode}: {stage}\n")
                raise RuntimeError(f"{stage} failed; see {log_path}")
        metrics = score_results(output / "results.jsonl")
        handle.write("\n")
        write_summary(handle, metrics)
    print(f"EM={100 * metrics['em']:.2f}% F1={100 * metrics['soft_f1']:.2f}%; {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="Evaluate from fresh state using external final-adapter model endpoints",
        description="Supply the original MATH data/shard and endpoints serving the final A1/A2/A3 adapters.",
    )
    run.add_argument("--source-root", type=Path, required=True, help="External MATH parquet root")
    run.add_argument("--shard-file", type=Path, required=True, help="Original shard_04.jsonl")
    run.add_argument("--output-root", type=Path, required=True, help="New directory for this evaluation")
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    for agent, size, concurrency in (("a1", "1.7B", 32), ("a2", "4B", 32), ("a3", "8B", 64)):
        run.add_argument(f"--api-base-{agent}", required=True, help=f"Qwen3-{size} final adapter endpoint(s)")
        run.add_argument(f"--api-model-{agent}", default=agent.upper(), help="Served final-adapter model name")
        run.add_argument(f"--max-concurrency-{agent}", type=int, default=concurrency)
    summary = subparsers.add_parser("summarize", help="Score stored records without running models")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True, help="New consolidated summary log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_evaluation(args)
    else:
        metrics = score_results(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            write_summary(handle, metrics)
        print(f"EM={100 * metrics['em']:.2f}% F1={100 * metrics['soft_f1']:.2f}%; {args.output}")


if __name__ == "__main__":
    main()
