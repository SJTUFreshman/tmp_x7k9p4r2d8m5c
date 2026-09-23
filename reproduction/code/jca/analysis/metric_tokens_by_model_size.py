"""Break down complete LLM output tokens by model size for each baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import count_tokens  # noqa: E402


MODEL_SIZES = ("1.7B", "4B", "8B")
MODEL_PARAMETERS_B = {"1.7B": 1.7, "4B": 4.0, "8B": 8.0}
PARAMETER_SUM_B = sum(MODEL_PARAMETERS_B.values())
AGENT_TO_SIZE = {"A1": "1.7B", "A2": "4B", "A3": "8B"}
GPTSWARM_TYPE_TO_SIZE = {
    "IO": "1.7B",
    "CoT": "4B",
    "Debate": "8B",
    "Aggregator": "8B",
}


@dataclass
class ModelTokenStats:
    baseline: str
    n_problems: int
    n_correct: int
    tokens_1_7b: int
    tokens_4b: int
    tokens_8b: int
    tokens_unknown: int
    total_tokens: int
    mean_tokens_1_7b: float
    mean_tokens_4b: float
    mean_tokens_8b: float
    pct_1_7b: float
    pct_4b: float
    pct_8b: float
    parameter_weighted_tokens: float
    mean_parameter_weighted_tokens: float
    parameter_weighted_tokens_per_correct: float


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def _raw_outputs(record: dict) -> Iterable[str]:
    outputs = record.get("raw_outputs")
    if isinstance(outputs, list):
        return (str(text) for text in outputs if text)
    raw_output = record.get("raw_output")
    return (str(raw_output),) if raw_output else ()


def _agent_size(agent_id: object) -> str:
    text = str(agent_id or "")
    for prefix, size in AGENT_TO_SIZE.items():
        if text == prefix or text.startswith(prefix + "_"):
            return size
    return "unknown"


def _iter_calls(baseline: str, raw: dict) -> Iterator[Tuple[str, str]]:
    if baseline == "mad":
        for turn in (raw.get("mad") or {}).get("turns", []) or []:
            size = _agent_size(turn.get("agent_id"))
            for text in _raw_outputs(turn):
                yield size, text
        return

    if baseline == "agentverse":
        av = raw.get("agentverse") or {}
        # The 8B meta model performs both role recruitment and evaluation.
        for text in _raw_outputs({
            "raw_outputs": av.get("recruit_raw_outputs"),
            "raw_output": av.get("recruit_raw_output"),
        }):
            yield "8B", text
        for iteration in av.get("iterations", []) or []:
            for answer in iteration.get("answers", []) or []:
                size = _agent_size(answer.get("agent_id"))
                for text in _raw_outputs(answer):
                    yield size, text
            for text in _raw_outputs(iteration.get("evaluation") or {}):
                yield "8B", text
        return

    if baseline == "aflow":
        for op in raw.get("op_records", []) or []:
            size = _agent_size(op.get("caller_id"))
            for text in _raw_outputs(op):
                yield size, text
        return

    if baseline == "gptswarm":
        for node in (raw.get("swarm") or {}).get("node_outputs", []) or []:
            size = GPTSWARM_TYPE_TO_SIZE.get(str(node.get("node_type")), "unknown")
            for text in _raw_outputs(node):
                yield size, text
        return

    if baseline == "jca":
        for step in (raw.get("trajectory") or {}).get("steps", []) or []:
            size = _agent_size(step.get("active_agent"))
            for text in _raw_outputs(step):
                yield size, text
        return

    raise ValueError(f"unsupported baseline for model-size accounting: {baseline}")


def compute_stats(baseline: str, path: Path) -> ModelTokenStats:
    baseline = baseline.lower()
    counts: Counter[str] = Counter()
    n_problems = 0
    n_correct = 0
    for raw in _iter_jsonl(path):
        n_problems += 1
        n_correct += int(bool(raw.get("correct")))
        for size, text in _iter_calls(baseline, raw):
            counts[size] += count_tokens(text)

    total = sum(counts.values())
    weighted = sum(
        counts[size] * MODEL_PARAMETERS_B[size] / PARAMETER_SUM_B
        for size in MODEL_SIZES
    )
    denom = total or 1
    n_denom = n_problems or 1
    return ModelTokenStats(
        baseline=baseline,
        n_problems=n_problems,
        n_correct=n_correct,
        tokens_1_7b=counts["1.7B"],
        tokens_4b=counts["4B"],
        tokens_8b=counts["8B"],
        tokens_unknown=counts["unknown"],
        total_tokens=total,
        mean_tokens_1_7b=counts["1.7B"] / n_denom,
        mean_tokens_4b=counts["4B"] / n_denom,
        mean_tokens_8b=counts["8B"] / n_denom,
        pct_1_7b=100 * counts["1.7B"] / denom,
        pct_4b=100 * counts["4B"] / denom,
        pct_8b=100 * counts["8B"] / denom,
        parameter_weighted_tokens=weighted,
        mean_parameter_weighted_tokens=weighted / n_denom,
        parameter_weighted_tokens_per_correct=(
            weighted / n_correct if n_correct else float("inf")),
    )


def _render_table(stats: list[ModelTokenStats]) -> str:
    lines = [
        "| Baseline | N | 1.7B tokens (mean/problem, share) | 4B tokens (mean/problem, share) | 8B tokens (mean/problem, share) | Total | Weighted / problem | Weighted / correct | Unknown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item.baseline} | {item.n_problems} "
            f"| {item.tokens_1_7b:,} ({item.mean_tokens_1_7b:.1f}, {item.pct_1_7b:.1f}%) "
            f"| {item.tokens_4b:,} ({item.mean_tokens_4b:.1f}, {item.pct_4b:.1f}%) "
            f"| {item.tokens_8b:,} ({item.mean_tokens_8b:.1f}, {item.pct_8b:.1f}%) "
            f"| {item.total_tokens:,} "
            f"| {item.mean_parameter_weighted_tokens:.1f} "
            f"| {item.parameter_weighted_tokens_per_correct:.1f} "
            f"| {item.tokens_unknown:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down complete output tokens by 1.7B/4B/8B model.")
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--path", action="append", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if len(args.baseline) != len(args.path):
        parser.error("must supply --baseline and --path the same number of times")

    stats = [compute_stats(name, path)
             for name, path in zip(args.baseline, args.path)]
    print(_render_table(stats))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(item) for item in stats], indent=2),
            encoding="utf-8",
        )
        print(f"[json] saved -> {args.json}")


if __name__ == "__main__":
    main()
