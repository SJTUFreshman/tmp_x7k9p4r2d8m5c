"""Shared GSM-Hard analysis loader.

Token metrics count every persisted response in ``raw_outputs``. This includes
failed logical attempts and transport-level retries, so the reported cost is
the complete observable output cost rather than only the accepted response.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional


_LOCAL_TOKENIZER = Path("/data/wangyuheng/models/Qwen3-1.7B")
DEFAULT_TOKENIZER = os.environ.get(
    "JCA_ANALYSIS_TOKENIZER",
    str(_LOCAL_TOKENIZER)
    if _LOCAL_TOKENIZER.joinpath("tokenizer.json").exists()
    else "Qwen/Qwen2.5-1.5B",
)

SUPPORTED_METHODS = ("mad", "agentverse", "aflow", "gptswarm", "jca")
MODEL_SIZES = ("1.7B", "4B", "8B")
MODEL_PARAMETERS_B = {"1.7B": 1.7, "4B": 4.0, "8B": 8.0}
AGENT_TO_SIZE = {"A1": "1.7B", "A2": "4B", "A3": "8B"}
GPTSWARM_TYPE_TO_SIZE = {
    "IO": "1.7B",
    "CoT": "4B",
    "Debate": "8B",
    "Aggregator": "8B",
}
COST_BASIS = "complete_raw_outputs_including_retries"


@dataclass
class ModelOutput:
    text: str
    model_size: str
    role: str


@dataclass
class ProblemRecord:
    method: str
    problem_id: str
    question: str
    gold_answer: str
    final_answer: str
    em: float
    f1: float
    correct: bool
    outputs: list[ModelOutput] = field(default_factory=list)
    error: Optional[str] = None


@lru_cache(maxsize=4)
def _get_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    if not text:
        return 0
    tokenizer = _get_tokenizer(tokenizer_name)
    if tokenizer is None:
        return max(1, len(text) // 4)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def strip_boxed(text: object) -> str:
    value = str(text or "").strip()
    marker = r"\boxed{"
    if marker not in value:
        return value
    start = value.find(marker) + len(marker)
    depth = 1
    for index in range(start, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[start:index].strip()
    return value


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def _agent_size(agent_id: object) -> str:
    value = str(agent_id or "")
    for agent, size in AGENT_TO_SIZE.items():
        if value == agent or value.startswith(f"{agent}_"):
            return size
    return "unknown"


def _append_output(
    outputs: list[ModelOutput],
    raw_output: object,
    model_size: str,
    role: str,
) -> None:
    if raw_output is not None and str(raw_output):
        outputs.append(ModelOutput(str(raw_output), model_size, role))


def _append_all_outputs(
    outputs: list[ModelOutput],
    container: dict,
    model_size: str,
    role: str,
) -> None:
    """Append every persisted response for one logical model call."""
    raw_outputs = container.get("raw_outputs")
    if isinstance(raw_outputs, list):
        for response_index, response in enumerate(raw_outputs, 1):
            _append_output(
                outputs,
                response,
                model_size,
                f"{role}/response{response_index}",
            )
        return
    _append_output(outputs, container.get("raw_output"), model_size, role)


def _mad_outputs(raw: dict) -> tuple[list[ModelOutput], Optional[str]]:
    record = raw.get("mad") or {}
    outputs: list[ModelOutput] = []
    for turn in record.get("turns", []) or []:
        _append_all_outputs(
            outputs,
            turn,
            _agent_size(turn.get("agent_id")),
            f"round{turn.get('round_idx', 0)}/{turn.get('agent_id', '?')}",
        )
    return outputs, record.get("error")


def _agentverse_outputs(raw: dict) -> tuple[list[ModelOutput], Optional[str]]:
    record = raw.get("agentverse") or {}
    outputs: list[ModelOutput] = []
    meta_size = _agent_size(record.get("meta_agent"))
    if meta_size == "unknown":
        meta_size = "8B"
    _append_all_outputs(
        outputs,
        {
            "raw_outputs": record.get("recruit_raw_outputs"),
            "raw_output": record.get("recruit_raw_output"),
        },
        meta_size,
        "recruiter",
    )
    for iteration in record.get("iterations", []) or []:
        iteration_id = iteration.get("iteration", 0)
        for answer in iteration.get("answers", []) or []:
            _append_all_outputs(
                outputs,
                answer,
                _agent_size(answer.get("agent_id")),
                f"iter{iteration_id}/{answer.get('agent_id', '?')}",
            )
        evaluation = iteration.get("evaluation") or {}
        _append_all_outputs(
            outputs,
            evaluation,
            meta_size,
            f"iter{iteration_id}/evaluator",
        )
    return outputs, record.get("error")


def _gptswarm_outputs(raw: dict) -> tuple[list[ModelOutput], Optional[str]]:
    record = raw.get("swarm") or {}
    node_models = record.get("node_models") or {}
    outputs: list[ModelOutput] = []
    for node in record.get("node_outputs", []) or []:
        node_type = str(node.get("node_type") or "")
        node_name = str(node.get("node_name") or node_type or "node")
        routed_size = _agent_size(node_models.get(node_name))
        if routed_size == "unknown":
            routed_size = GPTSWARM_TYPE_TO_SIZE.get(node_type, "unknown")
        _append_all_outputs(
            outputs,
            node,
            routed_size,
            node_name,
        )
    return outputs, record.get("error")


def _aflow_outputs(raw: dict) -> tuple[list[ModelOutput], Optional[str]]:
    outputs: list[ModelOutput] = []
    for op_index, op in enumerate(raw.get("op_records", []) or []):
        _append_all_outputs(
            outputs,
            op,
            _agent_size(op.get("caller_id")),
            f"op{op_index}/{op.get('op', '?')}/{op.get('caller_id', '?')}",
        )
    return outputs, raw.get("error")


def _jca_outputs(raw: dict) -> tuple[list[ModelOutput], Optional[str]]:
    trajectory = raw.get("trajectory") or {}
    outputs: list[ModelOutput] = []
    attempts = trajectory.get("generation_attempts")
    if isinstance(attempts, list) and attempts:
        for attempt in attempts:
            model_size = _agent_size(attempt.get("agent"))
            _append_all_outputs(
                outputs,
                attempt,
                model_size,
                f"turn{attempt.get('turn', 0)}/{attempt.get('agent', '?')}"
                f"/attempt{attempt.get('attempt', 0)}",
            )
        return outputs, trajectory.get("error")
    for step in trajectory.get("steps", []) or []:
        agent_id = step.get("active_agent")
        _append_all_outputs(
            outputs,
            step,
            _agent_size(agent_id),
            f"turn{step.get('turn', 0)}/{agent_id or '?'}",
        )
    return outputs, trajectory.get("error")


def jca_output_cost_available(path: Path) -> bool:
    for raw in iter_jsonl(path):
        attempts = (raw.get("trajectory") or {}).get("generation_attempts")
        return bool(
            isinstance(attempts, list)
            and attempts
            and all(isinstance(attempt.get("raw_outputs"), list)
                    and attempt["raw_outputs"] for attempt in attempts)
        )
    return False


def load_records(method: str, path: Path) -> list[ProblemRecord]:
    method = method.lower()
    loaders = {
        "mad": _mad_outputs,
        "agentverse": _agentverse_outputs,
        "aflow": _aflow_outputs,
        "gptswarm": _gptswarm_outputs,
        "jca": _jca_outputs,
    }
    if method not in loaders:
        raise ValueError(f"unsupported method {method!r}; expected one of {SUPPORTED_METHODS}")

    records: list[ProblemRecord] = []
    for raw in iter_jsonl(Path(path)):
        problem = raw.get("problem") or {}
        em = float(raw.get("em", 0.0) or 0.0)
        outputs, error = loaders[method](raw)
        records.append(ProblemRecord(
            method=method,
            problem_id=str(problem.get("id") or ""),
            question=str(problem.get("question") or ""),
            gold_answer=str(problem.get("answer") or ""),
            final_answer=strip_boxed(raw.get("final_answer")),
            em=em,
            f1=float(raw.get("f1", 0.0) or 0.0),
            correct=bool(raw["correct"]) if "correct" in raw else em >= 1.0,
            outputs=outputs,
            error=error,
        ))
    return records


def output_tokens(record: ProblemRecord) -> int:
    return sum(count_tokens(output.text) for output in record.outputs)


def all_outputs(records: Iterable[ProblemRecord]) -> Iterator[ModelOutput]:
    for record in records:
        yield from record.outputs
