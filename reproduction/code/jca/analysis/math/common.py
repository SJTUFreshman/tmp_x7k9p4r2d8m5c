"""Unified loaders and metrics for MATH JCA and baseline trajectories."""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from jca.src.math_eval import compute_math_em
except ImportError:
    from src.math_eval import compute_math_em


METHODS = ("jca", "mad", "agentverse", "gptswarm", "aflow")
MODEL_SIZES = ("1.7B", "4B", "8B", "unknown")
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
MATH_JCA_TURN_BUDGET = 3
MATH_JCA_COST_MAX_TURNS = MATH_JCA_TURN_BUDGET
MATH_JCA_PROJECTION_POLICY = (
    "MATH JCA: 轨迹超过 3 个记录 turn，或恰有 3 个 turn 但未成功 stop 时，"
    "使用首个 step 的 tentative_answer；否则使用原始 final_answer"
)
MATH_JCA_BOOTSTRAP_COPY_REASONING = (
    "The bootstrap answer is copied verbatim for independent verification."
)
_TOKENIZER: Any = None
_MATH_CANONICAL_IDS: dict[tuple[str, str], str] | None = None


@dataclass
class TaskRecord:
    method: str
    problem_id: str
    subject: str
    level: str
    final_answer: str
    raw_final_answer: str
    answer_projection: str
    projection_policy: str
    overlong: bool
    projection_applied: bool
    raw_correct: bool
    gold_answer: str
    correct: bool
    outcome: str
    output_tokens: int
    bootstrap_output_tokens: int
    protocol_output_tokens: int
    thinking_tokens: int
    visible_tokens: int
    calls: int
    bootstrap_calls: int
    protocol_calls: int
    retries: int
    bootstrap_retries: int
    protocol_retries: int
    collaboration_units: int
    raw_collaboration_units: int
    wall_time_s: float | None
    final_chars: int
    final_tokens: int
    model_tokens: dict[str, int]
    request_failures: int
    candidate_count: int
    has_correct_candidate: bool
    raw_candidate_count: int
    has_correct_raw_candidate: bool


def token_count(value: Any) -> int:
    if value is None or value == "":
        return 0
    text = str(value)
    global _TOKENIZER
    if _TOKENIZER is None:
        tokenizer_path = os.environ.get(
            "MATH_ANALYSIS_TOKENIZER", "/data/wangyuheng/models/Qwen3-1.7B"
        )
        if tokenizer_path.lower() in {"fallback", "none"}:
            _TOKENIZER = False
        else:
            try:
                from transformers import AutoTokenizer

                _TOKENIZER = AutoTokenizer.from_pretrained(
                    tokenizer_path, trust_remote_code=True
                )
            except Exception:
                _TOKENIZER = False
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text) // 4)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield row


def resolve_input(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    preferred = (
        source / "trajectories.jsonl",
        source / "eval_results.jsonl",
        source / "math_test_seed43.jsonl",
        source / "05_eval" / "math_test_seed43.jsonl",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(
        candidate
        for candidate in source.rglob("*.jsonl")
        if "journal" not in candidate.name and "progress" not in candidate.name
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"cannot resolve one formal result JSONL under {source}; candidates={candidates}"
    )


def model_size(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    if "a1" in lowered or "1.7b" in lowered:
        return "1.7B"
    if "a2" in lowered or "4b" in lowered:
        return "4B"
    if "a3" in lowered or "8b" in lowered:
        return "8B"
    return "unknown"


def attempt_raw(attempt: dict[str, Any]) -> str:
    raw = attempt.get("raw_output")
    if raw not in (None, ""):
        return str(raw)
    thinking = str(attempt.get("thinking") or "").strip()
    visible = str(attempt.get("visible_output") or "").strip()
    if thinking:
        return f"<think>\n{thinking}\n</think>\n{visible}"
    return visible


def raw_token_parts(raw: str, explicit_thinking: str = "") -> tuple[int, int, int]:
    total = token_count(raw)
    chunks = [match.group(1) for match in THINK_RE.finditer(raw)]
    thinking = sum(token_count(chunk) for chunk in chunks)
    if not chunks and explicit_thinking and raw:
        thinking = min(total, token_count(explicit_thinking))
    return total, thinking, total - thinking


def parse_visible_answer(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        if not match:
            return ""
        try:
            return str(json.loads(f'"{match.group(1)}"')).strip()
        except json.JSONDecodeError:
            return match.group(1).strip()
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict) and payload.get("answer") not in (None, ""):
        return str(payload["answer"]).strip()
    return ""


def component_answer(component: dict[str, Any]) -> str:
    if component.get("answer") not in (None, ""):
        return str(component["answer"]).strip()
    for attempt in reversed(component.get("attempts") or []):
        if attempt.get("parse_ok"):
            answer = parse_visible_answer(attempt.get("visible_output"))
            if answer:
                return answer
    return ""


def _raw_jca_final_answer(row: dict[str, Any]) -> str:
    trajectory = row.get("trajectory") or {}
    for value in (row.get("final_answer"), trajectory.get("final_answer")):
        if value not in (None, ""):
            return str(value).strip()
    return ""


def math_jca_bootstrap_component(
    row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Return the independent bootstrap-solver outputs as one logical call."""
    bootstrap = row.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return None
    raw_outputs = bootstrap.get("raw_outputs")
    if not isinstance(raw_outputs, list):
        raw_output = bootstrap.get("raw_output")
        raw_outputs = [] if raw_output in (None, "") else [raw_output]
    attempts = [
        {
            "raw_output": raw,
            "thinking": "",
            "parse_ok": index == len(raw_outputs) - 1,
        }
        for index, raw in enumerate(raw_outputs)
        if raw not in (None, "")
    ]
    if not attempts:
        return None
    return str(bootstrap.get("agent") or "A3"), attempts


def is_math_jca_bootstrap_copy_step(
    row: dict[str, Any],
    step_index: int,
    step: dict[str, Any],
) -> bool:
    """Identify the forced format-conversion step inserted after bootstrap."""
    if step_index != 0 or not isinstance(row.get("bootstrap"), dict):
        return False
    bootstrap = row["bootstrap"]
    return (
        str(step.get("reasoning") or "").strip()
        == MATH_JCA_BOOTSTRAP_COPY_REASONING
        and str(step.get("tentative_answer") or "").strip()
        == str(bootstrap.get("final_answer") or "").strip()
        and str(step.get("active_agent") or "").strip()
        == str(bootstrap.get("agent") or "A3").strip()
    )


def math_jca_answer_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Return the auditable final-answer projection used by the MATH JCA eval.

    The MATH protocol keeps a result that stops within the three-turn budget.
    A trajectory that exceeds the budget, or exhausts it without a successful
    stop, is scored using its first tentative answer.  This is intentionally
    limited to the MATH JCA result; raw trajectory steps remain available only
    in explicitly labelled audit fields.
    """
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    raw_answer = _raw_jca_final_answer(row)
    first_tentative = ""
    if steps and isinstance(steps[0], dict):
        value = steps[0].get("tentative_answer")
        if value not in (None, ""):
            first_tentative = str(value).strip()
    terminated_by = str(
        row.get("terminated_by") or trajectory.get("terminated_by") or ""
    ).strip().lower()
    exhausted_without_stop = (
        len(steps) == MATH_JCA_TURN_BUDGET and terminated_by != "stop"
    )
    overlong = len(steps) > MATH_JCA_TURN_BUDGET or exhausted_without_stop
    if overlong:
        projected = first_tentative
        source = "first_tentative_overlong" if first_tentative else "first_tentative_missing"
    else:
        projected = raw_answer
        source = "raw_final_answer"
    return {
        "raw_final_answer": raw_answer,
        "first_tentative_answer": first_tentative,
        "projected_answer": projected,
        "steps_count": len(steps),
        "terminated_by": terminated_by,
        "exhausted_without_stop": exhausted_without_stop,
        "overlong": overlong,
        "projection_applied": overlong,
        "source": source,
        "policy": MATH_JCA_PROJECTION_POLICY,
    }


def _math_canonical_problem_id(problem: dict[str, Any]) -> str:
    """Map JCA's subject-local ids to the shared shard ids when possible."""
    question = str(problem.get("question") or "").strip()
    if not question:
        return ""
    global _MATH_CANONICAL_IDS
    if _MATH_CANONICAL_IDS is None:
        _MATH_CANONICAL_IDS = {}
        split_path = (
            Path(__file__).resolve().parents[2]
            / "Math/data/MATH/splits/test_10x500_seed42/shard_04.jsonl"
        )
        if split_path.is_file():
            for row in iter_jsonl(split_path):
                key = (
                    str(row.get("type") or "").strip(),
                    str(row.get("problem") or "").strip(),
                )
                problem_id = str(row.get("problem_id") or "").strip()
                if key[1] and problem_id:
                    _MATH_CANONICAL_IDS[key] = problem_id
    subject = str(problem.get("subject") or problem.get("type") or "").strip()
    return _MATH_CANONICAL_IDS.get((subject, question), "")


def _identity(method: str, row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    if method == "jca":
        problem = row.get("problem") or {}
        trajectory = row.get("trajectory") or {}
        problem_id = _math_canonical_problem_id(problem) or str(
            problem.get("id")
            or trajectory.get("problem_id")
            or row.get("problem_id")
            or ""
        )
        subject = str(problem.get("subject") or row.get("subject") or "Unknown")
        level = str(problem.get("level") or row.get("level") or "Unknown")
        final_answer = str(row.get("final_answer") or trajectory.get("final_answer") or "")
        gold_answer = str(problem.get("gold_answer") or row.get("gold_answer") or "")
        return problem_id, subject, level, final_answer, gold_answer
    final_key = "prediction" if method == "aflow" else "final_answer"
    return (
        str(row.get("problem_id") or ""),
        str(row.get("subject") or "Unknown"),
        str(row.get("level") or "Unknown"),
        str(row.get(final_key) or ""),
        str(row.get("gold_answer") or ""),
    )


def _jca_components(row: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str], int, bool]:
    trajectory = row.get("trajectory") or {}
    components = []
    candidates = []
    for step in trajectory.get("steps") or []:
        raw_outputs = step.get("raw_outputs") or []
        if not raw_outputs and step.get("raw_output") not in (None, ""):
            raw_outputs = [step["raw_output"]]
        attempts = [
            {"raw_output": raw, "thinking": "", "parse_ok": index == len(raw_outputs) - 1}
            for index, raw in enumerate(raw_outputs)
        ]
        components.append((str(step.get("active_agent") or ""), attempts))
        if step.get("tentative_answer") not in (None, ""):
            candidates.append(str(step["tentative_answer"]).strip())
    failed = bool(trajectory.get("error")) or str(
        row.get("terminated_by") or trajectory.get("terminated_by") or ""
    ) not in {"", "stop"}
    return components, candidates, len(trajectory.get("steps") or []), failed


def _mad_components(row: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str], int, bool]:
    turns = row.get("turns") or {}
    values = list(turns.values()) if isinstance(turns, dict) else list(turns)
    components = [(str(turn.get("agent") or turn.get("agent_id") or ""), turn.get("attempts") or []) for turn in values]
    candidates = [str(turn.get("answer")).strip() for turn in values if turn.get("answer") not in (None, "")]
    rounds = {turn.get("round_idx") for turn in values if turn.get("round_idx") is not None}
    failed = bool(row.get("error")) or any(not turn.get("success", True) for turn in values)
    return components, candidates, len(rounds), failed


def _agentverse_components(row: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str], int, bool]:
    components: list[tuple[str, list[dict[str, Any]]]] = []
    candidates = []
    failed = bool(row.get("error"))
    recruit = row.get("recruit")
    if isinstance(recruit, dict):
        components.append(("A3", recruit.get("attempts") or []))
        failed = failed or not recruit.get("success", True)
    iterations = row.get("iterations") or {}
    values = list(iterations.values()) if isinstance(iterations, dict) else list(iterations)
    for iteration in values:
        answers = iteration.get("answers") or {}
        answer_values = list(answers.values()) if isinstance(answers, dict) else list(answers)
        for answer in answer_values:
            agent = str(answer.get("agent") or answer.get("agent_id") or "")
            components.append((agent, answer.get("attempts") or []))
            failed = failed or not answer.get("success", True)
            if answer.get("answer") not in (None, ""):
                candidates.append(str(answer["answer"]).strip())
        evaluation = iteration.get("evaluation")
        if isinstance(evaluation, dict):
            components.append(("A3", evaluation.get("attempts") or []))
            failed = failed or not evaluation.get("success", True)
    return components, candidates, len(values), failed


def _gptswarm_components(row: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str], int, bool]:
    outputs = row.get("node_outputs") or {}
    values = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    components = []
    candidates = []
    failed = bool(row.get("error"))
    for output in values:
        agent = str(output.get("agent") or output.get("api_model") or output.get("model") or "")
        components.append((agent, output.get("attempts") or []))
        failed = failed or not output.get("success", True)
        node_type = str(output.get("node_type") or "").lower()
        node_name = str(output.get("node_name") or "").lower()
        if node_type != "aggregator" and node_name != "aggregator":
            answer = component_answer(output)
            if answer:
                candidates.append(answer)
    return components, candidates, len(values), failed


def _aflow_components(row: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str], int, bool]:
    operations = row.get("op_records") or row.get("op_calls") or []
    components = []
    candidates = []
    failed = bool(row.get("error"))
    for operation in operations:
        agent = str(operation.get("agent") or operation.get("caller_id") or operation.get("model") or "")
        components.append((agent, operation.get("attempts") or []))
        failed = failed or not operation.get("success", operation.get("ok", True))
        if str(operation.get("op") or "").lower() == "solve" and operation.get("success", operation.get("ok", True)):
            answer = component_answer(operation)
            if answer:
                candidates.append(answer)
    return components, candidates, len(operations), failed


COMPONENT_LOADERS = {
    "jca": _jca_components,
    "mad": _mad_components,
    "agentverse": _agentverse_components,
    "gptswarm": _gptswarm_components,
    "aflow": _aflow_components,
}


def task_record(
    method: str,
    row: dict[str, Any],
    cost_max_units: int | None = None,
) -> TaskRecord:
    if cost_max_units is not None and cost_max_units < 0:
        raise ValueError("cost_max_units must be non-negative")
    if method == "jca" and cost_max_units is None:
        cost_max_units = MATH_JCA_COST_MAX_TURNS
    problem_id, subject, level, raw_final_answer, gold_answer = _identity(method, row)
    if not problem_id:
        raise ValueError(f"{method} row lacks problem_id")
    components, candidates, units, protocol_failed = COMPONENT_LOADERS[method](row)
    raw_candidates = candidates
    raw_units = units
    if method == "jca":
        projection = math_jca_answer_projection(row)
        effective_steps = ((row.get("trajectory") or {}).get("steps") or [])[
            :MATH_JCA_TURN_BUDGET
        ]
        candidates = [
            str(step["tentative_answer"]).strip()
            for step in effective_steps
            if isinstance(step, dict)
            and step.get("tentative_answer") not in (None, "")
        ]
        units = min(units, MATH_JCA_TURN_BUDGET)
    else:
        projection = {
            "raw_final_answer": raw_final_answer,
            "projected_answer": raw_final_answer,
            "steps_count": units,
            "overlong": False,
            "projection_applied": False,
            "source": "raw_final_answer",
            "policy": "no MATH JCA projection",
        }
    final_answer = str(projection["projected_answer"] or "").strip()
    raw_final_answer = str(projection["raw_final_answer"] or "").strip()
    cost_components = components if cost_max_units is None else components[:cost_max_units]
    bootstrap_components = []
    if method == "jca":
        bootstrap_component = math_jca_bootstrap_component(row)
        if bootstrap_component is not None:
            bootstrap_components.append(bootstrap_component)
    model_tokens = {size: 0 for size in MODEL_SIZES}
    total_tokens = thinking_tokens = visible_tokens = calls = retries = request_failures = 0
    bootstrap_output_tokens = bootstrap_calls = bootstrap_retries = 0
    protocol_output_tokens = protocol_calls = protocol_retries = 0
    for phase, phase_components in (
        ("bootstrap", bootstrap_components),
        ("protocol", cost_components),
    ):
        for model, attempts in phase_components:
            size = model_size(model)
            phase_calls = len(attempts)
            phase_retries = max(0, phase_calls - 1)
            calls += phase_calls
            retries += phase_retries
            if phase == "bootstrap":
                bootstrap_calls += phase_calls
                bootstrap_retries += phase_retries
            else:
                protocol_calls += phase_calls
                protocol_retries += phase_retries
            for attempt in attempts:
                raw = attempt_raw(attempt)
                total, thinking, visible = raw_token_parts(
                    raw, str(attempt.get("thinking") or "")
                )
                total_tokens += total
                thinking_tokens += thinking
                visible_tokens += visible
                model_tokens[size] += total
                if phase == "bootstrap":
                    bootstrap_output_tokens += total
                else:
                    protocol_output_tokens += total
                request_failures += int(
                    bool(attempt.get("validation_error") or attempt.get("exception"))
                )
    raw_correct = bool(compute_math_em(raw_final_answer, gold_answer))
    correct = bool(compute_math_em(final_answer, gold_answer))
    if correct:
        outcome = "correct"
    elif method == "jca" and projection["projection_applied"]:
        outcome = "turn_budget_exceeded"
    elif not final_answer.strip():
        outcome = "empty_answer"
    elif protocol_failed:
        outcome = "protocol_failure"
    else:
        outcome = "nonempty_em_wrong"
    return TaskRecord(
        method=method,
        problem_id=problem_id,
        subject=subject,
        level=level,
        final_answer=final_answer,
        raw_final_answer=raw_final_answer,
        answer_projection=str(projection["source"]),
        projection_policy=str(projection["policy"]),
        overlong=bool(projection["overlong"]),
        projection_applied=bool(projection["projection_applied"]),
        raw_correct=raw_correct,
        gold_answer=gold_answer,
        correct=correct,
        outcome=outcome,
        output_tokens=total_tokens,
        bootstrap_output_tokens=bootstrap_output_tokens,
        protocol_output_tokens=protocol_output_tokens,
        thinking_tokens=thinking_tokens,
        visible_tokens=visible_tokens,
        calls=calls,
        bootstrap_calls=bootstrap_calls,
        protocol_calls=protocol_calls,
        retries=retries,
        bootstrap_retries=bootstrap_retries,
        protocol_retries=protocol_retries,
        collaboration_units=units,
        raw_collaboration_units=raw_units,
        wall_time_s=float(row["wall_time_s"]) if row.get("wall_time_s") is not None else None,
        final_chars=len(final_answer),
        final_tokens=token_count(final_answer),
        model_tokens=model_tokens,
        request_failures=request_failures,
        candidate_count=len(candidates),
        has_correct_candidate=any(bool(compute_math_em(candidate, gold_answer)) for candidate in candidates),
        raw_candidate_count=len(raw_candidates),
        has_correct_raw_candidate=any(
            bool(compute_math_em(candidate, gold_answer)) for candidate in raw_candidates
        ),
    )


def load_run(
    method: str,
    path: str | Path,
    cost_max_units: int | None = None,
) -> tuple[Path, list[TaskRecord]]:
    normalized = method.lower()
    if normalized not in METHODS:
        raise ValueError(f"unsupported MATH method: {method}")
    source = resolve_input(path)
    records = [
        task_record(normalized, row, cost_max_units=cost_max_units)
        for row in iter_jsonl(source)
    ]
    ids = [record.problem_id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = [problem_id for problem_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate problem ids in {source}: {duplicates[:10]}")
    return source, records


def summarize(records: list[TaskRecord]) -> dict[str, Any]:
    n_tasks = len(records)
    n_correct = sum(record.correct for record in records)

    def mean(field: str) -> float:
        values = [getattr(record, field) for record in records if getattr(record, field) is not None]
        return statistics.mean(values) if values else 0.0

    wall_times = [record.wall_time_s for record in records if record.wall_time_s is not None]

    def grouped(field: str) -> dict[str, dict[str, Any]]:
        output = {}
        for value in sorted({str(getattr(record, field)) for record in records}):
            selected = [record for record in records if str(getattr(record, field)) == value]
            correct = sum(record.correct for record in selected)
            output[value] = {
                "tasks": len(selected),
                "correct": correct,
                "em": correct / len(selected),
            }
        return output

    ratios = [record.output_tokens / record.final_tokens for record in records if record.final_tokens]
    final_chars = [record.final_chars for record in records if record.final_chars]
    final_tokens = [record.final_tokens for record in records if record.final_tokens]
    model_tokens = {
        size: sum(record.model_tokens[size] for record in records) for size in MODEL_SIZES
    }
    parameter_weighted = sum(
        model_tokens[size] * weight
        for size, weight in {"1.7B": 1.7 / 13.7, "4B": 4 / 13.7, "8B": 8 / 13.7}.items()
    )
    candidate_oracle = sum(record.has_correct_candidate for record in records)
    retained = sum(record.has_correct_candidate and record.correct for record in records)
    raw_candidate_oracle = sum(record.has_correct_raw_candidate for record in records)
    raw_candidate_retained = sum(
        record.has_correct_raw_candidate and record.correct for record in records
    )
    subject_level = {}
    for record in records:
        key = f"{record.subject} | {record.level}"
        bucket = subject_level.setdefault(key, {"tasks": 0, "correct": 0})
        bucket["tasks"] += 1
        bucket["correct"] += int(record.correct)
    for bucket in subject_level.values():
        bucket["em"] = bucket["correct"] / bucket["tasks"]
    total_output_tokens = sum(record.output_tokens for record in records)
    raw_correct = sum(record.raw_correct for record in records)
    projection_applied = sum(record.projection_applied for record in records)
    projected_fallback_correct = sum(
        record.projection_applied and record.correct for record in records
    )
    correctness_transitions = Counter(
        (record.raw_correct, record.correct) for record in records
    )
    exactly_budget_records = sum(
        record.raw_collaboration_units == MATH_JCA_TURN_BUDGET for record in records
    )
    exactly_budget_fallback_records = sum(
        record.raw_collaboration_units == MATH_JCA_TURN_BUDGET
        and record.projection_applied
        for record in records
    )
    strictly_longer_projection_correct = sum(
        bool(
            compute_math_em(
                record.final_answer
                if record.raw_collaboration_units > MATH_JCA_TURN_BUDGET
                else record.raw_final_answer,
                record.gold_answer,
            )
        )
        for record in records
    )
    return {
        "n_tasks": n_tasks,
        "n_correct": n_correct,
        "em": n_correct / n_tasks if n_tasks else 0.0,
        "raw_n_correct": raw_correct,
        "raw_em": raw_correct / n_tasks if n_tasks else 0.0,
        "answer_projection": {
            "policy": next(
                (
                    record.projection_policy
                    for record in records
                    if record.projection_applied
                ),
                "no MATH JCA projection",
            ),
            "overlong_records": projection_applied,
            "fallback_records": projection_applied,
            "fallback_correct": projected_fallback_correct,
            "exactly_budget_records": exactly_budget_records,
            "exactly_budget_fallback_records": exactly_budget_fallback_records,
            "strictly_longer_projection_correct": strictly_longer_projection_correct,
            "raw_correct": raw_correct,
            "projected_correct": n_correct,
            "raw_em": raw_correct / n_tasks if n_tasks else 0.0,
            "projected_em": n_correct / n_tasks if n_tasks else 0.0,
            "both_correct": correctness_transitions[(True, True)],
            "wrong_to_correct": correctness_transitions[(False, True)],
            "correct_to_wrong": correctness_transitions[(True, False)],
            "both_wrong": correctness_transitions[(False, False)],
        },
        "n_with_final": sum(bool(record.final_answer.strip()) for record in records),
        "n_with_wall_time": len(wall_times),
        "outcome_counts": dict(Counter(record.outcome for record in records)),
        "total_output_tokens": total_output_tokens,
        "cost_breakdown": {
            "bootstrap_output_tokens": sum(
                record.bootstrap_output_tokens for record in records
            ),
            "protocol_window_output_tokens": sum(
                record.protocol_output_tokens for record in records
            ),
            "bootstrap_raw_responses": sum(record.bootstrap_calls for record in records),
            "protocol_window_raw_responses": sum(
                record.protocol_calls for record in records
            ),
            "bootstrap_response_overhead": sum(
                record.bootstrap_retries for record in records
            ),
            "protocol_window_response_overhead": sum(
                record.protocol_retries for record in records
            ),
        },
        "total_thinking_tokens": sum(record.thinking_tokens for record in records),
        "total_visible_tokens": sum(record.visible_tokens for record in records),
        "tokens_per_correct": total_output_tokens / n_correct if n_correct else None,
        "model_tokens": model_tokens,
        "parameter_weighted_tokens": parameter_weighted,
        "mean_parameter_weighted_tokens": parameter_weighted / n_tasks if n_tasks else 0.0,
        "parameter_weighted_tokens_per_correct": parameter_weighted / n_correct if n_correct else None,
        "means": {
            "output_tokens": mean("output_tokens"),
            "thinking_tokens": mean("thinking_tokens"),
            "visible_tokens": mean("visible_tokens"),
            "calls": mean("calls"),
            "bootstrap_calls": mean("bootstrap_calls"),
            "protocol_calls": mean("protocol_calls"),
            "retries": mean("retries"),
            "bootstrap_retries": mean("bootstrap_retries"),
            "protocol_retries": mean("protocol_retries"),
            "collaboration_units": mean("collaboration_units"),
            "raw_collaboration_units": mean("raw_collaboration_units"),
            "wall_time_s": statistics.mean(wall_times) if wall_times else None,
            "request_failures": mean("request_failures"),
            "final_chars": statistics.mean(final_chars) if final_chars else 0.0,
            "final_chars_median": statistics.median(final_chars) if final_chars else 0.0,
            "final_tokens": statistics.mean(final_tokens) if final_tokens else 0.0,
            "final_tokens_median": statistics.median(final_tokens) if final_tokens else 0.0,
            "output_final_ratio": statistics.mean(ratios) if ratios else 0.0,
            "output_final_ratio_median": statistics.median(ratios) if ratios else 0.0,
            "output_final_ratio_std": statistics.pstdev(ratios) if ratios else 0.0,
        },
        "candidate_funnel": {
            "with_candidate": sum(record.candidate_count > 0 for record in records),
            "candidate_oracle": candidate_oracle,
            "candidate_oracle_em": candidate_oracle / n_tasks if n_tasks else 0.0,
            "correct_candidate_retained": retained,
            "correct_candidate_lost": candidate_oracle - retained,
            "correct_candidate_retention_rate": retained / candidate_oracle if candidate_oracle else None,
            "rescued_without_correct_candidate": sum(record.correct and not record.has_correct_candidate for record in records),
            "full_trajectory_audit": {
                "with_candidate": sum(record.raw_candidate_count > 0 for record in records),
                "candidate_oracle": raw_candidate_oracle,
                "correct_candidate_retained": raw_candidate_retained,
                "correct_candidate_lost": raw_candidate_oracle - raw_candidate_retained,
                "correct_candidate_retention_rate": (
                    raw_candidate_retained / raw_candidate_oracle
                    if raw_candidate_oracle
                    else None
                ),
                "rescued_without_correct_candidate": sum(
                    record.correct and not record.has_correct_raw_candidate
                    for record in records
                ),
            },
        },
        "by_subject": grouped("subject"),
        "by_level": grouped("level"),
        "by_subject_level": dict(sorted(subject_level.items())),
    }


def records_json(records: Iterable[TaskRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
