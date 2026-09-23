#!/usr/bin/env python3
"""Fail-closed validation for MATH sampling, scoring, preparation, and eval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.math_eval import (  # noqa: E402
    AGENT_IDS,
    MATH_EVAL_VERSION,
    compute_math_em,
    extract_last_boxed,
    load_math_problems,
    math_answers_equivalent,
)
from jca.src.protocol_json import validate_protocol_object  # noqa: E402
from jca.experiments.math_rl_mas_thinking.sampling_coverage import validate_sampling_coverage


JUDGE_SCHEMA = "compact_canonical_math_v1"
JUDGE_STATUS = "scored"
DISCRETE_SCORES = frozenset({-1.0, -0.5, 0.0, 0.5, 1.0})
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
DIRECT_SOLVER_SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. Solve the problem carefully. "
    "Return a concise derivation and end the visible response with exactly one "
    "final answer in \\boxed{...}. A response without a final \\boxed{...} answer "
    "is unusable and will be retried."
)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "agent", "status", "system_prompt", "thinking", "visible_output",
        "final_answer", "raw_output", "raw_outputs", "attempts", "error",
    }
)
DEGENERATE_RECOVERY_AGENT = "A3"


def _is_audited_degenerate_recovery(
    item: Mapping[str, Any],
    *,
    turn: int,
    scheduled_agent: Any,
    actual_agent: Any,
) -> bool:
    if (
        actual_agent != DEGENERATE_RECOVERY_AGENT
        or scheduled_agent == actual_agent
        or scheduled_agent not in AGENT_IDS
    ):
        return False
    rejected = any(
        record.get("turn") == turn
        and record.get("agent") == scheduled_agent
        and str(record.get("reason") or "").startswith("degenerate repetition")
        for record in item.get("generation_rejections", [])
        if isinstance(record, Mapping)
    )
    recovered = any(
        record.get("turn") == turn
        and record.get("agent") == actual_agent
        and record.get("accepted") is True
        for record in item.get("generation_attempts", [])
        if isinstance(record, Mapping)
    )
    return rejected and recovered


def math_eval_fingerprint() -> str:
    """Return a content fingerprint for the active canonical evaluator."""
    module_path = REPO_ROOT / "src" / "math_eval.py"
    return hashlib.sha256(module_path.read_bytes()).hexdigest()


MATH_EVAL_FINGERPRINT = math_eval_fingerprint()


def evaluator_fingerprint() -> str:
    """Return a fingerprint for code that can change eval semantics."""
    paths = (
        REPO_ROOT / "src" / "math_eval.py",
        REPO_ROOT / "src" / "protocol_json.py",
        REPO_ROOT / "experiments" / "math_rl_mas_thinking" / "math_rollout.py",
        REPO_ROOT / "experiments" / "math_rl_mas_thinking" / "math_role_batched.py",
        REPO_ROOT / "experiments" / "math_rl_mas_thinking" / "score_math_rollouts.py",
        REPO_ROOT / "experiments" / "math_rl_mas_thinking" / "prepare_math_v13_data.py",
        Path(__file__).resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


EVALUATOR_FINGERPRINT = evaluator_fingerprint()


def _normalize_subjects(subjects: Any) -> list[str]:
    """Normalize a subject filter without silently changing its meaning."""
    if subjects is None or subjects == "":
        return []
    if isinstance(subjects, str):
        values = subjects.split(",")
    elif isinstance(subjects, (list, tuple, set)):
        values = list(subjects)
    else:
        raise ValueError("subjects must be a comma-separated string or a list")
    normalized = [value.strip() if isinstance(value, str) else value for value in values]
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("subjects must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("subjects must not contain duplicates")
    return sorted(normalized)


def _load_eval_problems(data_root: Path, split: str, subjects: Any = None):
    normalized = _normalize_subjects(subjects)
    if normalized:
        return load_math_problems(data_root, split=split, subjects=normalized)
    return load_math_problems(data_root, split=split)


_MUTABLE_SCORE_FIELDS = frozenset(
    {
        "collaboration_judge_v3",
        "collaboration_judge_v4",
        "deterministic_state_v3",
        "judge_state_disagreements_v3",
        "collaboration_judge_model_v3",
        "collaboration_judge_model_v4",
        "collaboration_judge_status_v3",
        "collaboration_judge_status_v4",
        "collaboration_judge_schema_v4",
        "judge_thinking_enabled",
        "math_eval_version",
        "math_eval_fingerprint",
        "evaluator_fingerprint",
        "em",
        "thinking_enabled",
        "enable_thinking",
    }
)


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    problem_id = row.get("problem_id")
    rollout_idx = row.get("rollout_idx")
    turn = row.get("turn")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("invalid problem_id")
    if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx < 0:
        raise ValueError(f"invalid rollout_idx for {problem_id}")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
        raise ValueError(f"invalid turn for {problem_id}/{rollout_idx}")
    return problem_id, rollout_idx, turn


def _json_projection(row: Mapping[str, Any], *, mutable: Iterable[str]) -> str:
    projection = {key: value for key, value in row.items() if key not in set(mutable)}
    try:
        return json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("row contains non-JSON or non-finite values") from exc


def _answer_text(value: Any, field: str = "final_answer") -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value.strip()


def _validate_bootstrap_payload(
    payload: Any,
    *,
    expected_agent: Optional[str],
    expected_mode: bool,
    expected_required: bool,
    allow_pending: bool,
) -> None:
    if not isinstance(payload, dict) or set(payload) != _BOOTSTRAP_FIELDS:
        raise ValueError("direct-solver bootstrap has invalid fields")
    agent = payload.get("agent")
    if agent not in AGENT_IDS or (expected_agent is not None and agent != expected_agent):
        raise ValueError("direct-solver bootstrap agent disagrees")
    status = payload.get("status")
    allowed_statuses = {"done", "failed"} | ({"pending"} if allow_pending else set())
    if status not in allowed_statuses:
        raise ValueError("direct-solver bootstrap has invalid status")
    if payload.get("system_prompt") != DIRECT_SOLVER_SYSTEM_PROMPT:
        raise ValueError("direct-solver bootstrap prompt disagrees")
    for field in ("thinking", "visible_output", "raw_output"):
        if not isinstance(payload.get(field), str):
            raise ValueError(f"direct-solver bootstrap {field} is invalid")
    if not isinstance(payload.get("raw_outputs"), list) or not all(
        isinstance(value, str) for value in payload["raw_outputs"]
    ):
        raise ValueError("direct-solver bootstrap raw_outputs is invalid")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("direct-solver bootstrap attempts are invalid")
    for index, attempt in enumerate(attempts):
        fields = {
            "attempt", "accepted", "reason", "raw_output", "raw_outputs", "visible_raw_output"
        }
        if not isinstance(attempt, dict) or set(attempt) != fields:
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid fields")
        if isinstance(attempt.get("attempt"), bool) or attempt.get("attempt") != index + 1:
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid number")
        if not isinstance(attempt.get("accepted"), bool):
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid acceptance")
        if attempt.get("reason") is not None and not isinstance(attempt.get("reason"), str):
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid reason")
        if not isinstance(attempt.get("raw_output"), str) or not isinstance(
            attempt.get("visible_raw_output"), str
        ):
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid output")
        if not isinstance(attempt.get("raw_outputs"), list) or not all(
            isinstance(value, str) for value in attempt["raw_outputs"]
        ):
            raise ValueError(f"direct-solver bootstrap attempt {index} has invalid raw outputs")
    final_answer = payload.get("final_answer")
    error = payload.get("error")
    if status == "pending":
        if attempts or final_answer is not None or error is not None:
            raise ValueError("pending direct-solver bootstrap has terminal fields")
        if any(payload.get(field) for field in ("thinking", "visible_output", "raw_output", "raw_outputs")):
            raise ValueError("pending direct-solver bootstrap has generated output")
    elif status == "done":
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError("completed direct-solver bootstrap has no final answer")
        if extract_last_boxed(payload["visible_output"]) != final_answer:
            raise ValueError("direct-solver bootstrap boxed answer disagrees")
        if not expected_mode and payload["thinking"].strip():
            raise ValueError("non-thinking direct-solver bootstrap contains thinking")
        if expected_mode and expected_required and not payload["thinking"].strip():
            raise ValueError("direct-solver bootstrap lacks required thinking")
        if error is not None or not attempts or not attempts[-1]["accepted"]:
            raise ValueError("completed direct-solver bootstrap attempts disagree")
    else:
        if final_answer is not None or not isinstance(error, str) or not error.strip():
            raise ValueError("failed direct-solver bootstrap terminal fields disagree")


def _parse_visible_protocol(row: Mapping[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response is missing or empty")
    visible = THINK_RE.sub("", response).strip()
    parsed = validate_protocol_object(visible, active_agent=row.get("agent_id"))
    explicit = row.get("protocol_response")
    if explicit is not None:
        if not isinstance(explicit, str):
            raise ValueError("protocol_response must be a string")
        explicit_parsed = validate_protocol_object(explicit, active_agent=row.get("agent_id"))
        if explicit_parsed != parsed:
            raise ValueError("protocol_response differs from visible response")
    return parsed


def _validate_protocol_terminal(row: Mapping[str, Any], parsed: Mapping[str, Any]) -> None:
    action = parsed.get("action")
    if action not in {"handoff", "confirm_stop"}:
        raise ValueError(f"invalid protocol action: {action!r}")
    if "action" in row and row.get("action") not in {None, action}:
        raise ValueError("row action disagrees with protocol action")
    final_answer = _answer_text(row.get("final_answer"))
    terminated = row.get("terminated_by")
    if action == "confirm_stop":
        if terminated != "stop":
            raise ValueError("confirm_stop row does not have stop termination")
        confirmed = _answer_text(parsed.get("confirmed_answer"), "confirmed_answer")
        if not final_answer:
            raise ValueError("stop row has no final_answer")
        if not confirmed or not math_answers_equivalent(final_answer, confirmed):
            raise ValueError("final_answer disagrees with confirmed_answer")
    elif terminated == "truncated":
        if not final_answer:
            raise ValueError("truncated row has no final_answer")
    elif terminated in {"rejected_quality", "exception"} and final_answer:
        raise ValueError("failed row has a terminal final_answer")


def _validate_policy(row: Mapping[str, Any], expected_mode: Optional[bool] = None) -> bool:
    values = []
    for field in ("thinking_enabled", "enable_thinking"):
        value = row.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} is missing or not boolean")
        values.append(value)
    if values[0] != values[1]:
        raise ValueError("thinking aliases disagree")
    mode = values[0]
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("thinking mode disagrees with the run")
    response = row.get("response")
    if not isinstance(response, str):
        raise ValueError("response is missing")
    matches = list(THINK_RE.finditer(response))
    thinking = _answer_text(row.get("thinking"), "thinking")
    retained = row.get("thinking_retained_in_response")
    if mode:
        if not matches or not any(match.group(1).strip() for match in matches):
            raise ValueError("thinking-enabled row has an empty thinking trace")
        if retained is not True:
            raise ValueError("thinking provenance is missing")
    else:
        if matches or re.search(r"</?think\b", response, re.IGNORECASE):
            raise ValueError("non-thinking row contains thinking tags")
        if thinking or retained is not False:
            raise ValueError("non-thinking thinking provenance is inconsistent")
    return mode


def _validate_common_version(row: Mapping[str, Any], *, require_fingerprint: bool) -> None:
    if row.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("stale math evaluator version")
    if require_fingerprint and row.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("stale or missing math evaluator fingerprint")
    if require_fingerprint and row.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("stale or missing runtime evaluator fingerprint")


def _validate_embedded_fingerprints(payload: Mapping[str, Any], *, location: str) -> None:
    version = payload.get("math_eval_version")
    if version is not None and version != MATH_EVAL_VERSION:
        raise ValueError(f"{location} has stale math evaluator version")
    math_fingerprint = payload.get("math_eval_fingerprint")
    if math_fingerprint is not None and math_fingerprint != MATH_EVAL_FINGERPRINT:
        raise ValueError(f"{location} has stale math evaluator fingerprint")
    runtime_fingerprint = payload.get("evaluator_fingerprint")
    if runtime_fingerprint is not None and runtime_fingerprint != EVALUATOR_FINGERPRINT:
        raise ValueError(f"{location} has stale runtime evaluator fingerprint")


def _validate_binary_em(value: Any, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("em is missing or not numeric")
    try:
        actual = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("em is not finite") from exc
    if not math.isfinite(actual) or actual not in {0.0, 1.0}:
        raise ValueError("em must be binary")
    if actual != float(expected):
        raise ValueError(f"em disagrees with {MATH_EVAL_VERSION}")


def _validate_judge(row: Mapping[str, Any], *, require_fingerprint: bool = True) -> None:
    if row.get("collaboration_judge_status_v3") != JUDGE_STATUS or row.get("collaboration_judge_status_v4") != JUDGE_STATUS:
        raise ValueError("row is not fully judged")
    if row.get("collaboration_judge_schema_v4") != JUDGE_SCHEMA:
        raise ValueError("unexpected judge schema")
    if row.get("judge_thinking_enabled") is not True:
        raise ValueError("judge thinking contract is missing")
    state = row.get("deterministic_state_v3")
    if not isinstance(state, dict) or state.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("stale deterministic state")
    if state.get("math_eval_fingerprint") not in {None, MATH_EVAL_FINGERPRINT}:
        raise ValueError("stale deterministic-state fingerprint")
    if require_fingerprint and state.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("missing deterministic-state fingerprint")
    if state.get("evaluator_fingerprint") not in {None, EVALUATOR_FINGERPRINT}:
        raise ValueError("stale deterministic-state runtime fingerprint")
    if require_fingerprint and state.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("missing deterministic-state runtime fingerprint")
    if not isinstance(row.get("judge_state_disagreements_v3"), list):
        raise ValueError("invalid judge disagreement ledger")
    scores = []
    for field in ("collaboration_judge_v3", "collaboration_judge_v4"):
        payload = row.get(field)
        if not isinstance(payload, dict) or payload.get("schema") != JUDGE_SCHEMA:
            raise ValueError(f"missing or invalid {field}")
        if payload.get("math_eval_version") != MATH_EVAL_VERSION:
            raise ValueError(f"stale {field}")
        if payload.get("math_eval_fingerprint") not in {None, MATH_EVAL_FINGERPRINT}:
            raise ValueError(f"stale {field} fingerprint")
        if require_fingerprint and payload.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
            raise ValueError(f"missing {field} fingerprint")
        if payload.get("evaluator_fingerprint") not in {None, EVALUATOR_FINGERPRINT}:
            raise ValueError(f"stale {field} runtime fingerprint")
        if require_fingerprint and payload.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
            raise ValueError(f"missing {field} runtime fingerprint")
        value = payload.get("process_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid {field} process_score")
        value = float(value)
        if not math.isfinite(value) or value not in DISCRETE_SCORES:
            raise ValueError(f"{field} process_score is not discrete")
        if payload.get("score_scale") not in {None, "discrete[-1,-0.5,0,0.5,1]"}:
            raise ValueError(f"invalid {field} score_scale")
        if "judge_score" in payload and payload.get("judge_score") != value:
            raise ValueError(f"{field} judge_score disagrees with process_score")
        scores.append(value)
    if scores[0] != scores[1]:
        raise ValueError("v3/v4 judge scores disagree")
    for field in ("collaboration_judge_model_v3", "collaboration_judge_model_v4"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"missing {field}")


def _group_terminal(rows: list[Mapping[str, Any]]) -> tuple[str, str]:
    if not rows:
        raise ValueError("empty trajectory group")
    ordered = sorted(rows, key=lambda row: int(row["turn"]))
    turns = [int(row["turn"]) for row in ordered]
    if turns != list(range(len(turns))):
        raise ValueError("trajectory turns are not contiguous")
    statuses = {row.get("terminated_by") for row in ordered}
    if len(statuses) != 1:
        raise ValueError("trajectory termination status differs across turns")
    status = str(next(iter(statuses)))
    answers = [_answer_text(row.get("final_answer")) for row in ordered]
    terminal = answers[-1]
    for answer in answers:
        if bool(answer) != bool(terminal) or (answer and not math_answers_equivalent(answer, terminal)):
            raise ValueError("trajectory rows disagree on final_answer")
    parsed_steps = []
    for row in ordered:
        parsed = _parse_visible_protocol(row)
        _validate_protocol_terminal(row, parsed)
        parsed_steps.append(parsed)
    for index, parsed in enumerate(parsed_steps[:-1]):
        if parsed.get("action") == "confirm_stop":
            raise ValueError("trajectory contains a turn after confirm_stop")
        if parsed.get("action") == "handoff":
            target = parsed.get("handoff_target")
            next_agent = ordered[index + 1].get("agent_id")
            if target != next_agent:
                raise ValueError("handoff target does not match next active agent")
    if status == "stop":
        if not terminal or parsed_steps[-1].get("action") != "confirm_stop":
            raise ValueError("stop trajectory has no terminal answer")
        confirmed = _answer_text(parsed_steps[-1].get("confirmed_answer"), "confirmed_answer")
        tentative = _answer_text(parsed_steps[-1].get("tentative_answer"), "tentative_answer")
        if not confirmed or not math_answers_equivalent(terminal, confirmed):
            raise ValueError("stop trajectory final answer disagrees with confirmation")
        if not tentative or not math_answers_equivalent(terminal, tentative):
            raise ValueError("stop trajectory final answer disagrees with tentative answer")
    elif status == "truncated":
        tentative = _answer_text(
            parsed_steps[-1].get("tentative_answer"), "tentative_answer"
        ) if parsed_steps else ""
        if not terminal or terminal != tentative:
            raise ValueError(
                "truncated trajectory final answer is not the last tentative answer"
            )
    elif terminal:
        raise ValueError("failed trajectory has terminal answer")
    return terminal, status


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_keys,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            yield line_no, row


def _problem_map(data_root: Path, split: str, subjects: Any = None) -> dict[str, Any]:
    problems = _load_eval_problems(data_root, split, subjects)
    return {str(problem.problem_id): problem for problem in problems}


def _validate_optional_problem_fields(
    payload: Mapping[str, Any],
    problem: Any,
    *,
    location: str,
) -> None:
    expected_fields = {
        "problem_id": problem.problem_id,
        "id": problem.problem_id,
        "subject": problem.subject,
        "level": problem.level,
        "prompt": problem.prompt,
        "question": problem.prompt,
        "solution": problem.solution,
        "gold_answer": problem.gold_answer,
    }
    for field, expected in expected_fields.items():
        if field not in payload:
            continue
        actual = payload.get(field)
        if field == "gold_answer":
            matches = isinstance(actual, str) and math_answers_equivalent(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"{location} {field} disagrees with canonical data")


def validate_scored_output(
    source_path: Path,
    judged_path: Path,
    stats_path: Path,
    data_root: Path,
    expected_rollouts: int,
    *,
    require_fingerprint: bool = True,
    failure_ledger: Optional[Path] = None,
    sample_state: Optional[Path] = None,
) -> dict[str, int | str]:
    """Validate a complete score artifact against source and canonical gold."""
    problems = _problem_map(data_root, "train")
    source_meta: dict[tuple[str, int, int], tuple[str, bool]] = {}
    source_groups: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    source_modes: set[bool] = set()
    for line_no, row in _iter_jsonl(source_path):
        identity = _identity(row)
        problem_id, rollout_idx, turn = identity
        if rollout_idx >= expected_rollouts:
            raise ValueError(f"rollout_idx is outside configured range at line {line_no}")
        if identity in source_meta:
            raise ValueError(f"duplicate source identity at line {line_no}")
        if problem_id not in problems:
            raise ValueError(f"unknown train problem: {problem_id}")
        _validate_optional_problem_fields(row, problems[problem_id], location=f"source line {line_no}")
        _validate_common_version(row, require_fingerprint=require_fingerprint)
        mode = _validate_policy(row)
        parsed = _parse_visible_protocol(row)
        _validate_protocol_terminal(row, parsed)
        source_modes.add(mode)
        source_meta[identity] = (_json_projection(row, mutable=_MUTABLE_SCORE_FIELDS), mode)
        source_groups[(problem_id, rollout_idx)][turn] = row
    if not source_meta:
        raise ValueError("source is empty")
    if len(source_modes) != 1:
        raise ValueError("source mixes thinking modes")
    for key, rows in source_groups.items():
        _group_terminal(list(rows.values()))
    source_rollout_counts = Counter(problem_id for problem_id, _ in source_groups)
    coverage = None
    if failure_ledger is not None or sample_state is not None:
        coverage = validate_sampling_coverage(
            source_groups,
            expected_rollouts=expected_rollouts,
            data_root=data_root,
            failure_ledger=failure_ledger,
            sample_state=sample_state,
        )
    elif any(count != expected_rollouts for count in source_rollout_counts.values()):
        raise ValueError("source does not contain configured rollout count per problem")

    judged_meta: set[tuple[str, int, int]] = set()
    judged_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for line_no, row in _iter_jsonl(judged_path):
        identity = _identity(row)
        if identity in judged_meta:
            raise ValueError(f"duplicate judged identity at line {line_no}")
        expected = source_meta.get(identity)
        if expected is None:
            raise ValueError(f"judged identity is not present in source at line {line_no}")
        _validate_common_version(row, require_fingerprint=require_fingerprint)
        mode = _validate_policy(row, expected_mode=next(iter(source_modes)))
        _validate_judge(row, require_fingerprint=require_fingerprint)
        _validate_optional_problem_fields(row, problems[identity[0]], location=f"judged line {line_no}")
        parsed = _parse_visible_protocol(row)
        _validate_protocol_terminal(row, parsed)
        if _json_projection(row, mutable=_MUTABLE_SCORE_FIELDS) != expected[0]:
            raise ValueError(f"judged row mutates source fields at line {line_no}")
        judged_meta.add(identity)
        judged_groups[identity[:2]].append(row)
        judged_problem = problems[identity[0]]
        terminal = _answer_text(row.get("final_answer"))
        _validate_binary_em(row.get("em"), compute_math_em(terminal, judged_problem.gold_answer))
    if judged_meta != set(source_meta):
        raise ValueError("judged identities do not exactly match source")
    for key, rows in judged_groups.items():
        terminal, _ = _group_terminal(rows)
        problem = problems[key[0]]
        expected_em = compute_math_em(terminal, problem.gold_answer)
        for row in rows:
            _validate_binary_em(row.get("em"), expected_em)
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read score stats: {stats_path}") from exc
    expected_stats = {
        "groups": len(judged_groups),
        "rows": len(judged_meta),
        "problems": len(source_rollout_counts),
    }
    if coverage is not None:
        expected_stats.update(coverage)
    for key, value in expected_stats.items():
        if stats.get(key) != value:
            raise ValueError(f"score stats {key} disagrees")
    if stats.get("schema") != JUDGE_SCHEMA or stats.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("score stats schema/version is stale")
    if require_fingerprint and stats.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("score stats fingerprint is stale or missing")
    if require_fingerprint and stats.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("score stats runtime fingerprint is stale or missing")
    return {**expected_stats, "math_eval_version": MATH_EVAL_VERSION}


def _validate_prepared_row(
    row: Mapping[str, Any],
    problem_map: Mapping[str, Any],
    expected_mode: bool,
    *,
    require_fingerprint: bool,
) -> None:
    identity = _identity(row)
    problem = problem_map.get(identity[0])
    if problem is None:
        raise ValueError(f"unknown MATH train problem: {identity[0]}")
    _validate_optional_problem_fields(row, problem, location=f"prepared row {identity}")
    _validate_common_version(row, require_fingerprint=require_fingerprint)
    _validate_policy(row, expected_mode=expected_mode)
    parsed = _parse_visible_protocol(row)
    _validate_protocol_terminal(row, parsed)
    _validate_judge(row, require_fingerprint=require_fingerprint)
    terminal = _answer_text(row.get("final_answer"))
    _validate_binary_em(row.get("em"), compute_math_em(terminal, problem.gold_answer))
    if not isinstance(row.get("reward"), (int, float)) or isinstance(row.get("reward"), bool) or not math.isfinite(float(row["reward"])):
        raise ValueError("prepared row has invalid reward")


def validate_prepared_outputs(
    train_path: Path,
    holdout_path: Path,
    stats_path: Path,
    manifest_path: Path,
    data_root: Path,
    *,
    expected_sample_mode: bool,
    expected_train_mode: bool,
    expected_eval_mode: bool,
    expected_eval_required: bool,
    require_fingerprint: bool = True,
) -> dict[str, int | str]:
    problems = _problem_map(data_root, "train")
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    split_paths = {"train": train_path, "holdout": holdout_path}
    prepared_groups: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {
        "train": defaultdict(list),
        "holdout": defaultdict(list),
    }
    identities: set[tuple[str, int, int]] = set()
    problem_splits: dict[str, str] = {}
    for split, path in split_paths.items():
        for _line_no, row in _iter_jsonl(path):
            _validate_prepared_row(
                row,
                problems,
                expected_sample_mode,
                require_fingerprint=require_fingerprint,
            )
            key = _identity(row)
            if key in identities:
                raise ValueError(f"duplicate prepared identity: {key}")
            identities.add(key)
            problem_id = key[0]
            prior = problem_splits.get(problem_id)
            if prior is not None and prior != split:
                raise ValueError("problem-level train/holdout leakage")
            problem_splits[problem_id] = split
            split_rows[split].append(row)
            prepared_groups[split][key[:2]].append(row)
    if not split_rows["train"] or not split_rows["holdout"]:
        raise ValueError("prepared split is empty")
    for groups in prepared_groups.values():
        for rows in groups.values():
            _group_terminal(rows)
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read prepared stats/manifest") from exc
    contract = stats.get("thinking_contract")
    if not isinstance(contract, dict):
        raise ValueError("prepared thinking contract is missing")
    expected_cross_mode = not expected_sample_mode and expected_eval_mode and expected_eval_required
    expected_contract = {
        "sampling_enable_thinking": expected_sample_mode,
        "training_enable_thinking": expected_train_mode,
        "evaluation_enable_thinking": expected_eval_mode,
        "evaluation_require_thinking": expected_eval_required,
        "documented_cross_mode": expected_cross_mode,
        "prepared_train_rows": len(split_rows["train"]),
        "prepared_holdout_rows": len(split_rows["holdout"]),
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"prepared stats contract {key} disagrees")
    if stats.get("dataset") != "MATH" or stats.get("judge_schema") != JUDGE_SCHEMA:
        raise ValueError("prepared stats schema is stale")
    if stats.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("prepared stats evaluator version is stale")
    if require_fingerprint and stats.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("prepared stats evaluator fingerprint is stale")
    if require_fingerprint and stats.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("prepared stats runtime evaluator fingerprint is stale")
    if manifest.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("prepared manifest evaluator version is stale")
    if require_fingerprint and manifest.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("prepared manifest evaluator fingerprint is stale")
    if require_fingerprint and manifest.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("prepared manifest runtime evaluator fingerprint is stale")
    manifest_outputs = manifest.get("outputs")
    if not isinstance(manifest_outputs, dict):
        raise ValueError("prepared manifest outputs are missing")
    expected_outputs = {
        "train": str(train_path),
        "holdout": str(holdout_path),
        "stats": str(stats_path),
    }
    if any(manifest_outputs.get(key) != value for key, value in expected_outputs.items()):
        raise ValueError("prepared manifest output paths disagree")
    return {
        "train_rows": len(split_rows["train"]),
        "holdout_rows": len(split_rows["holdout"]),
        "problems": len(problem_splits),
        "math_eval_version": MATH_EVAL_VERSION,
    }


def _eval_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("eval row is missing trajectory")
    problem_id = trajectory.get("problem_id")
    rollout_idx = trajectory.get("rollout_idx", 0)
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("eval trajectory has invalid problem_id")
    if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx < 0:
        raise ValueError("eval trajectory has invalid rollout_idx")
    return problem_id, rollout_idx


def _validate_eval_row(
    row: Mapping[str, Any],
    problem: Any,
    *,
    expected_mode: bool,
    expected_required: bool,
    expected_bootstrap_agent: Optional[str],
    require_fingerprint: bool,
) -> None:
    _validate_common_version(row, require_fingerprint=require_fingerprint)
    if row.get("thinking_enabled") is not expected_mode or row.get("enable_thinking") is not expected_mode:
        raise ValueError("eval thinking mode disagrees")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("eval trajectory is missing")
    bootstrap = row.get("bootstrap")
    nested_bootstrap = trajectory.get("bootstrap")
    if expected_bootstrap_agent is not None:
        if bootstrap != nested_bootstrap:
            raise ValueError("eval bootstrap fields disagree")
        _validate_bootstrap_payload(
            bootstrap,
            expected_agent=expected_bootstrap_agent,
            expected_mode=expected_mode,
            expected_required=expected_required,
            allow_pending=False,
        )
    elif bootstrap is not None or nested_bootstrap is not None:
        if bootstrap != nested_bootstrap:
            raise ValueError("eval bootstrap fields disagree")
        _validate_bootstrap_payload(
            bootstrap,
            expected_agent=None,
            expected_mode=expected_mode,
            expected_required=expected_required,
            allow_pending=False,
        )
    if trajectory.get("problem_id") != problem.problem_id:
        raise ValueError("eval trajectory/problem identity mismatch")
    row_start_agent = row.get("start_agent")
    trajectory_start_agent = trajectory.get("start_agent")
    if row_start_agent is not None and row_start_agent not in AGENT_IDS:
        raise ValueError("eval row has invalid start_agent")
    if trajectory_start_agent is not None and trajectory_start_agent not in AGENT_IDS:
        raise ValueError("eval trajectory has invalid start_agent")
    if (
        row_start_agent is not None
        and trajectory_start_agent is not None
        and row_start_agent != trajectory_start_agent
    ):
        raise ValueError("eval start_agent fields disagree")
    expected_start_agent = row_start_agent or trajectory_start_agent
    if (
        expected_bootstrap_agent is not None
        and expected_start_agent != expected_bootstrap_agent
    ):
        raise ValueError("eval first protocol agent disagrees with bootstrap agent")
    problem_payload = row.get("problem")
    if not isinstance(problem_payload, dict):
        raise ValueError("eval problem payload is missing")
    if (
        problem_payload.get("id") != problem.problem_id
        or not isinstance(problem_payload.get("gold_answer"), str)
        or not math_answers_equivalent(problem_payload.get("gold_answer"), problem.gold_answer)
    ):
        raise ValueError("eval problem payload disagrees with canonical data")
    _validate_optional_problem_fields(problem_payload, problem, location="eval problem payload")
    if trajectory.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError("stale eval trajectory evaluator version")
    if require_fingerprint and trajectory.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("stale eval trajectory fingerprint")
    if require_fingerprint and trajectory.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("stale eval trajectory runtime fingerprint")
    terminated = row.get("terminated_by")
    if terminated not in {"stop", "truncated", "rejected_quality", "exception"}:
        raise ValueError("eval trajectory has invalid termination status")
    nested_terminated = trajectory.get("terminated_by")
    if nested_terminated != terminated:
        raise ValueError("eval termination fields disagree")
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("eval trajectory steps are missing")
    parsed_steps = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"eval step {index} is not an object")
        if isinstance(step.get("turn"), bool) or step.get("turn") != index:
            raise ValueError("eval turns are not contiguous")
        active_agent = step.get("active_agent")
        if active_agent not in AGENT_IDS:
            raise ValueError(f"eval step {index} has invalid active_agent")
        if index == 0 and expected_start_agent is not None and active_agent != expected_start_agent:
            raise ValueError("eval first active agent disagrees with start_agent")
        allowed_step_fields = {
            "turn",
            "active_agent",
            "reasoning",
            "tentative_answer",
            "action",
            "handoff_target",
            "handoff_note",
            "confirmed_answer",
            "thinking",
            "raw_output",
            "visible_output",
            "raw_outputs",
        }
        unknown_step_fields = set(step) - allowed_step_fields
        if unknown_step_fields:
            raise ValueError(f"eval step {index} has unknown fields: {sorted(unknown_step_fields)}")
        protocol = {
            field: step.get(field)
            for field in (
                "reasoning",
                "tentative_answer",
                "action",
                "handoff_target",
                "handoff_note",
                "confirmed_answer",
            )
        }
        allow_bootstrap_self_handoff = (
            index == 0
            and isinstance(row.get("bootstrap"), Mapping)
            and protocol.get("handoff_target") == active_agent
        )
        try:
            parsed = validate_protocol_object(
                protocol,
                active_agent=active_agent,
                allow_self_handoff=allow_bootstrap_self_handoff,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid eval protocol at step {index}: {exc}") from exc
        if "thinking" not in step:
            if expected_required:
                raise ValueError(f"eval step {index} is missing thinking provenance")
        elif not isinstance(step.get("thinking"), str):
            raise ValueError(f"eval step {index} thinking must be a string")
        parsed_steps.append(parsed)
    for index, parsed in enumerate(parsed_steps[:-1]):
        if parsed.get("action") == "confirm_stop":
            raise ValueError("eval trajectory contains a turn after confirm_stop")
        if parsed.get("action") == "handoff":
            scheduled_agent = parsed.get("handoff_target")
            actual_agent = steps[index + 1].get("active_agent")
            if scheduled_agent != actual_agent and not _is_audited_degenerate_recovery(
                row,
                turn=index + 1,
                scheduled_agent=scheduled_agent,
                actual_agent=actual_agent,
            ):
                raise ValueError("eval handoff target does not match next active agent")
    if "active_agents" in trajectory:
        active_agents = trajectory.get("active_agents")
        expected_active_agents = []
        for step in steps:
            agent = step.get("active_agent")
            if agent not in expected_active_agents:
                expected_active_agents.append(agent)
        if active_agents != expected_active_agents:
            raise ValueError("eval active_agents summary disagrees with steps")
    if "n_handoffs" in trajectory:
        handoff_count = trajectory.get("n_handoffs")
        if (
            isinstance(handoff_count, bool)
            or not isinstance(handoff_count, int)
            or handoff_count != sum(parsed.get("action") == "handoff" for parsed in parsed_steps)
        ):
            raise ValueError("eval n_handoffs summary disagrees with steps")
    if terminated == "stop":
        if not steps or steps[-1].get("action") != "confirm_stop":
            raise ValueError("eval stop trajectory does not end in confirm_stop")
        final_answer = _answer_text(row.get("final_answer"))
        nested_final = _answer_text(trajectory.get("final_answer"))
        if not final_answer or not math_answers_equivalent(final_answer, nested_final):
            raise ValueError("eval final_answer fields disagree")
        confirmed = _answer_text(steps[-1].get("confirmed_answer"), "confirmed_answer")
        if not confirmed or not math_answers_equivalent(final_answer, confirmed):
            raise ValueError("eval final_answer disagrees with terminal confirmation")
    elif terminated == "truncated":
        final_answer = _answer_text(row.get("final_answer"))
        nested_final = _answer_text(trajectory.get("final_answer"))
        tentative = _answer_text(
            steps[-1].get("tentative_answer") if steps else None,
            "tentative_answer",
        )
        if not final_answer or final_answer != nested_final or final_answer != tentative:
            raise ValueError(
                "truncated eval final_answer is not the last tentative answer"
            )
    elif _answer_text(row.get("final_answer")) or _answer_text(trajectory.get("final_answer")):
        raise ValueError("failed eval trajectory has final_answer")
    if expected_mode:
        for step in steps:
            thinking = step.get("thinking")
            if expected_required and (not isinstance(thinking, str) or not thinking.strip()):
                raise ValueError("eval thinking trace is empty")
    else:
        for step in steps:
            if str(step.get("thinking") or "").strip():
                raise ValueError("non-thinking eval row contains thinking")
    terminal = _answer_text(row.get("final_answer"))
    _validate_binary_em(row.get("em"), compute_math_em(terminal, problem.gold_answer))


def validate_eval_output(
    output_path: Path,
    data_root: Path,
    *,
    split: str = "test",
    start: int = 0,
    limit: int = 0,
    subjects: Any = None,
    expected_mode: bool = True,
    expected_required: bool = True,
    expected_bootstrap_agent: Optional[str] = None,
    require_fingerprint: bool = True,
    allow_incomplete: bool = False,
) -> dict[str, int | str]:
    normalized_subjects = _normalize_subjects(subjects)
    problems = _load_eval_problems(data_root, split, normalized_subjects)
    selected = problems[start : start + limit] if limit else problems[start:]
    selected_map = {str(problem.problem_id): problem for problem in selected}
    if limit and len(selected) != limit:
        raise ValueError("eval selection is shorter than configured limit")
    seen: set[tuple[str, int]] = set()
    rows = 0
    for line_no, row in _iter_jsonl(output_path):
        key = _eval_identity(row)
        if key in seen:
            raise ValueError(f"duplicate eval identity at line {line_no}")
        if key[0] not in selected_map:
            raise ValueError(f"eval identity outside configured slice at line {line_no}")
        _validate_eval_row(
            row,
            selected_map[key[0]],
            expected_mode=expected_mode,
            expected_required=expected_required,
            expected_bootstrap_agent=expected_bootstrap_agent,
            require_fingerprint=require_fingerprint,
        )
        seen.add(key)
        rows += 1
    expected = {(problem_id, 0) for problem_id in selected_map}
    if not allow_incomplete and seen != expected:
        raise ValueError(f"eval identities are incomplete: missing={len(expected - seen)}")
    if rows == 0:
        raise ValueError("eval output is empty")
    return {"rows": rows, "problems": len({key[0] for key in seen}), "math_eval_version": MATH_EVAL_VERSION}


def validate_eval_state(
    state_path: Path,
    data_root: Path,
    *,
    split: str = "test",
    start: int = 0,
    limit: int = 0,
    expected_mode: bool = True,
    expected_required: bool = True,
    expected_bootstrap_agent: Optional[str] = None,
    fingerprint_path: Optional[Path] = None,
    subjects: Any = None,
) -> dict[str, int | str]:
    """Audit durable eval state without changing the runner protocol."""
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("evaluation state start must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("evaluation state limit must be a non-negative integer")
    if not isinstance(expected_mode, bool) or not isinstance(expected_required, bool):
        raise ValueError("evaluation state thinking expectations must be boolean")
    normalized_subjects = _normalize_subjects(subjects)
    if fingerprint_path is not None:
        validate_fingerprint_file(
            fingerprint_path,
            expected={
                "split": split,
                "start": start,
                "limit": limit,
                "expected_mode": expected_mode,
                "expected_required": expected_required,
                "output_mode": "trajectories",
                "num_rollouts": 1,
                "bootstrap_agent": expected_bootstrap_agent,
                "subjects": normalized_subjects,
            },
        )
    try:
        state = json.loads(
            state_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation state: {state_path}") from exc
    if not isinstance(state, dict):
        raise ValueError("evaluation state must be an object")
    _validate_embedded_fingerprints(state, location="evaluation state")
    if state.get("version") != 1:
        raise ValueError("unsupported evaluation state version")
    config = state.get("config")
    items = state.get("items")
    if not isinstance(config, dict) or not isinstance(items, list):
        raise ValueError("evaluation state structure is invalid")
    _validate_embedded_fingerprints(config, location="evaluation state config")
    if config.get("output_mode") != "trajectories":
        raise ValueError("evaluation state mode/rollout contract disagrees")
    if isinstance(config.get("num_rollouts"), bool) or config.get("num_rollouts") != 1:
        raise ValueError("evaluation state mode/rollout contract disagrees")
    if (
        isinstance(config.get("start"), bool)
        or not isinstance(config.get("start"), int)
        or isinstance(config.get("limit"), bool)
        or not isinstance(config.get("limit"), int)
    ):
        raise ValueError("evaluation state start/limit are not integers")
    if config.get("start") != start or (limit and config.get("limit") != limit):
        raise ValueError("evaluation state slice disagrees with this run")
    if isinstance(config.get("limit"), bool) or not isinstance(config.get("limit"), int) or config.get("limit") <= 0:
        raise ValueError("evaluation state config limit is invalid")
    if isinstance(config.get("t_max"), bool) or not isinstance(config.get("t_max"), int) or config.get("t_max") <= 0:
        raise ValueError("evaluation state config t_max is invalid")
    if config.get("enable_thinking") is not expected_mode or config.get("require_thinking") is not expected_required:
        raise ValueError("evaluation state thinking contract disagrees")
    if config.get("require_thinking") and not config.get("enable_thinking"):
        raise ValueError("evaluation state requires thinking while disabled")
    if config.get("bootstrap_agent") != expected_bootstrap_agent:
        raise ValueError("evaluation state bootstrap agent disagrees")
    if expected_bootstrap_agent is not None:
        if config.get("allow_first_turn_stop") is not False:
            raise ValueError("evaluation state allows the first protocol turn to stop")
        if config.get("json_transport") != "json_schema":
            raise ValueError("evaluation state does not use strict protocol JSON schema")
        protocol_thinking_max_tokens = config.get("protocol_thinking_max_tokens")
        if (
            isinstance(protocol_thinking_max_tokens, bool)
            or not isinstance(protocol_thinking_max_tokens, int)
            or protocol_thinking_max_tokens <= 0
        ):
            raise ValueError("evaluation state protocol thinking cap is invalid")
    if config.get("split", split) != split:
        raise ValueError("evaluation state split disagrees with this run")
    if "subjects" in config:
        if _normalize_subjects(config.get("subjects")) != normalized_subjects:
            raise ValueError("evaluation state subjects disagree with this run")
    elif normalized_subjects:
        raise ValueError("evaluation state is missing its subject filter")
    problems = _load_eval_problems(data_root, split, normalized_subjects)
    selected = problems[start : start + limit] if limit else problems[start:]
    if limit and len(selected) != limit:
        raise ValueError("evaluation state selection is shorter than configured limit")
    selected_map = {
        str(problem.problem_id): (start + index, problem)
        for index, problem in enumerate(selected)
    }
    if len(items) != len(selected):
        raise ValueError("evaluation state item count disagrees with configured limit")
    seen: set[tuple[int, int]] = set()
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation state item {item_index} is not an object")
        problem = item.get("problem")
        if not isinstance(problem, dict):
            raise ValueError(f"evaluation state item {item_index} has no problem")
        problem_id = problem.get("problem_id")
        if not isinstance(problem_id, str) or problem_id not in selected_map:
            raise ValueError(f"evaluation state item {item_index} is outside configured slice")
        problem_index, canonical = selected_map[problem_id]
        rollout_idx = item.get("rollout_idx")
        if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx != 0:
            raise ValueError(f"evaluation state item {item_index} has invalid rollout_idx")
        if item.get("problem_index") != problem_index:
            raise ValueError(f"evaluation state item {item_index} has invalid problem_index")
        if (
            not isinstance(problem.get("gold_answer"), str)
            or not math_answers_equivalent(problem.get("gold_answer"), canonical.gold_answer)
        ):
            raise ValueError(f"evaluation state item {item_index} gold answer disagrees")
        _validate_optional_problem_fields(problem, canonical, location=f"evaluation state item {item_index} problem")
        start_agent = item.get("start_agent")
        if start_agent not in AGENT_IDS:
            raise ValueError(f"evaluation state item {item_index} has invalid start_agent")
        if expected_bootstrap_agent is not None and start_agent != expected_bootstrap_agent:
            raise ValueError(
                f"evaluation state item {item_index} first protocol agent is not "
                f"{expected_bootstrap_agent}"
            )
        if "current_agent" in item:
            current_agent = item.get("current_agent")
            if item.get("status") == "pending":
                if current_agent not in AGENT_IDS:
                    raise ValueError(f"pending evaluation state item {item_index} has invalid current_agent")
            elif current_agent is not None:
                raise ValueError(f"terminal evaluation state item {item_index} has current_agent")
        if "group_attempt" in item:
            group_attempt = item.get("group_attempt")
            if isinstance(group_attempt, bool) or not isinstance(group_attempt, int) or group_attempt < 0:
                raise ValueError(f"evaluation state item {item_index} has invalid group_attempt")
        if "last_error" in item and item.get("last_error") is not None and not isinstance(item.get("last_error"), str):
            raise ValueError(f"evaluation state item {item_index} has invalid last_error")
        bootstrap = item.get("bootstrap")
        if expected_bootstrap_agent is not None:
            _validate_bootstrap_payload(
                bootstrap,
                expected_agent=expected_bootstrap_agent,
                expected_mode=expected_mode,
                expected_required=expected_required,
                allow_pending=True,
            )
        elif bootstrap is not None:
            raise ValueError(f"evaluation state item {item_index} has unexpected bootstrap")
        identity = (problem_index, rollout_idx)
        if identity in seen:
            raise ValueError(f"duplicate evaluation state identity: {identity}")
        seen.add(identity)
        trajectory = item.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError(f"evaluation state item {item_index} has no trajectory")
        if trajectory.get("problem_id") != problem_id or trajectory.get("rollout_idx") != rollout_idx:
            raise ValueError(f"evaluation state item {item_index} trajectory identity disagrees")
        if trajectory.get("start_agent") != item.get("start_agent"):
            raise ValueError(f"evaluation state item {item_index} start-agent identity disagrees")
        _validate_embedded_fingerprints(trajectory, location=f"evaluation state item {item_index} trajectory")
        if trajectory.get("final_answer") is not None and not isinstance(trajectory.get("final_answer"), str):
            raise ValueError(f"evaluation state item {item_index} final_answer is invalid")
        if trajectory.get("error") is not None and not isinstance(trajectory.get("error"), str):
            raise ValueError(f"evaluation state item {item_index} trajectory error is invalid")
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"evaluation state item {item_index} steps are invalid")
        if len(steps) > int(config.get("t_max", len(steps))):
            raise ValueError(f"evaluation state item {item_index} exceeds t_max")
        if "messages" in item:
            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"evaluation state item {item_index} messages are invalid")
            if len(messages) != len(steps) + 2:
                raise ValueError(f"evaluation state item {item_index} messages/steps disagree")
            for message_index, message in enumerate(messages):
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not message.get("role", "").strip()
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(
                        f"evaluation state item {item_index} message {message_index} is invalid"
                    )
            if len(messages) >= 2 and (
                messages[0].get("role") != "system" or messages[1].get("role") != "user"
            ):
                raise ValueError(f"evaluation state item {item_index} messages have invalid prefix")
        if "turn_messages" in trajectory:
            turn_messages = trajectory.get("turn_messages")
            if not isinstance(turn_messages, list) or len(turn_messages) != len(steps):
                raise ValueError(f"evaluation state item {item_index} turn_messages disagree")
            for turn_index, turn_messages_item in enumerate(turn_messages):
                if not isinstance(turn_messages_item, list):
                    raise ValueError(
                        f"evaluation state item {item_index} turn_messages[{turn_index}] is invalid"
                    )
                for message_index, message in enumerate(turn_messages_item):
                    if (
                        not isinstance(message, dict)
                        or not isinstance(message.get("role"), str)
                        or not isinstance(message.get("content"), str)
                    ):
                        raise ValueError(
                            f"evaluation state item {item_index} turn_messages[{turn_index}]"
                            f"[{message_index}] is invalid"
                        )
        parsed_steps = []
        min_agents_before_stop = config.get("min_agents_before_stop", 1)
        if (
            isinstance(min_agents_before_stop, bool)
            or not isinstance(min_agents_before_stop, int)
            or not 1 <= min_agents_before_stop <= len(AGENT_IDS)
        ):
            raise ValueError("evaluation state min_agents_before_stop is invalid")
        allow_first_turn_stop = config.get("allow_first_turn_stop", True)
        if not isinstance(allow_first_turn_stop, bool):
            raise ValueError("evaluation state allow_first_turn_stop is invalid")
        seen_agents = []
        expected_agent = item.get("start_agent")
        for step_index, step in enumerate(steps):
            if (
                not isinstance(step, dict)
                or isinstance(step.get("turn"), bool)
                or not isinstance(step.get("turn"), int)
                or step.get("turn") != step_index
            ):
                raise ValueError(f"evaluation state item {item_index} has non-contiguous steps")
            actual_agent = step.get("active_agent")
            if expected_agent not in AGENT_IDS or (
                actual_agent != expected_agent
                and not _is_audited_degenerate_recovery(
                    item,
                    turn=step_index,
                    scheduled_agent=expected_agent,
                    actual_agent=actual_agent,
                )
            ):
                raise ValueError(f"evaluation state item {item_index} active-agent sequence disagrees")
            allowed_step_fields = {
                "turn",
                "active_agent",
                "reasoning",
                "tentative_answer",
                "action",
                "handoff_target",
                "handoff_note",
                "confirmed_answer",
                "raw_output",
                "visible_output",
                "thinking",
                "raw_outputs",
            }
            unknown_step_fields = set(step) - allowed_step_fields
            if unknown_step_fields:
                raise ValueError(
                    f"evaluation state item {item_index} step {step_index} has unknown fields: "
                    f"{sorted(unknown_step_fields)}"
                )
            for text_field in ("reasoning", "tentative_answer", "raw_output", "visible_output"):
                if text_field in step and not isinstance(step.get(text_field), str):
                    raise ValueError(
                        f"evaluation state item {item_index} step {step_index} {text_field} is invalid"
                    )
            if "thinking" in step and not isinstance(step.get("thinking"), str):
                raise ValueError(f"evaluation state item {item_index} step {step_index} thinking is invalid")
            if "raw_outputs" in step and (
                not isinstance(step.get("raw_outputs"), list)
                or not all(isinstance(value, str) for value in step.get("raw_outputs"))
            ):
                raise ValueError(f"evaluation state item {item_index} step {step_index} raw_outputs is invalid")
            protocol = {field: step.get(field) for field in (
                "reasoning", "tentative_answer", "action", "handoff_target", "handoff_note", "confirmed_answer"
            )}
            allow_bootstrap_self_handoff = (
                step_index == 0
                and protocol.get("handoff_target") == step.get("active_agent")
                and protocol.get("handoff_target") == config.get("bootstrap_agent")
                and protocol.get("handoff_target")
                == config.get("bootstrap_handoff_target")
            )
            try:
                parsed = validate_protocol_object(
                    protocol,
                    active_agent=step.get("active_agent"),
                    allow_self_handoff=allow_bootstrap_self_handoff,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid state protocol at item {item_index}, step {step_index}: {exc}") from exc
            if expected_mode and expected_required and not str(step.get("thinking") or "").strip():
                raise ValueError(f"empty state thinking trace at item {item_index}, step {step_index}")
            if not expected_mode and str(step.get("thinking") or "").strip():
                raise ValueError(f"non-thinking state contains thinking at item {item_index}, step {step_index}")
            if parsed["action"] == "confirm_stop":
                if not allow_first_turn_stop and not seen_agents:
                    raise ValueError(
                        f"evaluation state item {item_index} confirms on the first turn"
                    )
                distinct_agents = set(seen_agents) | {step["active_agent"]}
                if len(distinct_agents) < min_agents_before_stop:
                    raise ValueError(
                        f"evaluation state item {item_index} confirms before the minimum agent count"
                    )
                if not math_answers_equivalent(
                    str(parsed.get("tentative_answer") or ""),
                    str(parsed.get("confirmed_answer") or ""),
                ):
                    raise ValueError(
                        f"evaluation state item {item_index} tentative/confirmed answers disagree"
                    )
            parsed_steps.append(parsed)
            if step["active_agent"] not in seen_agents:
                seen_agents.append(step["active_agent"])
            if parsed.get("action") == "handoff":
                expected_agent = parsed.get("handoff_target")
            else:
                expected_agent = None
        for step_index, parsed in enumerate(parsed_steps[:-1]):
            if parsed.get("action") == "confirm_stop":
                raise ValueError(f"state item {item_index} continues after confirm_stop")
            if parsed.get("action") == "handoff":
                scheduled_agent = parsed.get("handoff_target")
                actual_agent = steps[step_index + 1].get("active_agent")
                if scheduled_agent != actual_agent and not _is_audited_degenerate_recovery(
                    item,
                    turn=step_index + 1,
                    scheduled_agent=scheduled_agent,
                    actual_agent=actual_agent,
                ):
                    raise ValueError(f"state item {item_index} handoff target disagrees")
        status = item.get("status")
        terminated = trajectory.get("terminated_by")
        if status not in {"pending", "done", "retry", "failed"}:
            raise ValueError(f"evaluation state item {item_index} has invalid status")
        if terminated not in {"stop", "truncated", "rejected_quality", "exception"}:
            raise ValueError(f"evaluation state item {item_index} has invalid termination")
        if not isinstance(item.get("prior_tentative"), bool):
            raise ValueError(f"evaluation state item {item_index} prior_tentative is not boolean")
        if bool(steps) != item.get("prior_tentative"):
            raise ValueError(f"evaluation state item {item_index} prior_tentative disagrees")
        if status == "pending":
            if terminated != "truncated" or _answer_text(trajectory.get("final_answer")):
                raise ValueError(f"pending evaluation state item {item_index} has terminal fields")
            if len(steps) >= int(config["t_max"]):
                raise ValueError(f"pending evaluation state item {item_index} reached t_max")
            if "current_agent" in item and item.get("current_agent") != expected_agent:
                raise ValueError(
                    f"pending evaluation state item {item_index} current agent disagrees"
                )
        elif terminated == "stop":
            if status != "done" or not parsed_steps or parsed_steps[-1].get("action") != "confirm_stop":
                raise ValueError(f"state item {item_index} stop status is inconsistent")
            final_answer = _answer_text(trajectory.get("final_answer"))
            confirmed = _answer_text(parsed_steps[-1].get("confirmed_answer"), "confirmed_answer")
            if not final_answer or not confirmed or not math_answers_equivalent(final_answer, confirmed):
                raise ValueError(f"state item {item_index} final answer is not bound to confirmation")
            tentative = _answer_text(parsed_steps[-1].get("tentative_answer"), "tentative_answer")
            if not tentative or not math_answers_equivalent(final_answer, tentative):
                raise ValueError(f"state item {item_index} final answer disagrees with tentative answer")
            if not compute_math_em(final_answer, canonical.gold_answer) in {0.0, 1.0}:
                raise ValueError(f"state item {item_index} has invalid terminal EM")
        elif terminated == "truncated":
            final_answer = _answer_text(trajectory.get("final_answer"))
            tentative = _answer_text(
                parsed_steps[-1].get("tentative_answer") if parsed_steps else None,
                "tentative_answer",
            )
            if not final_answer or final_answer != tentative:
                raise ValueError(
                    f"state item {item_index} truncated final answer is not the last tentative answer"
                )
        elif _answer_text(trajectory.get("final_answer")):
            raise ValueError(f"state item {item_index} failed trajectory has final answer")
        if terminated == "truncated" and status != "pending" and len(steps) != int(config["t_max"]):
            raise ValueError(f"state item {item_index} truncated before t_max")
    expected_identities = {(start + index, 0) for index in range(len(selected))}
    if seen != expected_identities:
        raise ValueError("evaluation state identities are incomplete")
    return {"items": len(items), "completed": sum(item.get("status") == "done" for item in items), "math_eval_version": MATH_EVAL_VERSION}


def validate_fingerprint_file(path: Path, expected: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluator fingerprint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("evaluator fingerprint must be an object")
    if payload.get("math_eval_version") != MATH_EVAL_VERSION or payload.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError("evaluator fingerprint/version is stale")
    if "evaluator_fingerprint" in payload and payload.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError("runtime evaluator fingerprint is stale")
    if expected:
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"evaluator fingerprint field {key} disagrees")
    return payload


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--source", type=Path, required=True)
    score.add_argument("--judged", type=Path, required=True)
    score.add_argument("--stats", type=Path, required=True)
    score.add_argument("--data-root", type=Path, required=True)
    score.add_argument("--expected-rollouts", type=int, required=True)
    score.add_argument("--failure-ledger", type=Path, default=os.environ.get("MATH_FAILURE_LEDGER") or None)
    score.add_argument("--sample-state", type=Path, default=os.environ.get("MATH_SAMPLE_STATE") or None)
    score.add_argument("--allow-missing-fingerprint", action="store_true")
    prepared = subparsers.add_parser("prepared")
    prepared.add_argument("--train", type=Path, required=True)
    prepared.add_argument("--holdout", type=Path, required=True)
    prepared.add_argument("--stats", type=Path, required=True)
    prepared.add_argument("--manifest", type=Path, required=True)
    prepared.add_argument("--data-root", type=Path, required=True)
    prepared.add_argument("--sample-mode", type=int, choices=(0, 1), required=True)
    prepared.add_argument("--train-mode", type=int, choices=(0, 1), required=True)
    prepared.add_argument("--eval-mode", type=int, choices=(0, 1), required=True)
    prepared.add_argument("--eval-required", type=int, choices=(0, 1), required=True)
    prepared.add_argument("--allow-missing-fingerprint", action="store_true")
    evaluation = subparsers.add_parser("eval")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path, required=True)
    evaluation.add_argument("--split", default="test")
    evaluation.add_argument("--subjects", default="")
    evaluation.add_argument("--start", type=int, default=0)
    evaluation.add_argument("--limit", type=int, default=0)
    evaluation.add_argument("--expected-mode", type=int, choices=(0, 1), default=1)
    evaluation.add_argument("--expected-required", type=int, choices=(0, 1), default=1)
    evaluation.add_argument("--expected-bootstrap-agent", choices=AGENT_IDS)
    evaluation.add_argument("--allow-incomplete", action="store_true")
    evaluation.add_argument("--allow-missing-fingerprint", action="store_true")
    state = subparsers.add_parser("state")
    state.add_argument("--state", type=Path, required=True)
    state.add_argument("--data-root", type=Path, required=True)
    state.add_argument("--split", default="test")
    state.add_argument("--subjects", default="")
    state.add_argument("--start", type=int, default=0)
    state.add_argument("--limit", type=int, default=0)
    state.add_argument("--expected-mode", type=int, choices=(0, 1), default=1)
    state.add_argument("--expected-required", type=int, choices=(0, 1), default=1)
    state.add_argument("--expected-bootstrap-agent", choices=AGENT_IDS)
    state.add_argument("--fingerprint", type=Path)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--path", type=Path, required=True)
    fingerprint.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "score":
            result = validate_scored_output(
                args.source,
                args.judged,
                args.stats,
                args.data_root,
                args.expected_rollouts,
                require_fingerprint=not args.allow_missing_fingerprint,
                failure_ledger=args.failure_ledger,
                sample_state=args.sample_state,
            )
        elif args.command == "prepared":
            result = validate_prepared_outputs(
                args.train,
                args.holdout,
                args.stats,
                args.manifest,
                args.data_root,
                expected_sample_mode=bool(args.sample_mode),
                expected_train_mode=bool(args.train_mode),
                expected_eval_mode=bool(args.eval_mode),
                expected_eval_required=bool(args.eval_required),
                require_fingerprint=not args.allow_missing_fingerprint,
            )
        elif args.command == "eval":
            result = validate_eval_output(
                args.output,
                args.data_root,
                split=args.split,
                subjects=args.subjects,
                start=args.start,
                limit=args.limit,
                expected_mode=bool(args.expected_mode),
                expected_required=bool(args.expected_required),
                expected_bootstrap_agent=args.expected_bootstrap_agent,
                require_fingerprint=not args.allow_missing_fingerprint,
                allow_incomplete=args.allow_incomplete,
            )
        elif args.command == "state":
            result = validate_eval_state(
                args.state,
                args.data_root,
                split=args.split,
                subjects=args.subjects,
                start=args.start,
                limit=args.limit,
                expected_mode=bool(args.expected_mode),
                expected_required=bool(args.expected_required),
                expected_bootstrap_agent=args.expected_bootstrap_agent,
                fingerprint_path=args.fingerprint,
            )
        else:
            payload = {
                "math_eval_version": MATH_EVAL_VERSION,
                "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
                "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
            }
            if args.write:
                args.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.path.with_name(f".{args.path.name}.tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(args.path)
            result = payload
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[validator] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
