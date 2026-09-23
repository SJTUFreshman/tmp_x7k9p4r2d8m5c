#!/usr/bin/env python3
"""Sample or evaluate three-role MATH trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.inference import (  # noqa: E402
    GenerationOptions,
    OpenAIChatLLMCaller,
    response_attempts,
)
from jca.src.math_eval import (  # noqa: E402
    AGENT_IDS,
    MATH_EVAL_VERSION,
    MathProblem,
    compute_math_em,
    format_math_problem_as_prompt,
    load_math_problems,
    math_answers_equivalent,
    normalize_math_answer,
    render_math_mas_system_prompt,
)
from jca.experiments.math_rl_mas_thinking.math_artifact_validator import (  # noqa: E402
    EVALUATOR_FINGERPRINT,
    MATH_EVAL_FINGERPRINT,
)
from jca.src.protocol_json import (  # noqa: E402
    json_object_well_formed,
    parse_protocol_object,
    protocol_schema_error,
    protocol_json_schema_well_formed as strict_protocol_json_well_formed,
    validate_protocol_object,
)


THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)


@dataclass
class MathStep:
    turn: int
    active_agent: str
    reasoning: str
    tentative_answer: str
    action: str
    handoff_target: Optional[str]
    handoff_note: Optional[str]
    confirmed_answer: Optional[str]
    raw_output: str
    visible_output: str
    thinking: str
    raw_outputs: List[str] = field(default_factory=list)


@dataclass
class MathTrajectory:
    problem_id: str
    rollout_idx: int
    start_agent: str
    steps: List[MathStep] = field(default_factory=list)
    turn_messages: List[List[Dict[str, str]]] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated_by: str = "truncated"
    error: Optional[str] = None
    failure_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def active_agents(self) -> List[str]:
        output: List[str] = []
        for step in self.steps:
            if step.active_agent not in output:
                output.append(step.active_agent)
        return output

    @property
    def n_handoffs(self) -> int:
        return sum(step.action == "handoff" for step in self.steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MATH RL-MAS sampler/evaluator with configurable thinking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument(
        "--start-agent", choices=[*AGENT_IDS, "balanced", "random"], default="A1"
    )
    parser.add_argument("--start-agent-seed", type=int, default=42)
    parser.add_argument("--min-agents-before-stop", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--group-retries", type=int, default=5)
    parser.add_argument("--step-retries", type=int, default=4)
    parser.add_argument("--output-mode", choices=["turns", "trajectories"], default="turns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-raw-chars", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_thinking(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    matches = list(THINK_RE.finditer(text))
    thinking = "\n\n".join(match.group(1).strip() for match in matches if match.group(1).strip())
    visible = THINK_RE.sub("", text)
    unclosed = re.search(r"<think\b[^>]*>", visible, flags=re.IGNORECASE)
    if unclosed is not None:
        fragment = visible[unclosed.end() :].strip()
        if fragment:
            thinking = "\n\n".join(part for part in (thinking, fragment) if part)
        visible = visible[: unclosed.start()]
    visible = re.sub(r"</?think\b[^>]*>", "", visible, flags=re.IGNORECASE).strip()
    return thinking, visible


def _nullable(value: Any) -> Optional[str]:
    if value is None:
        return None
    result = str(value).strip()
    return None if result.lower() in {"", "none", "null"} else result


def parse_protocol(visible: str, agent: str, turn: int, raw: str) -> MathStep:
    payload: Dict[str, Any] = {}
    try:
        payload = parse_protocol_object(visible)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return MathStep(
        turn=turn,
        active_agent=agent,
        reasoning=str(payload.get("reasoning") or "").strip(),
        tentative_answer=str(payload.get("tentative_answer") or "").strip(),
        action=str(payload.get("action") or "").strip(),
        handoff_target=_nullable(payload.get("handoff_target")),
        handoff_note=_nullable(payload.get("handoff_note")),
        confirmed_answer=_nullable(payload.get("confirmed_answer")),
        raw_output=raw,
        visible_output=visible,
        thinking="",
    )


def parse_strict_protocol(visible: str, agent: str, turn: int, raw: str) -> MathStep:
    """Build a MathStep without coercing or repairing protocol fields."""

    payload = validate_protocol_object(visible, active_agent=agent)
    return MathStep(
        turn=turn,
        active_agent=agent,
        reasoning=payload["reasoning"].strip(),
        tentative_answer=payload["tentative_answer"].strip(),
        action=payload["action"],
        handoff_target=payload["handoff_target"],
        handoff_note=payload["handoff_note"],
        confirmed_answer=payload["confirmed_answer"],
        raw_output=raw,
        visible_output=visible,
        thinking="",
    )


def protocol_json_well_formed(
    visible: str, active_agent: Optional[str] = None
) -> bool:
    """Return whether the visible model output satisfies the protocol schema."""
    return strict_protocol_json_well_formed(
        visible,
        active_agent=active_agent,
        agent_ids=AGENT_IDS,
    )


def invalid_step_reason(
    step: MathStep,
    *,
    prior_tentative: bool,
    seen_agents: Sequence[str],
    min_agents_before_stop: int,
    require_thinking: bool,
    allow_self_handoff: bool = False,
) -> Optional[str]:
    if require_thinking and not step.thinking.strip():
        return "missing non-empty thinking trace"
    if not step.reasoning or not step.tentative_answer:
        return "missing reasoning or tentative_answer"
    if step.action not in {"handoff", "confirm_stop"}:
        return "invalid action"
    if step.action == "handoff":
        if step.handoff_target not in AGENT_IDS or (
            step.handoff_target == step.active_agent and not allow_self_handoff
        ):
            return "invalid handoff target"
        if step.confirmed_answer is not None:
            return "handoff contains confirmed_answer"
    else:
        distinct = set(seen_agents) | {step.active_agent}
        if not prior_tentative:
            return "confirm_stop before a previous tentative answer"
        if len(distinct) < min_agents_before_stop:
            return "confirm_stop before minimum distinct-agent count"
        if not step.confirmed_answer:
            return "confirm_stop missing confirmed_answer"
        if not math_answers_equivalent(step.confirmed_answer, step.tentative_answer):
            return "confirm_stop tentative/confirmed mismatch"
        if step.handoff_target is not None:
            return "confirm_stop contains handoff_target"
    return None


def render_shared_message(step: MathStep) -> str:
    parts = [f"[{step.active_agent}]", step.reasoning]
    parts.append(f"tentative_answer: {step.tentative_answer}")
    if step.action == "handoff":
        line = f"→ handoff to {step.handoff_target}"
        if step.handoff_note:
            line += f": {step.handoff_note}"
        parts.append(line)
    else:
        parts.append(f"confirmed_answer: {step.confirmed_answer}")
    return "\n".join(part for part in parts if part)


def start_agent_for(problem_index: int, configured: str, seed: int) -> str:
    if configured == "balanced":
        return AGENT_IDS[problem_index % len(AGENT_IDS)]
    if configured == "random":
        digest = hashlib.sha256(f"{seed}:{problem_index}".encode()).digest()
        return AGENT_IDS[int.from_bytes(digest[:8], "big") % len(AGENT_IDS)]
    return configured


def request_seed(base: int, problem_index: int, rollout_idx: int, turn: int, attempt: int) -> int:
    digest = hashlib.sha256(
        f"{base}:{problem_index}:{rollout_idx}:{turn}:{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_once(
    problem: MathProblem,
    problem_index: int,
    rollout_idx: int,
    callers: Dict[str, OpenAIChatLLMCaller],
    args: argparse.Namespace,
    group_attempt: int,
) -> MathTrajectory:
    first_agent = start_agent_for(problem_index, args.start_agent, args.start_agent_seed)
    trajectory = MathTrajectory(problem.problem_id, rollout_idx, first_agent)
    messages = [
        {
            "role": "system",
            "content": render_math_mas_system_prompt(
                first_agent, min_agents_before_stop=args.min_agents_before_stop
            ),
        },
        {"role": "user", "content": format_math_problem_as_prompt(problem)},
    ]
    current_agent = first_agent
    prior_tentative = False
    try:
        for turn in range(args.t_max):
            accepted: Optional[MathStep] = None
            last_reason = "no generation"
            output_attempts = 0
            malformed_json_attempts = 0
            protocol_schema_violation_attempts = 0
            thinking_mode_violation_attempts = 0
            generation_error_attempts = 0
            raw_outputs: list[str] = []
            for attempt in range(args.step_retries + 1):
                request_messages = [dict(message) for message in messages]
                if attempt:
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "The last response violated the protocol. Recompute the math, then "
                            "output one complete JSON object with every required field."
                        ),
                    })
                try:
                    raw = callers[current_agent].generate(
                        request_messages,
                        seed=request_seed(
                            args.generation_seed + group_attempt * 100_003,
                            problem_index,
                            rollout_idx,
                            turn,
                            attempt,
                        ),
                    )
                    output_attempts += 1
                    raw_outputs.extend(response_attempts(raw))
                    thinking, visible = split_thinking(str(raw))
                    if not json_object_well_formed(visible):
                        malformed_json_attempts += 1
                        last_reason = "malformed JSON protocol"
                        continue
                    schema_reason = protocol_schema_error(
                        visible, active_agent=current_agent
                    )
                    if schema_reason is not None:
                        protocol_schema_violation_attempts += 1
                        last_reason = schema_reason
                        continue
                    if not args.enable_thinking and thinking.strip():
                        thinking_mode_violation_attempts += 1
                        last_reason = (
                            "thinking mode violation: non-empty thinking in "
                            "non-thinking mode"
                        )
                        continue
                    if not args.enable_thinking:
                        thinking = ""
                    candidate = parse_strict_protocol(
                        visible, current_agent, turn, str(raw)
                    )
                    candidate.thinking = thinking
                    candidate.raw_outputs = list(raw_outputs)
                    last_reason = invalid_step_reason(
                        candidate,
                        prior_tentative=prior_tentative,
                        seen_agents=trajectory.active_agents,
                        min_agents_before_stop=args.min_agents_before_stop,
                        require_thinking=args.require_thinking,
                    ) or ""
                    if not last_reason:
                        candidate.raw_outputs = list(raw_outputs)
                        accepted = candidate
                        break
                except Exception as exc:
                    generation_error_attempts += 1
                    last_reason = str(exc)
            if accepted is None:
                trajectory.terminated_by = "rejected_quality"
                trajectory.error = f"turn {turn} agent {current_agent}: {last_reason}"
                raw_log_chars = int(getattr(args, "log_raw_chars", 0))
                trajectory.failure_diagnostics.append({
                    "turn": turn,
                    "agent": current_agent,
                    "output_attempts": output_attempts,
                    "step_retries": args.step_retries,
                    "malformed_json_attempts": malformed_json_attempts,
                    "protocol_schema_violation_attempts": protocol_schema_violation_attempts,
                    "thinking_mode_violation_attempts": thinking_mode_violation_attempts,
                    "generation_error_attempts": generation_error_attempts,
                    "last_reason": last_reason,
                    "raw_output_count": len(raw_outputs),
                    "raw_outputs": (
                        [str(value)[:raw_log_chars] for value in raw_outputs]
                        if raw_log_chars > 0
                        else []
                    ),
                })
                if (
                    output_attempts == args.step_retries + 1
                    and (
                        malformed_json_attempts
                        + protocol_schema_violation_attempts
                        + thinking_mode_violation_attempts
                        == output_attempts
                    )
                    and generation_error_attempts == 0
                ):
                    trajectory.error = f"turn {turn} agent {current_agent}: {last_reason}"
                return trajectory
            trajectory.turn_messages.append([dict(message) for message in messages])
            trajectory.steps.append(accepted)
            prior_tentative = True
            messages.append({"role": "assistant", "content": render_shared_message(accepted)})
            if accepted.action == "confirm_stop":
                trajectory.final_answer = accepted.confirmed_answer or accepted.tentative_answer
                trajectory.terminated_by = "stop"
                return trajectory
            current_agent = str(accepted.handoff_target)
            messages[0] = {
                "role": "system",
                "content": render_math_mas_system_prompt(
                    current_agent, min_agents_before_stop=args.min_agents_before_stop
                ),
            }
        return trajectory
    except Exception as exc:
        trajectory.terminated_by = "exception"
        trajectory.error = str(exc)
        trajectory.failure_diagnostics.append({
            "turn": len(trajectory.steps),
            "agent": current_agent,
            "output_attempts": 0,
            "last_reason": str(exc),
        })
        return trajectory


def process_group(
    problem: MathProblem,
    problem_index: int,
    rollout_idx: int,
    callers: Dict[str, OpenAIChatLLMCaller],
    args: argparse.Namespace,
) -> MathTrajectory:
    last = None
    for group_attempt in range(args.group_retries + 1):
        last = run_once(
            problem,
            problem_index,
            rollout_idx,
            callers,
            args,
            group_attempt,
        )
        if last.terminated_by == "stop":
            return last
        if last.terminated_by == "rejected_quality" and (
            "malformed JSON protocol" in str(last.error)
            or "protocol schema" in str(last.error)
            or "thinking mode violation" in str(last.error)
        ):
            return last
    assert last is not None
    return last


def problem_payload(problem: MathProblem) -> Dict[str, Any]:
    return {
        "id": problem.problem_id,
        "subject": problem.subject,
        "level": problem.level,
        "question": problem.prompt,
        "gold_answer": problem.gold_answer,
        "solution": problem.solution,
    }


def trajectory_record(
    problem: MathProblem,
    trajectory: MathTrajectory,
    *,
    thinking_enabled: bool,
) -> Dict[str, Any]:
    em = compute_math_em(trajectory.final_answer or "", problem.gold_answer)
    return {
        "problem": problem_payload(problem),
        "trajectory": {
            "problem_id": trajectory.problem_id,
            "rollout_idx": trajectory.rollout_idx,
            "start_agent": trajectory.start_agent,
            "steps": [asdict(step) for step in trajectory.steps],
            "final_answer": trajectory.final_answer,
            "terminated_by": trajectory.terminated_by,
            "error": trajectory.error,
            "active_agents": trajectory.active_agents,
            "n_handoffs": trajectory.n_handoffs,
            "math_eval_version": MATH_EVAL_VERSION,
            "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
            "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        },
        "start_agent": trajectory.start_agent,
        "final_answer": trajectory.final_answer,
        "terminated_by": trajectory.terminated_by,
        "em": em,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "thinking_enabled": thinking_enabled,
        "enable_thinking": thinking_enabled,
    }


def turn_records(
    problem: MathProblem,
    trajectory: MathTrajectory,
    *,
    thinking_enabled: bool,
) -> List[Dict[str, Any]]:
    em = compute_math_em(trajectory.final_answer or "", problem.gold_answer)
    output: List[Dict[str, Any]] = []
    for step, messages in zip(trajectory.steps, trajectory.turn_messages):
        protocol = {
            key: value
            for key, value in asdict(step).items()
            if key in {
                "reasoning", "tentative_answer", "action", "handoff_target",
                "handoff_note", "confirmed_answer",
            }
        }
        visible = json.dumps(protocol, ensure_ascii=False)
        if thinking_enabled:
            response = f"<think>\n{step.thinking.strip()}\n</think>\n{visible}"
        else:
            response = visible
        training_messages = [dict(message) for message in messages]
        for message in training_messages:
            if message.get("role") == "system":
                message["content"] = render_math_mas_system_prompt(
                    step.active_agent, min_agents_before_stop=1
                )
                break
        output.append({
            "problem_id": problem.problem_id,
            "rollout_idx": trajectory.rollout_idx,
            "turn": step.turn,
            "agent_id": step.active_agent,
            "messages": training_messages,
            "response": response,
            "protocol_response": visible,
            "raw_response": step.raw_output,
            "raw_outputs": step.raw_outputs,
            "thinking": step.thinking,
            "thinking_enabled": thinking_enabled,
            "enable_thinking": thinking_enabled,
            "thinking_retained_in_response": thinking_enabled and bool(step.thinking.strip()),
            "action": step.action,
            "start_agent": trajectory.start_agent,
            "terminated_by": trajectory.terminated_by,
            "final_answer": trajectory.final_answer,
            "gold_answer": problem.gold_answer,
            "subject": problem.subject,
            "level": problem.level,
            "em": em,
            "math_eval_version": MATH_EVAL_VERSION,
            "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
            "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        })
    return output


_RESUME_TERMINAL_STATUSES = frozenset(
    {"stop", "rejected_quality", "truncated", "exception"}
)


def _strict_json_loads(line: str, line_no: int) -> Dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at output line {line_no}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"output line {line_no} must contain a JSON object")
    return value


def _resume_int(value: Any, field: str, line_no: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"output line {line_no} has invalid non-negative {field}")
    return int(value)


def _resume_key(row: Dict[str, Any], mode: str, line_no: int) -> tuple[str, int]:
    if mode == "turns":
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError(f"output line {line_no} has invalid problem_id")
        return problem_id, _resume_int(row.get("rollout_idx"), "rollout_idx", line_no)
    if mode != "trajectories":
        raise ValueError(f"unsupported output mode for resume: {mode!r}")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError(f"output line {line_no} is missing trajectory object")
    problem_id = trajectory.get("problem_id")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError(f"output line {line_no} has invalid trajectory.problem_id")
    # Pre-v13 trajectory records omitted rollout_idx because they only ever
    # emitted one rollout; retain that unambiguous legacy default.
    return problem_id, _resume_int(trajectory.get("rollout_idx", 0), "rollout_idx", line_no)


def _validate_resume_row(row: Dict[str, Any], mode: str, line_no: int) -> tuple[str, int]:
    key = _resume_key(row, mode, line_no)
    terminated = row.get("terminated_by")
    if terminated not in _RESUME_TERMINAL_STATUSES:
        raise ValueError(
            f"output line {line_no} has invalid terminated_by: {terminated!r}"
    )
    if mode == "turns":
        _resume_int(row.get("turn"), "turn", line_no)
        action = row.get("action")
        # Legacy quality-failure rows sometimes had no parsed action.  They
        # remain resumable as terminal failures; successful stop rows must
        # carry the explicit confirm_stop marker (checked below as well).
        if action is not None and not isinstance(action, str):
            raise ValueError(f"output line {line_no} has invalid action")
        if terminated == "stop" and not isinstance(action, str):
            raise ValueError(f"output line {line_no} stop row has no action")
    else:
        trajectory = row["trajectory"]
        nested_terminated = trajectory.get("terminated_by")
        if nested_terminated is not None and nested_terminated != terminated:
            raise ValueError(f"output line {line_no} has inconsistent termination fields")
        steps = trajectory.get("steps")
        if steps is not None and not isinstance(steps, list):
            raise ValueError(f"output line {line_no} trajectory.steps must be a list")
        if isinstance(steps, list):
            for step_index, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise ValueError(
                        f"output line {line_no} trajectory.steps[{step_index}] must be an object"
                    )
                if step.get("turn") != step_index:
                    raise ValueError(
                        f"output line {line_no} trajectory turns are not contiguous"
                    )
                if not isinstance(step.get("action"), str):
                    raise ValueError(
                        f"output line {line_no} trajectory step has invalid action"
                    )
            if terminated == "stop" and steps and steps[-1].get("action") != "confirm_stop":
                raise ValueError(
                    f"output line {line_no} stop trajectory does not end in confirm_stop"
                )
        nested_final = trajectory.get("final_answer")
        outer_final = row.get("final_answer")
        if nested_final is not None and outer_final is not None and nested_final != outer_final:
            raise ValueError(f"output line {line_no} has inconsistent final_answer fields")
        if terminated == "stop" and not str(row.get("final_answer") or "").strip():
            raise ValueError(f"output line {line_no} stop trajectory has no final_answer")
    return key


def existing_groups(
    path: Path,
    mode: str,
    expected_keys: Optional[Set[tuple[str, int]]] = None,
) -> set[tuple[str, int]]:
    """Return complete groups and atomically remove partial resume output.

    Every non-empty complete line is validated.  An unterminated final line is
    treated as an interrupted append and discarded; malformed lines before it
    fail closed with a line number instead of silently changing the sample
    set.  ``expected_keys`` lets callers reuse an output file for a slice
    without counting rows belonging to another slice.
    """
    if mode not in {"turns", "trajectories"}:
        raise ValueError(f"unsupported output mode for resume: {mode!r}")
    if not path.exists():
        return set()
    grouped: Dict[tuple[str, int], List[tuple[Dict[str, Any], int]]] = {}
    order: List[tuple[str, int]] = []
    truncated_tail = False
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        line_no = 0
        while True:
            raw = handle.readline()
            if not raw:
                break
            line_no += 1
            complete_line = raw.endswith(b"\n")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                if not complete_line and handle.tell() == path.stat().st_size:
                    truncated_tail = True
                    break
                raise ValueError(f"invalid UTF-8 at output line {line_no}") from exc
            if not text.strip():
                if not complete_line:
                    truncated_tail = True
                    break
                continue
            try:
                row = _strict_json_loads(text, line_no)
                key = _validate_resume_row(row, mode, line_no)
            except ValueError:
                if not complete_line and handle.tell() == path.stat().st_size:
                    truncated_tail = True
                    break
                raise
            if not complete_line and handle.tell() == file_size:
                # A newline is the commit marker used by the append writer;
                # do not treat a syntactically valid but unterminated record
                # as durable after an interrupted process.
                truncated_tail = True
                break
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append((row, line_no))

    def complete_rows(entries: List[tuple[Dict[str, Any], int]]) -> bool:
        rows = [entry[0] for entry in entries]
        if mode == "trajectories":
            if len(rows) != 1:
                return False
            return rows[0].get("terminated_by") in _RESUME_TERMINAL_STATUSES
        if not rows:
            return False
        turns = [_resume_int(row.get("turn"), "turn", entries[index][1]) for index, row in enumerate(rows)]
        if len(set(turns)) != len(turns) or sorted(turns) != list(range(len(rows))):
            return False
        statuses = {row.get("terminated_by") for row in rows}
        if len(statuses) != 1 or statuses.pop() not in _RESUME_TERMINAL_STATUSES:
            return False
        rows_by_turn = sorted(rows, key=lambda row: int(row["turn"]))
        if rows_by_turn[-1].get("terminated_by") == "stop":
            return (
                rows_by_turn[-1].get("action") == "confirm_stop"
                and bool(str(rows_by_turn[-1].get("final_answer") or "").strip())
            )
        return True

    completed = {
        key
        for key, entries in grouped.items()
        if (expected_keys is None or key in expected_keys) and complete_rows(entries)
    }
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for key in order:
                entries = grouped[key]
                # A caller may reuse one output file for several disjoint
                # slices.  Only the requested slice is compacted: preserve
                # valid rows belonging to other keys instead of silently
                # deleting them during resume cleanup.
                if expected_keys is not None and key not in expected_keys:
                    rows = entries
                elif key not in completed:
                    continue
                else:
                    rows = sorted(entries, key=lambda entry: int(entry[0].get("turn", 0)))
                for row, _line_no in rows:
                    handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if truncated_tail:
        print("[resume] discarded unterminated final output record", file=sys.stderr)
    return completed


def failure_ledger_path(output: Path) -> Path:
    """Return the durable sidecar used for zero-step terminal failures."""
    return Path(f"{output}.failures.jsonl")


def _validate_failure_record(row: Dict[str, Any], line_no: int) -> tuple[str, int]:
    if isinstance(row.get("schema_version"), bool) or row.get("schema_version") != 1:
        raise ValueError(f"failure ledger line {line_no} has unsupported schema_version")
    problem_id = row.get("problem_id")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError(f"failure ledger line {line_no} has invalid problem_id")
    rollout_idx = _resume_int(row.get("rollout_idx"), "rollout_idx", line_no)
    if "rollout" in row:
        rollout = _resume_int(row["rollout"], "rollout", line_no)
        if rollout != rollout_idx:
            raise ValueError(f"failure ledger line {line_no} has inconsistent rollout fields")
    terminated = row.get("terminated_by")
    if terminated not in {"rejected_quality", "exception", "truncated"}:
        raise ValueError(f"failure ledger line {line_no} has invalid terminated_by")
    error = row.get("error")
    if not isinstance(error, str) or not error.strip():
        raise ValueError(f"failure ledger line {line_no} has no error")
    if _resume_int(row.get("n_steps"), "n_steps", line_no) != 0:
        raise ValueError(f"failure ledger line {line_no} is not a zero-step failure")
    if not isinstance(row.get("start_agent"), str) or row["start_agent"] not in AGENT_IDS:
        raise ValueError(f"failure ledger line {line_no} has invalid start_agent")
    diagnostics = row.get("attempt_diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError(f"failure ledger line {line_no} has invalid attempt_diagnostics")
    if any(not isinstance(attempt, dict) for attempt in diagnostics):
        raise ValueError(f"failure ledger line {line_no} has invalid attempt diagnostic")
    problem = row.get("problem")
    required_problem_fields = {"id", "subject", "level", "question", "gold_answer", "solution"}
    if (
        not isinstance(problem, dict)
        or set(problem) != required_problem_fields
        or problem.get("id") != problem_id
    ):
        raise ValueError(f"failure ledger line {line_no} has incomplete or inconsistent problem")
    return problem_id, rollout_idx


def _read_failure_ledger(path: Path) -> Dict[tuple[str, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[tuple[str, int], Dict[str, Any]] = {}
    with path.open("rb") as handle:
        line_no = 0
        file_size = path.stat().st_size
        while True:
            raw = handle.readline()
            if not raw:
                break
            line_no += 1
            if not raw.endswith(b"\n") and handle.tell() == file_size:
                raise ValueError(f"failure ledger has unterminated final line: {path}")
            try:
                row = _strict_json_loads(raw.decode("utf-8"), line_no)
                key = _validate_failure_record(row, line_no)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"invalid failure ledger {path}: {exc}") from exc
            if key in records:
                raise ValueError(f"failure ledger has duplicate group: {key[0]}/{key[1]}")
            records[key] = row
    return records


def failure_groups(
    path: Path, expected_keys: Optional[Set[tuple[str, int]]] = None
) -> set[tuple[str, int]]:
    records = _read_failure_ledger(path)
    if expected_keys is None:
        return set(records)
    stale = set(records) - expected_keys
    if stale:
        key = next(iter(stale))
        raise ValueError(
            "failure ledger contains groups outside selected slice: "
            f"{key[0]}/{key[1]}"
        )
    return set(records)


def validate_failure_ledger(
    path: Path,
    *,
    expected_keys: Optional[Set[tuple[str, int]]] = None,
    expected_problems: Optional[Dict[str, Dict[str, Any]]] = None,
) -> set[tuple[str, int]]:
    """Validate a ledger and reject records outside the current selection."""
    records = _read_failure_ledger(path)
    if expected_keys is not None:
        stale = set(records) - expected_keys
        if stale:
            key = next(iter(stale))
            raise ValueError(
                "failure ledger contains groups outside selected slice: "
                f"{key[0]}/{key[1]}"
            )
    if expected_problems is not None:
        for key, record in records.items():
            problem = expected_problems.get(key[0])
            if problem is not None and record["problem"] != problem:
                raise ValueError(
                    f"failure ledger problem mismatch for {key[0]}/{key[1]}"
                )
    return set(records)


def append_failure_ledger(path: Path, record: Dict[str, Any]) -> None:
    """Atomically append one validated zero-step failure record."""
    key = _validate_failure_record(record, 1)
    records = _read_failure_ledger(path)
    if key in records:
        if records[key] != record:
            raise ValueError(f"failure ledger conflict for group: {key[0]}/{key[1]}")
        return
    records[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in records.values():
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def failure_record(problem: MathProblem, trajectory: MathTrajectory) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "problem": problem_payload(problem),
        "problem_id": problem.problem_id,
        "rollout": trajectory.rollout_idx,
        "rollout_idx": trajectory.rollout_idx,
        "terminated_by": trajectory.terminated_by,
        "error": trajectory.error or "unknown terminal failure",
        "n_steps": len(trajectory.steps),
        "start_agent": trajectory.start_agent,
        "attempt_diagnostics": trajectory.failure_diagnostics,
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
    }


def main() -> None:
    args = parse_args()
    if min(args.limit, args.num_rollouts, args.t_max, args.max_concurrency) <= 0:
        raise SystemExit("limit, num-rollouts, t-max, and max-concurrency must be positive")
    if args.require_thinking and not args.enable_thinking:
        raise SystemExit("--require-thinking requires --enable-thinking")
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    problems = load_math_problems(args.data_root, split=args.split, subjects=subjects)
    selected = problems[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise SystemExit(f"requested {args.limit} problems but selected {len(selected)}")
    if args.dry_run:
        print(
            f"MATH dry-run: split={args.split} problems={len(selected)} "
            f"rollouts={args.num_rollouts} thinking={int(args.enable_thinking)} "
            f"output_mode={args.output_mode}"
        )
        return

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    bases = {"A1": args.api_base_a1, "A2": args.api_base_a2, "A3": args.api_base_a3}
    names = {"A1": args.api_model_a1, "A2": args.api_model_a2, "A3": args.api_model_a3}
    callers = {
        agent: OpenAIChatLLMCaller(
            bases[agent], names[agent], generation=generation, timeout=args.api_timeout,
            api_key=args.api_key, max_model_len=args.max_model_len,
        )
        for agent in AGENT_IDS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output exists: {args.output}; pass --resume")
    expected_keys = {
        (problem.problem_id, rollout_idx)
        for problem in selected
        for rollout_idx in range(args.num_rollouts)
    }
    completed = (
        existing_groups(args.output, args.output_mode, expected_keys=expected_keys)
        if args.resume
        else set()
    )
    failure_path = failure_ledger_path(args.output)
    if failure_path.exists() and not args.resume:
        raise SystemExit(f"failure ledger exists: {failure_path}; pass --resume")
    if args.resume and failure_path.exists():
        selected_by_key = {
            (problem.problem_id, rollout_idx): problem
            for problem in selected
            for rollout_idx in range(args.num_rollouts)
        }
        try:
            validate_failure_ledger(
                failure_path,
                expected_keys=expected_keys,
                expected_problems={
                    problem_id: problem_payload(problem)
                    for (problem_id, _rollout_idx), problem in selected_by_key.items()
                },
            )
        except ValueError as exc:
            raise SystemExit(f"invalid failure ledger on resume: {exc}") from exc
    recorded_failures = (
        failure_groups(failure_path, expected_keys=expected_keys)
        if args.resume
        else set()
    )
    overlap = completed & recorded_failures
    if overlap:
        sample = next(iter(overlap))
        raise SystemExit(
            "resume found the same group in output and failure ledger: "
            f"{sample[0]}/{sample[1]}"
        )
    completed |= recorded_failures
    jobs = [
        (problem, args.start + offset, rollout_idx)
        for offset, problem in enumerate(selected)
        for rollout_idx in range(args.num_rollouts)
        if (problem.problem_id, rollout_idx) not in completed
    ]
    print("MATH RL-MAS")
    print(
        f"  split={args.split} problems={len(selected)} rollouts={args.num_rollouts} "
        f"groups={len(selected) * args.num_rollouts} completed={len(completed)} pending={len(jobs)}"
    )
    if recorded_failures:
        print(
            f"  recorded_failures={len(recorded_failures)} ledger={failure_path}",
            file=sys.stderr,
        )
    print(
        f"  thinking={int(args.enable_thinking)} required={int(args.require_thinking)} "
        f"max_new_tokens={args.max_new_tokens} "
        f"temperature={args.temperature}"
    )
    started = time.monotonic()
    done = errors = 0
    em_values: List[float] = []
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, max(1, len(jobs)))) as pool:
            futures = {
                pool.submit(process_group, problem, index, rollout, callers, args):
                (problem, index, rollout)
                for problem, index, rollout in jobs
            }
            for future in as_completed(futures):
                problem, problem_index, rollout_idx = futures[future]
                try:
                    trajectory = future.result()
                except Exception as exc:
                    errors += 1
                    failed = MathTrajectory(
                        problem.problem_id,
                        rollout_idx,
                        start_agent_for(problem_index, args.start_agent, args.start_agent_seed),
                        terminated_by="exception",
                        error=str(exc),
                    )
                    append_failure_ledger(failure_path, failure_record(problem, failed))
                    print(
                        f"[error] {problem.problem_id}: "
                        f"worker exception: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if args.output_mode == "turns":
                    rows = turn_records(
                        problem,
                        trajectory,
                        thinking_enabled=args.enable_thinking,
                    )
                else:
                    rows = [
                        trajectory_record(
                            problem,
                            trajectory,
                            thinking_enabled=args.enable_thinking,
                        )
                    ]
                # A terminal quality/service failure is a completed group in
                # trajectory mode and must be persisted so --resume cannot
                # repeatedly regenerate the same bad response.  Turn mode
                # has no representable zero-turn row; partial trajectories
                # still retain all accepted turns, while a zero-turn failure
                # remains an explicit rerun error for the caller to inspect.
                if not rows:
                    errors += 1
                    append_failure_ledger(failure_path, failure_record(problem, trajectory))
                    print(
                        f"[error] {problem.problem_id}/{trajectory.rollout_idx}: "
                        f"{trajectory.terminated_by}: {trajectory.error}",
                        file=sys.stderr,
                    )
                    continue
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
                handle.flush()
                done += 1
                em_values.append(compute_math_em(trajectory.final_answer or "", problem.gold_answer))
                if done == 1 or done % 50 == 0 or done + errors == len(jobs):
                    rate = (done + errors) / max(time.monotonic() - started, 1e-9)
                    current_em = mean(em_values) if em_values else 0.0
                    print(
                        f"  progress={done + errors}/{len(jobs)} errors={errors} "
                        f"EM={current_em:.4f} rate={rate:.2f}/s",
                        flush=True,
                    )
    if errors:
        recorded_failures = failure_groups(failure_path, expected_keys=expected_keys)
        raise SystemExit(
            f"sampling incomplete: {len(recorded_failures)} groups failed; "
            f"failure ledger: {failure_path}; rerun with --resume to skip recorded failures"
        )
    recorded_failures = failure_groups(failure_path, expected_keys=expected_keys)
    if recorded_failures:
        raise SystemExit(
            f"sampling incomplete: {len(recorded_failures)} groups recorded as failures; "
            f"failure ledger: {failure_path}"
        )
    print(f"Done. groups={len(completed) + done}; new_EM={mean(em_values) if em_values else 0.0:.4f}")


if __name__ == "__main__":
    main()
