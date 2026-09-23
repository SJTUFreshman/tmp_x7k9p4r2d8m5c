"""Code-specific judge for MultiPL-E MAS trajectories.

This module mirrors the small public API used by the existing RL pipeline:
``judge_trajectory(result, *, model) -> Optional[TrajectoryReward]``.

The caller combines the returned per-turn process scores with execution
success. This judge only evaluates whether each agent's reasoning and protocol
action helped produce a correct code continuation.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT.parent
JUDGE_MODEL = "gpt-4o-mini"
CORRECT_BONUS = 0.2
JUDGE_PROMPT_VERSION = "multipl_e_code_process_quality_v1"
_FAILURE_LOG_LOCK = Lock()

for import_root in (PACKAGE_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_client import call_llm_stream  # type: ignore  # noqa: E402


@dataclass
class TurnScore:
    turn: int
    agent_id: str
    reasoning_score: float
    action_score: float
    comment: str = ""

    @property
    def judge_score(self) -> float:
        return 0.5 * self.reasoning_score + 0.5 * self.action_score


@dataclass
class TrajectoryReward:
    problem_id: str
    r_task: float
    turn_scores: List[TurnScore]


_JUDGE_SYSTEM = """You are an expert judge for MultiPL-E code-completion multi-agent trajectories.

You will receive:
1. A target programming language.
2. The source-code prefix ending at the completion cursor.
3. Whether the final submitted completion passed the execution tests.
4. A turn-by-turn MAS trajectory with reasoning, tentative completions,
   handoff notes, confirmations, and protocol status.

Judge PROCESS QUALITY for each turn, not just final correctness.

Score every turn on two dimensions in [-1.0, +1.0]:

reasoning_score:
  +1.0: clearly identifies the needed implementation, edge cases, syntax, and evaluator boundary.
  +0.7: mostly correct reasoning with minor omissions.
  +0.3: plausible but shallow or incomplete reasoning.
   0.0: generic restatement with little useful analysis.
  -0.5: materially confused reasoning or missed important syntax/boundary issues.
  -1.0: reasoning would strongly mislead the team.

action_score:
  For a proposer or correcting turn:
    +1.0: proposes a correct/helpful complete continuation or a clearly useful correction.
    +0.7: candidate is directionally useful but may miss minor details.
    +0.2: candidate is incomplete but not harmful.
    -0.5: candidate is likely wrong or repeats the prefix/function header against instructions.
    -1.0: candidate is invalid, empty when needed, or severely harmful.

  For a verifier/confirmation turn:
    +1.0: catches a real bug and fixes it, or confirms a correct candidate with meaningful audit.
    +0.5: confirms a likely correct candidate with limited but relevant audit.
     0.0: passive confirmation with no real verification.
    -0.6: blindly confirms a bad candidate or makes unnecessary confused changes.
    -1.0: violates protocol or turns a good candidate into a bad one.

Important protocol checks:
- Code fields must contain only text appended at the cursor.
- Repeating the full source prefix or function declaration is usually bad.
- Respect whether the evaluator supplies the final delimiter.
- confirm_stop should confirm a previous candidate, not introduce a new one.
- Handoffs should be useful and problem-specific, not mechanical.

Output ONLY a JSON array with one object per turn, in turn order:
[
  {
    "turn": 0,
    "agent_id": "A1",
    "reasoning_score": 0.70,
    "action_score": 0.60,
    "comment": "brief concrete explanation"
  }
]
Use precise decimals and avoid assigning the same score to every turn."""


def _truncate(text: Any, limit: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def _format_steps(steps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    previous_candidate = ""
    for step in sorted(steps, key=lambda s: int(s.get("turn", 0))):
        turn = step.get("turn", "?")
        agent = step.get("active_agent", "?")
        action = step.get("action", "?")
        lines.append(f"[Turn {turn}] agent={agent} action={action} status={step.get('status', '')}")
        if previous_candidate:
            lines.append("previous_candidate:")
            lines.append(_truncate(previous_candidate, 1200))
        reasoning = (step.get("reasoning") or "").strip()
        if reasoning:
            lines.append("reasoning:")
            lines.append(_truncate(reasoning, 1200))
        tentative = step.get("tentative_completion") or ""
        if tentative:
            lines.append("tentative_completion:")
            lines.append(_truncate(tentative, 1800))
            previous_candidate = str(tentative)
        confirmed = step.get("confirmed_completion") or ""
        if confirmed:
            lines.append("confirmed_completion:")
            lines.append(_truncate(confirmed, 1800))
        note = step.get("handoff_note") or ""
        if step.get("handoff_target") or note:
            lines.append(f"handoff_target={step.get('handoff_target')} note={_truncate(note, 600)}")
        lines.append(f"candidate_changed={bool(step.get('candidate_changed'))} passed_this_turn={bool(step.get('passed_this_turn'))}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_user_prompt(result: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
    return (
        f"# Problem\n"
        f"problem_id: {result.get('problem_id', '')}\n"
        f"root_dataset: {result.get('root_dataset', '')}\n"
        f"language: {result.get('language', '')}\n"
        f"execution_passed: {'YES' if result.get('passed') else 'NO'}\n\n"
        f"# Source Prefix\n{_truncate(result.get('source_prefix', ''), 5000)}\n\n"
        f"# Trajectory\n{_format_steps(steps)}\n\n"
        f"Score every turn in the trajectory."
    )


def _extract_json_array(content: str) -> Optional[List[Any]]:
    text = content.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
        if not match:
            return None
        text = match.group(1)
    else:
        text = text[start:end]
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_turn_scores(content: str, steps: List[Dict[str, Any]]) -> Optional[List[TurnScore]]:
    raw_scores = _extract_json_array(content)
    if not raw_scores:
        return None

    expected_agents = {int(step.get("turn", 0)): str(step.get("active_agent", "")) for step in steps}
    if len(raw_scores) != len(expected_agents):
        return None
    seen_turns = set()
    for obj in raw_scores:
        if not isinstance(obj, dict):
            return None
        turn = obj.get("turn")
        if type(turn) is not int or turn not in expected_agents or turn in seen_turns:
            return None
        if obj.get("agent_id") != expected_agents[turn]:
            return None
        for field in ("reasoning_score", "action_score"):
            value = obj.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or not -1.0 <= value <= 1.0:
                return None
        if not isinstance(obj.get("comment"), str):
            return None
        seen_turns.add(turn)

    valid_turns = {int(step.get("turn", 0)) for step in steps}
    scores: List[TurnScore] = []
    for obj in raw_scores:
        if not isinstance(obj, dict):
            continue
        try:
            turn = int(obj.get("turn", 0))
            if turn not in valid_turns:
                continue
            reasoning_score = max(-1.0, min(1.0, float(obj.get("reasoning_score", 0.0))))
            action_score = max(-1.0, min(1.0, float(obj.get("action_score", 0.0))))
            scores.append(
                TurnScore(
                    turn=turn,
                    agent_id=str(obj.get("agent_id", "")),
                    reasoning_score=reasoning_score,
                    action_score=action_score,
                    comment=str(obj.get("comment", "")),
                )
            )
        except Exception:
            continue

    if not scores:
        return None
    by_turn = {score.turn: score for score in scores}
    ordered = [by_turn[turn] for turn in sorted(valid_turns) if turn in by_turn]
    return ordered or None


def _retry_response_format(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    turns = sorted({int(step.get("turn", 0)) for step in steps})
    agents = sorted({str(step.get("active_agent", "")) for step in steps})
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "multipl_e_turn_scores",
            "strict": True,
            "schema": {
                "type": "array",
                "minItems": len(turns),
                "maxItems": len(turns),
                "items": {
                    "type": "object",
                    "properties": {
                        "turn": {"type": "integer", "enum": turns},
                        "agent_id": {"type": "string", "enum": agents},
                        "reasoning_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                        "action_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                        "comment": {"type": "string"},
                    },
                    "required": ["turn", "agent_id", "reasoning_score", "action_score", "comment"],
                    "additionalProperties": False,
                },
            },
        },
    }


def _record_judge_failure(
    result: Dict[str, Any],
    *,
    model: str,
    attempt: int,
    reason: str,
    content: str = "",
    error: Optional[str] = None,
) -> None:
    destination = os.environ.get("JCA_CODE_JUDGE_FAILURE_LOG")
    if not destination:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root_dataset": result.get("root_dataset"),
        "language": result.get("language"),
        "problem_id": result.get("problem_id"),
        "rollout_idx": result.get("rollout_idx"),
        "model": model,
        "attempt": attempt,
        "reason": reason,
        "content": content,
        "error": error,
    }
    path = Path(destination)
    with _FAILURE_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def judge_trajectory(
    result: Dict[str, Any],
    *,
    model: str = JUDGE_MODEL,
    max_tokens: int = 8192,
    reasoning_effort: str = "",
    temperature: float = 0.0,
    parse_retries: int = 2,
) -> Optional[TrajectoryReward]:
    steps = result.get("steps") or []
    if not steps:
        return None

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _build_user_prompt(result, steps)},
    ]

    attempts = max(0, int(parse_retries)) + 1
    for attempt in range(attempts):
        try:
            response = call_llm_stream(
                messages=messages,
                tools=None,
                model=model,
                temperature=temperature,
                top_p=float(os.environ.get("JCA_JUDGE_TOP_P", "0.95")),
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or None,
                enable_thinking=(
                    False if os.environ.get("JCA_JUDGE_DISABLE_THINKING") == "1"
                    else True if os.environ.get("JCA_JUDGE_ENABLE_THINKING") == "1"
                    else None
                ),
                **({"response_format": _retry_response_format(steps)} if attempt else {}),
            )
        except Exception as exc:
            _record_judge_failure(
                result, model=model, attempt=attempt + 1,
                reason="request_error", error=f"{type(exc).__name__}: {exc}",
            )
            return None

        content = (response.get("content") or "").strip()
        scores = _parse_turn_scores(content, steps)
        if scores is not None:
            if result.get("passed"):
                for score in reversed(scores):
                    matched = next(
                        (step for step in steps if int(step.get("turn", 0)) == score.turn),
                        None,
                    )
                    if matched and matched.get("action") == "confirm_stop":
                        score.action_score = min(1.0, score.action_score + CORRECT_BONUS)
                        break
            return TrajectoryReward(
                problem_id=str(result.get("problem_id", "")),
                r_task=1.0 if result.get("passed") else 0.0,
                turn_scores=scores,
            )
        _record_judge_failure(
            result, model=model, attempt=attempt + 1,
            reason="invalid_turn_scores", content=content,
        )
        messages.append({
            "role": "user",
            "content": (
                "Your previous response could not be parsed as the required JSON array. "
                "Evaluate the same trajectory using the unchanged scoring rubric. "
                "Return only a valid JSON array with one object for every requested turn, "
                "using turn, agent_id, reasoning_score, action_score, and comment. "
                "Use JSON numbers for both scores, double-quoted strings, escaped quotes "
                "inside comments, no trailing commas, and no text or Markdown outside the array."
                " Keep comments concise plain prose; describe code behavior without quoting code, "
                "regular expressions, backslash escapes, or control characters."
            ),
        })

    return None
