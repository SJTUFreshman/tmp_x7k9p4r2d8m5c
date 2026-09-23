"""Collaboration-focused LLM judge for GSM-HARD offline RL v3."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jca.llm_client import call_llm_stream


JUDGE_MODEL = "gpt-5"
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 0.95
JUDGE_MAX_TOKENS = 16384
JUDGE_REASONING_EFFORT = "low"
JUDGE_PARSE_RETRIES = 2


@dataclass
class CollaborationTurnScore:
    turn: int
    agent_id: str
    reasoning_score: float
    verification_score: float
    action_score: float
    route_score: float
    previous_answer_correct: Optional[bool]
    current_answer_correct: bool
    answer_changed: bool
    independent_verification: bool
    recommended_action: str
    comment: str = ""

    @property
    def judge_score(self) -> float:
        if self.turn == 0:
            return (
                0.50 * self.reasoning_score
                + 0.30 * self.action_score
                + 0.20 * self.route_score
            )
        return (
            0.35 * self.reasoning_score
            + 0.30 * self.verification_score
            + 0.20 * self.action_score
            + 0.15 * self.route_score
        )


@dataclass
class CollaborationTrajectoryScore:
    problem_id: str
    turn_scores: List[CollaborationTurnScore]


GSM_COLLABORATION_JUDGE_SYSTEM = """You are an expert process judge for a three-agent
GSM-style math system. The experiment always starts with A1, but A1, A2, and A3
have equal status after that. Never reward or penalize a route merely because
the target is A2 rather than A3, or vice versa.

Derive the trusted numeric answer yourself before scoring. For every turn after
turn 0, first determine whether the previous tentative answer was correct, then
determine whether the current agent independently recomputed the original
problem. Copying, paraphrasing, or rubber-stamping the previous chain is not an
independent verification.

Score each dimension in [-1.0, 1.0] using precise decimal values.

reasoning_score:
  +1.0 correct setup, arithmetic, units, signs, time spans, percentages, and rounding
   0.0 unauditable or irrelevant computation
  -1.0 materially wrong or misleading computation

verification_score:
  +1.0 independently repairs a wrong previous answer or independently confirms it
  +0.4 useful partial audit that identifies the critical check
   0.0 no independent check
  -0.7 rubber-stamps a wrong answer or claims verification without doing it
  -1.0 changes an established correct answer to a wrong answer
  Use 0.0 on turn 0 because there is no previous answer to verify.

action_score:
  Turn 0 must provide a concrete tentative answer and hand off for verification.
  If a verifier changes the tentative answer, confirm_stop is forbidden on that
  same turn; the corrected answer must be handed off for a later check.
  If the previous answer is correct and the current agent independently obtains
  the same answer, confirm_stop is usually appropriate.
  Penalize premature stopping, failure to pass on a correction, unnecessary
  extra handoffs, and changing a correct answer to a wrong one.

route_score:
  Reward a handoff only when the next turn is useful for checking an unresolved
  computation or a newly corrected answer. Penalize redundant loops, repeatedly
  bouncing an already established answer, or routing that does not add checking
  value. A2 and A3 are equally valid first handoff targets.

The score must reflect this turn's causal contribution, not final trajectory
correctness or problem difficulty. Good-looking reasoning alone does not justify
a positive action or route score.

Output only one JSON array with exactly one object per turn:
[
  {
    "turn": 0,
    "agent_id": "A1",
    "reasoning_score": 0.8,
    "verification_score": 0.0,
    "action_score": 0.7,
    "route_score": 0.6,
    "previous_answer_correct": null,
    "current_answer_correct": true,
    "answer_changed": false,
    "independent_verification": false,
    "recommended_action": "handoff",
    "comment": "brief evidence-based explanation"
  }
]

recommended_action must be exactly "handoff" or "confirm_stop". Output no
markdown and no text outside the JSON array."""


def _format_steps(steps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    previous_tentative: Optional[str] = None
    for step in steps:
        turn = step.get("turn", "?")
        agent = step.get("active_agent", "?")
        reasoning = str(step.get("reasoning") or "").strip()
        tentative = str(step.get("tentative_answer") or "").strip()
        action = str(step.get("action") or "?")
        target = step.get("handoff_target")
        note = str(step.get("handoff_note") or "").strip()
        confirmed = step.get("confirmed_answer")

        lines.append(f"[Turn {turn}] agent={agent} action={action}")
        if previous_tentative is not None:
            lines.append(f"  previous_tentative_answer: {previous_tentative}")
        lines.append(f"  reasoning: {reasoning or '(empty)'}")
        lines.append(f"  tentative_answer: {tentative or '(empty)'}")
        if target:
            lines.append(f"  handoff_target: {target}")
        if note:
            lines.append(f"  handoff_note: {note}")
        if confirmed is not None:
            lines.append(f"  confirmed_answer: {confirmed}")
        if tentative:
            previous_tentative = tentative
    return "\n".join(lines)


def build_judge_prompt(result: Dict[str, Any]) -> str:
    problem = result.get("problem", {})
    trajectory = result.get("trajectory", {})
    return (
        f"# Problem\n{problem.get('question', '')}\n\n"
        f"# Trusted Numeric Answer\n{problem.get('answer', '')}\n\n"
        f"# Trajectory\n{_format_steps(trajectory.get('steps', []))}\n\n"
        f"# Final Answer\n{result.get('final_answer') or '(missing)'}\n\n"
        "Score every turn independently and audit the collaboration state."
    )


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError("expected bool or null")


def _required_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("expected bool")


def parse_turn_scores(
    content: str,
    expected_steps: List[Dict[str, Any]],
) -> Optional[List[CollaborationTurnScore]]:
    start = (content or "").find("[")
    end = (content or "").rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        raw_scores = json.loads(content[start:end])
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_scores, list):
        return None

    expected = {
        int(step["turn"]): str(step["active_agent"])
        for step in expected_steps
    }
    parsed: Dict[int, CollaborationTurnScore] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            return None
        try:
            score = CollaborationTurnScore(
                turn=int(item["turn"]),
                agent_id=str(item["agent_id"]),
                reasoning_score=float(item["reasoning_score"]),
                verification_score=float(item["verification_score"]),
                action_score=float(item["action_score"]),
                route_score=float(item["route_score"]),
                previous_answer_correct=_optional_bool(
                    item["previous_answer_correct"]
                ),
                current_answer_correct=_required_bool(
                    item["current_answer_correct"]
                ),
                answer_changed=_required_bool(item["answer_changed"]),
                independent_verification=_required_bool(
                    item["independent_verification"]
                ),
                recommended_action=str(item["recommended_action"]),
                comment=str(item.get("comment", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if score.turn in parsed or expected.get(score.turn) != score.agent_id:
            return None
        if score.recommended_action not in {"handoff", "confirm_stop"}:
            return None
        for field_name in (
            "reasoning_score",
            "verification_score",
            "action_score",
            "route_score",
        ):
            value = getattr(score, field_name)
            setattr(score, field_name, max(-1.0, min(1.0, value)))
        parsed[score.turn] = score

    if set(parsed) != set(expected):
        return None
    return [parsed[turn] for turn in sorted(parsed)]


def judge_trajectory(
    result: Dict[str, Any],
    *,
    model: str = JUDGE_MODEL,
    temperature: float = JUDGE_TEMPERATURE,
    top_p: float = JUDGE_TOP_P,
    max_tokens: int = JUDGE_MAX_TOKENS,
    reasoning_effort: str = JUDGE_REASONING_EFFORT,
    parse_retries: int = JUDGE_PARSE_RETRIES,
) -> Optional[CollaborationTrajectoryScore]:
    steps = result.get("trajectory", {}).get("steps", [])
    if not steps:
        return None

    prompt = build_judge_prompt(result)
    turn_scores: Optional[List[CollaborationTurnScore]] = None
    for _attempt in range(parse_retries + 1):
        try:
            response = call_llm_stream(
                messages=[
                    {
                        "role": "system",
                        "content": GSM_COLLABORATION_JUDGE_SYSTEM,
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or None,
            )
        except Exception:
            continue
        turn_scores = parse_turn_scores(
            str(response.get("content") or ""),
            steps,
        )
        if turn_scores is not None:
            break

    if turn_scores is None:
        return None
    problem = result.get("problem", {})
    trajectory = result.get("trajectory", {})
    return CollaborationTrajectoryScore(
        problem_id=str(problem.get("id") or trajectory.get("problem_id") or ""),
        turn_scores=turn_scores,
    )
