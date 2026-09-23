"""LLM process judge for GSM-HARD multi-agent trajectories."""
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
CORRECT_BONUS = 0.3


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
    task_em: float
    turn_scores: List[TurnScore]


GSM_JUDGE_SYSTEM = """You are an expert process judge for a multi-agent math system.

You receive a GSM-style word problem, its trusted numeric answer, and the full
agent trajectory. Derive the answer yourself before scoring. Score every turn
on two dimensions in [-1.0, 1.0].

reasoning_score:
  +1.0: correct setup, arithmetic, units, sign, time span, percentages, and rounding
  +0.7: correct result and method with a minor presentation gap
   0.0: no auditable computation or mostly irrelevant work
  -0.7: substantive arithmetic or modeling error
  -1.0: entirely wrong or misleading reasoning

action_score:
  For the first proposer, reward a concrete tentative answer and a useful handoff.
  For a verifier, first decide whether the previous tentative answer was correct.
  Reward an independent recomputation that catches and clearly corrects an error.
  Reward an independent confirmation of a correct answer less strongly.
  Penalize rubber-stamping, changing a correct answer to a wrong one, failing to
  catch a clear error, premature confirm_stop, unnecessary repeated handoffs, or
  violations of the three-agent collaboration protocol.

Judge process quality rather than copying the final correctness label. Use precise
decimal scores. Output only one JSON array with exactly one object per turn:
[
  {
    "turn": 0,
    "agent_id": "A1",
    "reasoning_score": 0.7,
    "action_score": 0.6,
    "comment": "brief explanation"
  }
]
"""


def _format_steps(steps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    previous_tentative: Optional[str] = None
    for step in steps:
        turn = step.get("turn", "?")
        agent = step.get("active_agent", "?")
        reasoning = str(step.get("reasoning") or "").strip()
        tentative = str(step.get("tentative_answer") or "").strip()
        action = step.get("action", "?")
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
        f"# Exact Match\n{'YES' if float(result.get('em', 0.0) or 0.0) == 1.0 else 'NO'}\n\n"
        "Score every turn."
    )


def parse_turn_scores(
    content: str,
    expected_steps: List[Dict[str, Any]],
) -> Optional[List[TurnScore]]:
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
    parsed: Dict[int, TurnScore] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            return None
        try:
            score = TurnScore(
                turn=int(item["turn"]),
                agent_id=str(item["agent_id"]),
                reasoning_score=float(item["reasoning_score"]),
                action_score=float(item["action_score"]),
                comment=str(item.get("comment", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if score.turn in parsed or expected.get(score.turn) != score.agent_id:
            return None
        score.reasoning_score = max(-1.0, min(1.0, score.reasoning_score))
        score.action_score = max(-1.0, min(1.0, score.action_score))
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
) -> Optional[TrajectoryReward]:
    steps = result.get("trajectory", {}).get("steps", [])
    if not steps:
        return None

    prompt = build_judge_prompt(result)
    turn_scores: Optional[List[TurnScore]] = None
    for _attempt in range(parse_retries + 1):
        try:
            response = call_llm_stream(
                messages=[
                    {"role": "system", "content": GSM_JUDGE_SYSTEM},
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

    em = float(result.get("em", 0.0) or 0.0)
    if em == 1.0:
        for score in reversed(turn_scores):
            step = next(
                (candidate for candidate in steps if int(candidate["turn"]) == score.turn),
                None,
            )
            if step and step.get("action") == "confirm_stop":
                score.action_score = min(1.0, score.action_score + CORRECT_BONUS)
                break

    problem = result.get("problem", {})
    trajectory = result.get("trajectory", {})
    return TrajectoryReward(
        problem_id=str(problem.get("id") or trajectory.get("problem_id") or ""),
        task_em=em,
        turn_scores=turn_scores,
    )
