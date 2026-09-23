"""JCA Judge: per-turn process-quality scoring via LLM.

Given a full trajectory + gold answer + final F1, the judge scores
each turn on two process dimensions in [-1, +1]:
    reasoning_score : cites paragraphs, logical chain valid
    action_score    : was handoff / confirm_stop appropriate

judge_score(turn) = 0.5 * reasoning + 0.5 * action

The final RL reward is combined by the caller (rl_rollout.py):
    reward(turn) = alpha * (2*F1 - 1) + (1 - alpha) * judge_score(turn)

Final confirm_stop turn gets +CORRECT_BONUS on action_score if F1 > 0.

Public API:
    TurnScore
    TrajectoryReward
    judge_trajectory(result_dict, *, model) -> Optional[TrajectoryReward]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JUDGE_MODEL            = "gpt-5"
JUDGE_TEMPERATURE      = 0.0
JUDGE_TOP_P            = 0.95
JUDGE_MAX_TOKENS       = 16384
JUDGE_REASONING_EFFORT = "low"
JUDGE_PARSE_RETRIES    = 2
ALPHA                  = 0.5   # weight on task_reward vs judge_score
CORRECT_BONUS          = 0.3   # added to action_score of final confirm_stop when correct

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from llm_client import call_llm_stream  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    def call_llm_stream(
        *,
        messages: List[Dict[str, str]],
        tools: Any = None,
        model: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        top_p: float = JUDGE_TOP_P,
        max_tokens: int = JUDGE_MAX_TOKENS,
        reasoning_effort: Optional[str] = JUDGE_REASONING_EFFORT,
    ) -> Dict[str, Any]:
        """Small OpenAI-compatible fallback used when project llm_client is absent."""
        api_base = (
            os.environ.get("JCA_JUDGE_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        api_key = (
            os.environ.get("JCA_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_KEY")
            or "EMPTY"
        )
        timeout = float(os.environ.get("JCA_JUDGE_TIMEOUT", "180"))
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools is not None:
            payload["tools"] = tools

        request = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"judge request HTTP {exc.code}: {body}") from exc
        data = json.loads(raw)
        content = data["choices"][0]["message"].get("content", "")
        return {"content": content, "raw": data}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnScore:
    turn:            int
    agent_id:        str
    reasoning_score: float   # [-1, 1] process quality: paragraph citations, logic
    action_score:    float   # [-1, 1] process quality: handoff/confirm_stop timing
    comment:         str = ""

    @property
    def judge_score(self) -> float:
        """Process-quality composite in [-1, 1]. Does NOT include answer correctness
        — that comes from F1 (task reward), combined at the rollout level."""
        return 0.5 * self.reasoning_score + 0.5 * self.action_score


@dataclass
class TrajectoryReward:
    """Container for per-turn judge scores of one trajectory.

    Consumers (rl_rollout.py) combine turn_scores with task_reward (2*F1-1)
    at the per-turn granularity to produce final RL reward.
    """
    problem_id:  str
    r_task:      float              # raw F1 in [0, 1]
    turn_scores: List[TurnScore]


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an expert judge evaluating a multi-agent QA system.

You will be given:
  1. A multi-hop question with supporting paragraphs
  2. The gold answer
  3. A multi-turn agent trajectory
  4. Whether the final answer was correct

BEFORE scoring each turn, do this internally:
  Step 1: Read ALL paragraphs carefully. Derive the correct answer yourself.
  Step 2: For turn 0 (proposer): compare agent's tentative to your derived answer.
  Step 3: For turn > 0 (verifier): compare agent's tentative to the PREVIOUS tentative AND to your answer.
  Step 4: Ask: did this agent's action help, hurt, or do nothing for the team?

You judge PROCESS QUALITY only. Use PRECISE decimal values — not just round numbers.
Good scores: 0.3, 0.6, 0.7, -0.3, -0.4, -0.6, -0.7. Avoid defaulting to ±0.5/±1.0 for everything.

Score EACH turn on two dimensions in [-1.0, +1.0]:

  reasoning_score — quality of the reasoning chain
    +1.0 = cites correct paragraph numbers, logic is sound, multi-hop chain is complete
    +0.7 = mostly correct, minor gap or imprecise citation
    +0.5 = partially correct, one hop missing or citation vague
     0.0 = restates question or no citations
    -0.5 = wrong paragraph cited, partially misleading
    -1.0 = completely wrong, cites irrelevant paragraphs, would mislead next agent

  action_score — was the action NECESSARY and did it HELP?

    For PROPOSER (turn = 0): first derive answer yourself, then compare.
      +1.0 = tentative matches your derived answer AND reasoning is solid
      +0.7 = tentative direction correct but reasoning imprecise
      +0.3 = tentative exists, direction uncertain but reasonable attempt
       0.0 = tentative is vague or empty-ish (e.g. "I cannot determine")
      -0.4 = no tentative given at all (blank field)
      -0.8 = tentative is clearly wrong direction with bad reasoning
      -1.0 = no tentative AND harmful/confused reasoning

    For VERIFIER (turn > 0): first check if previous tentative was correct.

      If previous tentative MATCHES gold:
        +0.8 = confirms with new independent evidence (adds value)
        +0.4 = confirms without adding new evidence (ok but passive)
         0.0 = passes through mechanically, no verification
        -0.6 = unnecessarily challenges a correct answer without evidence
        -1.0 = changes correct answer to wrong one (misfire — very harmful)

      If previous tentative was WRONG:
        +1.0 = identifies error, provides correct answer with clear evidence
        +0.7 = identifies error, partially corrected, evidence present
        +0.3 = notices something off but correction imprecise
         0.0 = fails to catch the error, passes through
        -0.6 = blindly confirms the wrong answer
        -1.0 = confirms wrong answer AND adds more wrong reasoning

      If correct answer was ALREADY ESTABLISHED (redundant turn):
        -0.3 to -0.6 depending on how redundant (minor duplication vs. full loop)

  KEY RULE: Do NOT give action_score > 0 just because reasoning looks good.
  The question is: did this specific action CONTRIBUTE to getting the right answer?

Output a JSON array with one object per turn, in order:
[
  {
    "turn": 0,
    "agent_id": "A1",
    "reasoning_score": 0.70,
    "action_score": 0.60,
    "comment": "one sentence explaining what the agent did right/wrong"
  },
  ...
]
Output ONLY the JSON array. No other text."""


def _format_paragraphs(problem: Dict[str, Any]) -> str:
    paras = problem.get("paragraphs", [])
    if not paras:
        return problem.get("paragraphs_text", "(not available)")

    lines = []
    for i, p in enumerate(paras):
        # MuSiQueProblem.paragraphs is list[tuple[title, text]];
        # eval results.jsonl stores dicts with 'title'/'paragraph_text'.
        if isinstance(p, dict):
            title = p.get("title", "")
            text  = p.get("paragraph_text", "")
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            title, text = p[0], p[1]
        else:
            title, text = "", str(p)
        lines.append(f"[{i+1}] ({title}) {text}")
    return "\n".join(lines)


def _format_steps(steps: List[Dict[str, Any]]) -> str:
    lines = []
    prev_tentative = None
    for s in steps:
        turn      = s.get("turn", "?")
        agent     = s.get("active_agent", "?")
        action    = s.get("action", "?")
        reasoning = (s.get("reasoning") or "").strip()[:400]
        tentative = (s.get("tentative_answer") or "").strip()
        handoff   = s.get("handoff_target")
        note      = (s.get("handoff_note") or "").strip()
        confirmed = s.get("confirmed_answer") or s.get("final_answer")

        lines.append(f"[Turn {turn}] Agent={agent}  action={action}")
        if reasoning:
            lines.append(f"  reasoning: {reasoning}")
        if tentative:
            lines.append(f"  tentative_answer: {tentative}")
        # Show what the previous tentative was so judge can evaluate verifier behavior
        if prev_tentative and turn > 0:
            lines.append(f"  [previous tentative was: {prev_tentative}]")
        if handoff:
            line = f"  → handoff to {handoff}"
            if note:
                line += f": {note}"
            lines.append(line)
        if confirmed and str(confirmed) not in ("None", ""):
            lines.append(f"  → confirmed_answer: {confirmed}")

        if tentative:
            prev_tentative = tentative
    return "\n".join(lines)


def _build_user_prompt(
    problem:      Dict[str, Any],
    steps:        List[Dict[str, Any]],
    final_answer: str,
    is_correct:   bool,
) -> str:
    return (
        f"# Question\n{problem.get('question','')}\n\n"
        f"# Paragraphs\n{_format_paragraphs(problem)}\n\n"
        f"# Gold Answer\n{problem.get('answer','')}\n\n"
        f"# Trajectory\n{_format_steps(steps)}\n\n"
        f"# Final Answer: {final_answer}\n"
        f"# Correct: {'YES (F1 > 0)' if is_correct else 'NO'}\n\n"
        f"Score each turn."
    )


# ---------------------------------------------------------------------------
# Core judge call
# ---------------------------------------------------------------------------

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
    """Score one trajectory. Returns TrajectoryReward with per-turn scores.

    The caller (rl_rollout) combines these with task reward (2*F1-1) at
    per-turn granularity to produce the final RL signal.

    Returns None on judge API failure.
    """
    problem = result.get("problem", {})
    traj    = result.get("trajectory", {})
    steps   = traj.get("steps", [])

    if not steps:
        return None

    final_answer = result.get("final_answer", "")
    f1           = float(result.get("f1", 0.0))
    is_correct   = f1 > 0.0

    user_prompt = _build_user_prompt(
        problem=problem,
        steps=steps,
        final_answer=final_answer,
        is_correct=is_correct,
    )

    raw_scores = None
    for _attempt in range(max(0, parse_retries) + 1):
        try:
            resp = call_llm_stream(
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                tools=None,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort or None,
            )
            content = (resp.get("content") or "").strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start < 0 or end <= start:
                continue
            candidate = json.loads(content[start:end])
            if isinstance(candidate, list) and candidate:
                raw_scores = candidate
                break
        except Exception:
            continue

    if raw_scores is None:
        return None

    turn_scores: List[TurnScore] = []
    for obj in raw_scores:
        if not isinstance(obj, dict):
            continue
        try:
            ts = TurnScore(
                turn             = int(obj.get("turn", 0)),
                agent_id         = str(obj.get("agent_id", "")),
                reasoning_score  = float(obj.get("reasoning_score", 0.0)),
                action_score     = float(obj.get("action_score", 0.0)),
                comment          = str(obj.get("comment", "")),
            )
            # Clamp scores to [-1, 1]
            ts.reasoning_score = max(-1.0, min(1.0, ts.reasoning_score))
            ts.action_score    = max(-1.0, min(1.0, ts.action_score))
            turn_scores.append(ts)
        except Exception:
            continue

    if not turn_scores:
        return None

    # Add correct_bonus to the last confirm_stop turn when answer is correct
    if is_correct:
        for ts in reversed(turn_scores):
            matched_step = next(
                (s for s in steps if str(s.get("turn", "")) == str(ts.turn)), None
            )
            if matched_step and matched_step.get("action") == "confirm_stop":
                ts.action_score = min(1.0, ts.action_score + CORRECT_BONUS)
                break

    problem_id = problem.get("id") or traj.get("problem_id", "")
    return TrajectoryReward(
        problem_id  = problem_id,
        r_task      = f1,
        turn_scores = turn_scores,
    )
