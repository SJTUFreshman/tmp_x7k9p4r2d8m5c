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
JUDGE_MODEL   = "gpt-5"
ALPHA         = 0.5   # weight on task_reward vs judge_score
CORRECT_BONUS = 0.3   # added to action_score of final confirm_stop when correct

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
        }
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

BEFORE scoring, do the following internally:
  Step 1: Read the paragraphs and derive the correct answer yourself.
  Step 2: For each agent turn, compare the agent's tentative_answer to your derived answer.
  Step 3: Judge action_score strictly based on whether the agent's action was NECESSARY and CORRECT.

You judge PROCESS QUALITY only.

Score EACH turn on two dimensions. Each score is SIGNED in [-1.0, +1.0]:

  reasoning_score
    +1.0 = clear reasoning with specific paragraph citations, logically valid chain
     0.0 = mediocre -- vague, restates without citing evidence
    -1.0 = harmful -- wrong reasoning, cites wrong paragraphs, misleads next agent

  action_score  (BE STRICT -- most handoffs are unnecessary)
    For VERIFIER turns (turn > 0):
      FIRST: Is the previous tentative_answer correct (matches gold)?

      If previous tentative was CORRECT:
        +0.8 = agent confirms it with independent evidence (good)
         0.0 = agent confirms it without checking evidence (lazy but ok)
        -0.8 = agent unnecessarily challenges/changes a correct answer (harmful)
        -1.0 = agent changes correct answer to wrong answer (very harmful)

      If previous tentative was WRONG:
        +1.0 = agent identifies the error AND provides the correct answer with evidence
        +0.5 = agent identifies the error but answer still imprecise
         0.0 = agent passes through without catching the error
        -0.8 = agent confirms a wrong answer (blind confirmation)
        -1.0 = agent confirms a wrong answer AND makes reasoning worse

      If this is a REDUNDANT handoff (correct answer already established, no new info added):
        -0.5 = penalize -- the team should have stopped earlier

    For PROPOSER turns (turn = 0):
      +1.0 = gave a non-empty, well-reasoned tentative_answer AND appropriate handoff
      +0.5 = gave tentative_answer but reasoning weak
       0.0 = gave tentative_answer but it is vague/empty-ish
      -0.5 = gave no tentative_answer (left it blank)
      -1.0 = completely wrong action or harmful reasoning

  IMPORTANT: Do NOT give action_score > 0 just because reasoning looks good.
  Action quality is judged by OUTCOME CONTRIBUTION, not by effort.

Output a JSON array, one object per turn, in turn order:
[
  {
    "turn": 0,
    "agent_id": "A1",
    "reasoning_score": 0.80,
    "action_score": 0.30,
    "comment": "one sentence: what the agent did and why the action score"
  },
  ...
]
Output ONLY the JSON array. No prose before or after."""


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

    try:
        resp = call_llm_stream(
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            tools=None,
            model=model,
        )
        content = (resp.get("content") or "").strip()

        start = content.find("[")
        end   = content.rfind("]") + 1
        if start < 0 or end <= start:
            return None

        raw_scores = json.loads(content[start:end])
        if not isinstance(raw_scores, list) or not raw_scores:
            return None

    except Exception:
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
