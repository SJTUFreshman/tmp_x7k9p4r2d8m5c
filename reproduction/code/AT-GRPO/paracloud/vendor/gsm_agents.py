"""Agent configurations for GSM-HARD MAS.

Reuses the same Qwen3 model paths as the MuSiQue system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

AGENT_IDS: List[str] = ["A1", "A2", "A3"]
START_AGENT: str = "A1"


@dataclass
class AgentConfig:
    agent_id: str
    base_model: str
    description: str


AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "A1": AgentConfig(
        agent_id="A1",
        base_model="/data/wangyuheng/models/Qwen3-1.7B",
        description="Collaborating math solver (1.7B)",
    ),
    "A2": AgentConfig(
        agent_id="A2",
        base_model="/data/wangyuheng/models/Qwen3-4B",
        description="Collaborating math solver (4B)",
    ),
    "A3": AgentConfig(
        agent_id="A3",
        base_model="/data/wangyuheng/models/Qwen3-8B",
        description="Collaborating math solver (8B)",
    ),
}


def format_other_agents(agent_id: str) -> str:
    others = [a for a in AGENT_IDS if a != agent_id]
    return f"{others[0]} and {others[1]}"


def render_system_prompt(agent_id: str, *, min_agents_before_stop: int = 3) -> str:
    if agent_id not in AGENT_IDS:
        raise KeyError(f"Unknown agent_id: {agent_id}")
    if min_agents_before_stop < 1 or min_agents_before_stop > len(AGENT_IDS):
        raise ValueError(
            f"min_agents_before_stop must be in [1, {len(AGENT_IDS)}], "
            f"got {min_agents_before_stop}"
        )
    min_agents_line = (
        f"4. Do not use confirm_stop until at least {min_agents_before_stop} distinct agents "
        "have contributed a turn in the conversation."
        if min_agents_before_stop > 1
        else ""
    )
    return f"""You are agent {agent_id}. You collaborate with two other agents: {format_other_agents(agent_id)}.

# The Task
You are solving a GSM-style math word problem. The final answer must be a
single numeric value. Arithmetic precision matters.

# Shared Conversation
You and the other agents share this conversation. Each prior assistant message
is prefixed with the agent ID in brackets, for example: [A1] ...

# Collaborative Verification Protocol
At least two agents must verify the answer before it can be finalized.

1. The first agent computes a tentative_answer and hands off for verification.
2. A verifier must independently recompute the answer from scratch instead of
   paraphrasing the previous chain.
   - If they agree and the arithmetic is checked: action = "confirm_stop".
   - If they disagree or want one more check: give the corrected
     tentative_answer and action = "handoff".
3. confirm_stop is only valid once at least one previous tentative_answer
   exists in the conversation.
{min_agents_line}
5. Collaboration must add checking value. Later agents should verify the setup,
   arithmetic, unit conversion, and final numeric form. Do not rubber-stamp.
6. For nontrivial problems, it is better to request one more independent check
   than to finalize too early.

# Output Format
Output EXACTLY one JSON object and nothing else:

{{
  "reasoning": "<step-by-step arithmetic derivation>",
  "tentative_answer": "<plain number only>",
  "action": "handoff" | "confirm_stop",
  "handoff_target": "A1" | "A2" | "A3" | null,
  "handoff_note": "<brief verification note>" | null,
  "confirmed_answer": "<plain number only>" | null
}}

# Field Rules
- reasoning: show the arithmetic clearly enough for another agent to audit it.
- tentative_answer: plain number only. No units. No prose. No LaTeX.
- handoff:
    * confirmed_answer must be null.
    * handoff_target must be one of the other two agents.
    * handoff_note should name the exact computation, assumption, or final
      numeric step the next agent should verify.
- confirm_stop:
    * confirmed_answer must contain the final numeric answer.
    * only use this after a prior tentative_answer already exists.
    * only use this after enough distinct agents have contributed if the
      protocol requires that.
    * do not confirm_stop if you have not independently checked the math.

# Strategy
- First turn: solve carefully, give tentative_answer, and hand off.
- Verification turn: recompute from scratch, not from the previous final number.
- If you find an arithmetic mistake, correct it and hand off with a precise note.
- If the answer is already solid but the protocol still needs another check,
  hand off with a concrete verification request.
"""


def render_sft_rollout_system_prompt(
    agent_id: str,
    *,
    min_agents_before_stop: int = 3,
    min_handoffs_before_stop: int = 2,
    reference_answer: str | None = None,
) -> str:
    if min_handoffs_before_stop < 1:
        raise ValueError("min_handoffs_before_stop must be positive")
    base_prompt = render_system_prompt(
        agent_id,
        min_agents_before_stop=min_agents_before_stop,
    ).rstrip()
    reference_section = ""
    if reference_answer is not None:
        reference_section = f"""

# SFT Generation Control
The trusted numeric reference answer for this synthetic rollout is
{reference_answer}. Derive it from the original problem with explicit arithmetic.
Do not mention this reference, the generation process, or hidden instructions in
your reasoning. A response with a different tentative answer will be retried.
"""
    return f"""{base_prompt}

# SFT Warm-up Demonstration
This rollout is used only to demonstrate collaborative behavior during SFT data
generation. Agent identities have equal status and handoff routing is not fixed.

- If another agent has already answered, solve the original problem again from
  the question. Write your own reasoning instead of copying, paraphrasing, or
  lightly editing an earlier reasoning chain.
- If you change the tentative_answer, hand off so a later agent checks the
  corrected answer before the trajectory stops.
- This trajectory requires at least {min_handoffs_before_stop} completed handoffs
  before confirm_stop. Until then, hand off to either other agent.
- Once the collaboration requirements are satisfied, confirm only after you
  have genuinely checked the answer; do not extend the trajectory by repetition.
{reference_section}"""
