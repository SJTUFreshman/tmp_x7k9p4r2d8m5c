"""The debate protocol, ported verbatim from the official implementation.

Provenance: /tmp/maporl_ref/trl/trl/trainer/utils_multi_unified_chat.py
  construct_message_multi_agent :200-284
  combine_agent_contexts        :141-180
  formatting_question           :183-196

The prompt strings below are reproduced **character for character**, including
the upstream typos ("internaly", "justifing") and the missing space before
"Ensure". They are pinned by golden-string tests: if a future edit normalises
them, the test fails and the deviation becomes a deliberate choice rather than
an accident.

Two structural properties worth knowing:

* ``idx = 2*turn - 1`` (no verifier) means an agent at turn ``t`` sees only the
  **immediately preceding** turn's responses from its peers -- not the whole
  debate. Its own history is complete, the peers' is not.
* Per agent per turn the context grows by exactly 2 messages without a verifier
  (assistant response, then the next debate prompt) or 3 with one. We run
  ``reward_feedback=False``, so stride 2, and ``combine_agent_contexts`` does no
  merging.
"""
from __future__ import annotations

from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Verbatim strings. Do not "fix" the typos -- tests pin them.
# ---------------------------------------------------------------------------
PREFIX = "These are the solutions to the problem from other agents: "

SUFFIX_NO_VERIFIER = (
    "Focus on providing a well-reasoned response that not only considers your own "
    "previous solution but also takes into account answers from other agents. "
    "If you believe your previous answer was incorrect, feel free to revise it. "
    "However, avoid repeating the same answer you or other agents have already "
    "provided. Also, internaly think about the reward of your and other agents' "
    "answer."
    "Ensure that your explanation is well justifing your final answer. Please "
    "maintain your answer with very simple reasoning.\n\n"
    "Once again, the question is: {question}"
)

SUFFIX_WITH_VERIFIER = (
    "Here, each reward represents the probability that a suggested answer is "
    "correct, as evaluated by a verifier. "
    "The reward value is between 0 and 1, with values closer to 1 indicating a "
    "higher likelihood of correctness. "
    "While these rewards offer useful context, they are not always perfect, "
    "though generally quite reliable.\n\n"
) + SUFFIX_NO_VERIFIER


def agent_solution_block(agent_number: int, response: str) -> str:
    """One peer's contribution. ``agent_number`` is 1-based (:259)."""
    return f"\n\n Agent {agent_number} solution: ```{response}```"


def peer_response_index(turn: int, *, reward_feedback: bool = False) -> int:
    """Which message of a peer's history to quote. :228-232.

    turn 1 -> 1, turn 2 -> 3, ... with stride 2 (no verifier). Always the peer's
    response from turn ``turn - 1``.
    """
    if turn < 1:
        raise ValueError(f"debate messages only exist for turn >= 1 (got {turn})")
    return 3 * turn - 2 if reward_feedback else 2 * turn - 1


def construct_message_multi_agent(
    peer_contexts: Sequence[Sequence[dict[str, str]]],
    question: str,
    turn: int,
    *,
    reward_feedback: bool = False,
) -> dict[str, str]:
    """Build the turn>=1 user message. Port of :200-284.

    ``peer_contexts`` is the other agents' message histories, self excluded.
    """
    if not peer_contexts:
        # The official len(agents)==0 branch (:214-224) is unreachable: the
        # function is only ever called with turn != 0 and agent_num >= 2. We
        # reject rather than reproduce dead code.
        raise ValueError("construct_message_multi_agent requires at least one peer")

    idx = peer_response_index(turn, reward_feedback=reward_feedback)
    body = PREFIX
    for number, context in enumerate(peer_contexts, start=1):
        if idx >= len(context):
            raise IndexError(
                f"peer history too short for turn {turn}: need index {idx}, "
                f"have {len(context)} messages"
            )
        body += agent_solution_block(number, context[idx]["content"])

    suffix = SUFFIX_WITH_VERIFIER if reward_feedback else SUFFIX_NO_VERIFIER
    return {"role": "user", "content": body + suffix.format(question=question)}


def combine_agent_context(
    context: Sequence[dict[str, str]],
    turn: int,
    *,
    reward_feedback: bool = False,
) -> list[dict[str, str]]:
    """Flatten one agent's history into strict user/assistant alternation.

    Port of :141-180 for a single agent. Without a verifier: keep message 0,
    then for each completed turn keep ``2i+1`` (assistant) and ``2i+2`` (user).
    With one, messages ``3i+2`` and ``3i+3`` are merged with "\\n\\n".
    """
    out = [dict(context[0])]
    for i in range(turn):
        if reward_feedback:
            out.append(dict(context[3 * i + 1]))
            out.append(
                {
                    "role": "user",
                    "content": context[3 * i + 2]["content"]
                    + "\n\n"
                    + context[3 * i + 3]["content"],
                }
            )
        else:
            out.append(dict(context[2 * i + 1]))
            out.append(dict(context[2 * i + 2]))
    return out


def assert_alternating(messages: Sequence[dict[str, str]]) -> None:
    """A rendered context must alternate user/assistant after the first message."""
    if not messages:
        raise ValueError("empty context")
    if messages[0]["role"] != "user":
        raise ValueError(f"context must open with a user turn, got {messages[0]['role']}")
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == current["role"]:
            raise ValueError(
                f"context is not alternating: {previous['role']} followed by "
                f"{current['role']}"
            )


class DebateContext:
    """Per-agent message history across a debate, with the official stride.

    Turn 0 seeds one user message; each turn appends the assistant response and
    then the next turn's debate prompt, so ``len == 2*turn + 1`` before the call
    at ``turn``.
    """

    def __init__(self, seed_prompt: str, *, reward_feedback: bool = False) -> None:
        self.reward_feedback = reward_feedback
        self.messages: list[dict[str, str]] = [{"role": "user", "content": seed_prompt}]

    def prompt_for(self, turn: int) -> list[dict[str, str]]:
        expected = (3 if self.reward_feedback else 2) * turn + 1
        if len(self.messages) != expected:
            raise RuntimeError(
                f"context has {len(self.messages)} messages, expected {expected} "
                f"before turn {turn}"
            )
        rendered = combine_agent_context(
            self.messages, turn, reward_feedback=self.reward_feedback
        )
        assert_alternating(rendered)
        return rendered

    def add_response(self, response: str) -> None:
        self.messages.append({"role": "assistant", "content": response})

    def add_reward_message(self, content: str) -> None:
        if not self.reward_feedback:
            raise RuntimeError("reward messages require reward_feedback=True")
        self.messages.append({"role": "user", "content": content})

    def add_debate_message(self, message: dict[str, str]) -> None:
        self.messages.append(dict(message))


def reward_feedback_message(score: float) -> str:
    """The agent's own reward message. Port of :1420-1432.

    Unused while ``reward_feedback=False``; kept so the verifier extension does
    not have to re-derive the bucket boundaries.
    """
    if score < 0.3:
        feedback = "Your answer is highly likely wrong."
    elif score < 0.6:
        feedback = (
            "Your answer might be wrong, or your reasoning needs to have a "
            "stronger argument."
        )
    elif score < 0.8:
        feedback = (
            "Your answer seems right, but check your reasoning again, there "
            "might be some room for improvement."
        )
    else:
        feedback = "Your answer is likely right with high probability."
    return f"Reward from a verifier of your answer: {score:.3f} out of 1.0, which means {feedback}"
