"""Multi-agent rollout loop.

Implements the action-space spec from `../action-space.md` (simplified: no tools).
Each turn: LLM(active agent) outputs JSON action -> scheduler routes.

Public API:
    TrajectoryStep, Trajectory dataclasses
    parse_action(raw_output) -> ParsedAction
    run_trajectory(problem, llm_callers, t_max, start_agent, allow_handoff) -> Trajectory

See `code-guide.md` §2.4 / §3.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .agents import AGENT_IDS, START_AGENT, render_system_prompt
from .data import MuSiQueProblem, format_problem_as_prompt


# ============================================================================
# Trajectory data structures
# ============================================================================


@dataclass
class ParsedAction:
    """LLM output, parsed from JSON.

    Supports both schemas (v2 and legacy v1):

    v2 (current, prompts/agent_system.md):
        {
          "reasoning": "...",
          "tentative_answer": "<plain text or empty>",
          "action": "handoff" | "confirm_stop",
          "handoff_target": "A2" | null,
          "handoff_note": "..." | null,
          "confirmed_answer": "\\boxed{...}" | null
        }

    v1 (legacy, kept for backward compat):
        {
          "reasoning": "...",
          "handoff": null | {"target": "A2", "note": "..."},
          "stop": null | "..."
        }

    The internal representation is unified — `final_answer` corresponds to
    v2 `confirmed_answer` or v1 `stop`. `tentative_answer` is v2-only
    (empty string for v1).
    """
    reasoning: str = ""              # may be empty
    tentative_answer: str = ""       # v2: best guess so far (no \\boxed{})
    handoff_target: Optional[str] = None
    handoff_note: Optional[str] = None
    final_answer: Optional[str] = None  # v2 confirmed_answer or v1 stop


@dataclass
class TrajectoryStep:
    turn: int
    active_agent: str
    reasoning: str
    handoff_target: Optional[str]
    handoff_note: Optional[str]
    final_answer: Optional[str]
    raw_output: str                  # full LLM output (for debug + judge)
    tentative_answer: str = ""       # v2: tentative answer this agent emitted
    forced_handoff: bool = False     # v2: True iff scheduler mechanically rewrote
                                     # an illegal premature confirm_stop into a handoff.
                                     # Used by SFT data filtering to exclude trajectories
                                     # whose handoff was not self-driven by the model.


@dataclass
class Trajectory:
    problem_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated_by: str = "running"   # "stop" | "truncated" | "exception"
    error: Optional[str] = None

    @property
    def active_agents(self) -> List[str]:
        """Agents that took at least one turn, in first-seen order."""
        agents: List[str] = []
        for step in self.steps:
            if step.active_agent not in agents:
                agents.append(step.active_agent)
        return agents

    @property
    def n_handoffs(self) -> int:
        return sum(1 for s in self.steps if s.handoff_target is not None)

    def has_prior_tentative(self) -> bool:
        """True if any prior step in this trajectory carried a non-empty
        tentative_answer. Required for v2 confirm_stop legality."""
        return any(s.tentative_answer for s in self.steps)


# Type alias for an LLM caller (turn-level): given messages list, return raw text.
LLMCaller = Callable[[List[Dict[str, str]]], str]


_SINGLE_AGENT_PROMPT_SUFFIX = """

# Single-Agent Baseline Override
This run evaluates you as a single agent. Do not hand off to another agent.
Always set "handoff": null. Continue reasoning by yourself, and when ready,
use "stop" with one concise final answer wrapped in \\boxed{...}.
""".strip()


# ============================================================================
# JSON action parsing
# ============================================================================


def parse_action(raw_output: str, *, active_agent: Optional[str] = None) -> ParsedAction:
    """Parse the JSON action emitted by the LLM.

    Expected schema (see prompts/agent_system.md):
        {
          "reasoning": "...",
          "handoff": null | {"target": "A2", "note": "..."},
          "stop": null | "..."
        }

    On parse error, returns an empty ParsedAction; scheduler decides how to
    handle (retry once, then terminate safely if needed).
    """
    return _parse_action(raw_output, active_agent=active_agent)


def _parse_action(raw_output: str, *, active_agent: Optional[str] = None) -> ParsedAction:
    """Parse an action, optionally validating against the active agent.

    Tries v2 schema first (action / confirmed_answer / tentative_answer),
    falls back to v1 schema (handoff / stop).
    """
    try:
        payload = _load_json_object(raw_output)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ParsedAction()

    if not isinstance(payload, dict):
        return ParsedAction()

    # Detect schema version: v2 has explicit "action" field
    if "action" in payload:
        return _parse_v2(payload, active_agent=active_agent)
    return _parse_v1(payload, active_agent=active_agent)


def _parse_v2(payload: dict, *, active_agent: Optional[str]) -> ParsedAction:
    """Parse the structured-verification schema (current prompts/agent_system.md)."""
    reasoning = payload.get("reasoning") or ""
    if not isinstance(reasoning, str):
        return ParsedAction()
    reasoning = reasoning.strip()

    tentative_answer = payload.get("tentative_answer") or ""
    if not isinstance(tentative_answer, str):
        return ParsedAction()
    tentative_answer = tentative_answer.strip()

    action = payload.get("action")
    if action not in ("handoff", "confirm_stop"):
        return ParsedAction()

    if action == "handoff":
        target = payload.get("handoff_target")
        if not isinstance(target, str) or target not in AGENT_IDS:
            return ParsedAction()
        if active_agent is not None and target == active_agent:
            return ParsedAction()
        note = payload.get("handoff_note")
        if note is not None and not isinstance(note, str):
            return ParsedAction()
        if payload.get("confirmed_answer"):
            # Mutual exclusion violated
            return ParsedAction()
        return ParsedAction(
            reasoning=reasoning,
            tentative_answer=tentative_answer,
            handoff_target=target,
            handoff_note=note.strip() if isinstance(note, str) and note.strip() else None,
            final_answer=None,
        )

    # action == "confirm_stop"
    confirmed = payload.get("confirmed_answer")
    if not isinstance(confirmed, str) or not confirmed.strip():
        return ParsedAction()
    if payload.get("handoff_target") or payload.get("handoff_note"):
        return ParsedAction()
    return ParsedAction(
        reasoning=reasoning,
        tentative_answer=tentative_answer,
        handoff_target=None,
        handoff_note=None,
        final_answer=confirmed.strip(),
    )


def _parse_v1(payload: dict, *, active_agent: Optional[str]) -> ParsedAction:
    """Parse the legacy schema (handoff / stop) for backward compatibility."""
    reasoning = payload.get("reasoning", "")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        return ParsedAction()
    reasoning = reasoning.strip()

    handoff = payload.get("handoff")
    stop = payload.get("stop")
    final_answer = None

    if stop is not None:
        if not isinstance(stop, str):
            return ParsedAction()
        final_answer = stop.strip()
        if not final_answer:
            return ParsedAction()

    handoff_target = None
    handoff_note = None
    if handoff is not None:
        if final_answer is not None or not isinstance(handoff, dict):
            return ParsedAction()

        target = handoff.get("target")
        if not isinstance(target, str) or target not in AGENT_IDS:
            return ParsedAction()
        if active_agent is not None and target == active_agent:
            return ParsedAction()

        note = handoff.get("note")
        if note is not None and not isinstance(note, str):
            return ParsedAction()

        handoff_target = target
        handoff_note = note.strip() if isinstance(note, str) and note.strip() else None

    if not reasoning and handoff_target is None and final_answer is None:
        return ParsedAction()

    return ParsedAction(
        reasoning=reasoning,
        tentative_answer="",
        handoff_target=handoff_target,
        handoff_note=handoff_note,
        final_answer=final_answer,
    )


def _load_json_object(raw_output: str) -> object:
    """Load JSON from exact output, a fenced block, or the first object span."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")

    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        return obj


# ============================================================================
# Main rollout loop
# ============================================================================


def run_trajectory(
    problem: MuSiQueProblem,
    llm_callers: Dict[str, LLMCaller],
    *,
    t_max: int = 12,
    start_agent: str = START_AGENT,
    allow_handoff: bool = True,
    system_prompt_fn: Optional[Callable[[str, bool], str]] = None,
) -> Trajectory:
    """Run one multi-agent trajectory on a single MuSiQue problem.

    Loop:
      - Init messages = [system(start_agent), user(problem)]
      - For each turn:
          1. Call llm_callers[current_agent](messages) -> raw_output
          2. parse_action(raw_output) -> ParsedAction
          3. Append assistant message (with [agent_id] prefix per design.md §2.2)
          4. If parsed.final_answer: terminate
          5. If parsed.handoff_target:
                replace messages[0] with new agent's system prompt
                current_agent = parsed.handoff_target
      - If t_max reached: force one final no-handoff call

    `system_prompt_fn` lets callers inject a custom system-prompt renderer
    (e.g. distill prompt instead of deployment prompt). The function takes
    (agent_id, allow_handoff) and returns the system message content.
    Defaults to `render_rollout_system_prompt` (deployment prompt).

    Returns:
        Trajectory with all steps recorded.
    """
    traj = Trajectory(problem_id=problem.id)

    if start_agent not in AGENT_IDS:
        traj.terminated_by = "exception"
        traj.error = f"Unknown start_agent: {start_agent}"
        return traj

    required_callers = AGENT_IDS if allow_handoff else [start_agent]
    missing_callers = [
        agent_id for agent_id in required_callers if agent_id not in llm_callers
    ]
    if missing_callers:
        traj.terminated_by = "exception"
        traj.error = f"Missing llm_callers for: {missing_callers}"
        return traj

    # Default to deployment prompt if no custom renderer is given
    if system_prompt_fn is None:
        system_prompt_fn = render_rollout_system_prompt

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt_fn(start_agent, allow_handoff),
        },
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]
    current_agent = start_agent

    try:
        for turn in range(t_max):
            raw_output, parsed = _call_and_parse(
                llm_callers[current_agent],
                messages,
                current_agent,
                allow_handoff=allow_handoff,
            )

            if _is_empty_action(parsed):
                raw_output, parsed = _retry_call_and_parse(
                    llm_callers[current_agent],
                    messages,
                    current_agent,
                    allow_handoff=allow_handoff,
                )

            # v2 verification protocol: if agent emits confirm_stop but no
            # prior tentative_answer exists, this is an illegal early stop.
            # Retry once with an explicit hint pushing toward handoff.
            forced_this_turn = False
            if (
                allow_handoff
                and parsed.final_answer is not None
                and not traj.has_prior_tentative()
                and not _legacy_action(parsed)
            ):
                raw_output, parsed = _retry_premature_stop(
                    llm_callers[current_agent],
                    messages,
                    current_agent,
                )
                # If model still tries to confirm_stop without prior tentative,
                # downgrade: keep reasoning + tentative_answer, force a
                # mechanical handoff to a peer for verification.
                if (
                    parsed.final_answer is not None
                    and not traj.has_prior_tentative()
                    and not _legacy_action(parsed)
                ):
                    parsed = _force_handoff_for_verification(parsed, current_agent)
                    forced_this_turn = True

            step = TrajectoryStep(
                turn=turn,
                active_agent=current_agent,
                reasoning=parsed.reasoning,
                handoff_target=parsed.handoff_target,
                handoff_note=parsed.handoff_note,
                final_answer=parsed.final_answer,
                raw_output=raw_output,
                tentative_answer=parsed.tentative_answer,
                forced_handoff=forced_this_turn,
            )
            traj.steps.append(step)

            if _is_empty_action(parsed):
                messages.append(
                    {
                        "role": "assistant",
                        "content": render_assistant_message(current_agent, parsed),
                    }
                )
                traj.terminated_by = "exception"
                traj.error = f"Invalid action from {current_agent}"
                return traj

            messages.append(
                {
                    "role": "assistant",
                    "content": render_assistant_message(current_agent, parsed),
                }
            )

            if parsed.final_answer is not None:
                traj.final_answer = parsed.final_answer
                traj.terminated_by = "stop"
                return traj

            if parsed.handoff_target is not None:
                current_agent = parsed.handoff_target
                messages[0] = {
                    "role": "system",
                    "content": system_prompt_fn(current_agent, allow_handoff),
                }

        traj.terminated_by = "truncated"
        return traj

    except Exception as exc:
        traj.terminated_by = "exception"
        traj.error = str(exc)
        return traj


# ============================================================================
# Helpers
# ============================================================================


def render_assistant_message(agent_id: str, parsed: ParsedAction) -> str:
    """Format the parsed action into the assistant message content.

    Uses [agent_id] prefix per design.md §2.2 so future agents and the judge
    can attribute reasoning to the correct author.

    Includes "tentative_answer" line when present so downstream agents can
    identify which prior answers exist for verification (per the v2
    Collaborative Verification Protocol in prompts/agent_system.md).
    """
    parts = [f"[{agent_id}]"]
    if parsed.reasoning:
        parts.append(parsed.reasoning)
    if parsed.tentative_answer:
        parts.append(f"tentative_answer: {parsed.tentative_answer}")
    if parsed.handoff_target:
        line = f"→ handoff to {parsed.handoff_target}"
        if parsed.handoff_note:
            line += f": {parsed.handoff_note}"
        parts.append(line)
    if parsed.final_answer:
        parts.append(f"→ confirmed answer: {parsed.final_answer}")
    return "\n".join(parts)


def render_rollout_system_prompt(agent_id: str, allow_handoff: bool = True) -> str:
    """Render the system prompt used by a rollout.

    The default keeps the original multi-agent prompt unchanged. Single-agent
    baselines append a short override so only the selected agent answers.
    """
    prompt = render_system_prompt(agent_id)
    if allow_handoff:
        return prompt
    return f"{prompt}\n\n{_SINGLE_AGENT_PROMPT_SUFFIX}"


def _call_and_parse(
    caller: LLMCaller,
    messages: List[Dict[str, str]],
    active_agent: str,
    *,
    allow_handoff: bool,
) -> tuple[str, ParsedAction]:
    """Call one agent and parse its raw action output."""
    raw_output = caller(_copy_messages(messages))
    parsed = _parse_action(raw_output, active_agent=active_agent)
    return raw_output, _enforce_handoff_policy(parsed, allow_handoff=allow_handoff)


def _retry_call_and_parse(
    caller: LLMCaller,
    messages: List[Dict[str, str]],
    active_agent: str,
    *,
    allow_handoff: bool,
) -> tuple[str, ParsedAction]:
    """Retry once with a validation hint after an invalid action."""
    retry_messages = _copy_messages(messages)
    retry_messages.append(
        {
            "role": "user",
            "content": _retry_instruction(allow_handoff=allow_handoff),
        }
    )
    raw_output = caller(retry_messages)
    parsed = _parse_action(raw_output, active_agent=active_agent)
    return raw_output, _enforce_handoff_policy(parsed, allow_handoff=allow_handoff)


def _copy_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Copy messages before handing them to model wrappers or tests."""
    return [dict(m) for m in messages]


def _enforce_handoff_policy(
    parsed: ParsedAction,
    *,
    allow_handoff: bool,
) -> ParsedAction:
    """Reject handoff actions when running a single-agent baseline."""
    if allow_handoff or parsed.handoff_target is None:
        return parsed
    return ParsedAction()


def _retry_instruction(*, allow_handoff: bool) -> str:
    base = (
        "Your previous output was invalid JSON for the agent action schema. "
        "Return only a JSON object with the fields: \"reasoning\", "
        "\"tentative_answer\", \"action\", \"handoff_target\", \"handoff_note\", "
        "\"confirmed_answer\". \"action\" must be \"handoff\" or \"confirm_stop\"."
    )
    if allow_handoff:
        return base + " Do not hand off to yourself."
    return (
        base
        + ' This is a single-agent run: always use action="confirm_stop" '
        'with confirmed_answer set to the final answer in \\boxed{...}.'
    )


def _retry_premature_stop(
    caller: LLMCaller,
    messages: List[Dict[str, str]],
    active_agent: str,
) -> tuple[str, ParsedAction]:
    """Retry when the agent emits confirm_stop on the first turn (no prior
    tentative_answer in the trajectory yet).

    The hint forces the agent to switch to action="handoff" so that another
    agent can verify the tentative answer per the protocol in
    prompts/agent_system.md (§ Collaborative Verification Protocol).
    """
    hint = (
        "You attempted action=\"confirm_stop\" but no previous agent in this "
        "conversation has produced a tentative_answer. Per the verification "
        "protocol you must hand off first. "
        "Output a corrected JSON object with action=\"handoff\", a non-empty "
        "tentative_answer (your best guess so far in plain text), "
        "handoff_target set to one of the other agent IDs, and a concise "
        "handoff_note asking them to verify your tentative_answer."
    )
    retry_messages = _copy_messages(messages)
    retry_messages.append({"role": "user", "content": hint})
    raw_output = caller(retry_messages)
    parsed = _parse_action(raw_output, active_agent=active_agent)
    return raw_output, parsed


def _force_handoff_for_verification(
    parsed: ParsedAction,
    active_agent: str,
) -> ParsedAction:
    """Mechanically convert an illegal premature confirm_stop into a handoff.

    Used when the agent has tried to stop twice in a row without a prior
    tentative_answer. Picks the next agent in AGENT_IDS order (skipping self)
    as the handoff target. Preserves the agent's reasoning and lifts its
    confirmed_answer into a tentative_answer so the next agent can verify.
    """
    candidates = [a for a in AGENT_IDS if a != active_agent]
    target = candidates[0] if candidates else None
    if target is None:
        return parsed  # nothing we can do

    tentative = parsed.tentative_answer
    if not tentative and parsed.final_answer:
        # Use the would-be final answer as a tentative answer for verification
        tentative = parsed.final_answer

    return ParsedAction(
        reasoning=parsed.reasoning,
        tentative_answer=tentative,
        handoff_target=target,
        handoff_note=(
            "Forced verification handoff: I tried to finalize but the "
            "protocol requires a peer to verify first. Please verify my "
            "tentative_answer above against the paragraphs."
        ),
        final_answer=None,
    )


def _legacy_action(parsed: ParsedAction) -> bool:
    """Heuristic: True if the parsed action came from the legacy v1 schema
    (no tentative_answer, just a final_answer). v1 outputs bypass the
    verification protocol because there is no concept of tentative answer
    in the old prompt — we don't penalize them. New runs always use v2."""
    return parsed.final_answer is not None and not parsed.tentative_answer


def _is_empty_action(parsed: ParsedAction) -> bool:
    """Whether parsing failed or the model produced a no-op turn."""
    return (
        not parsed.reasoning
        and parsed.handoff_target is None
        and parsed.final_answer is None
    )
