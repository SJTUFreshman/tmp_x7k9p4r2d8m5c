"""Multi-Agent Debate (MAD) baseline for MuSiQue.

Implements the standard Du et al. 2023 MAD protocol on 3 heterogeneous
Qwen3 base models:

    A1 = Qwen3-1.7B   (no LoRA, pure base)
    A2 = Qwen3-4B     (no LoRA, pure base)
    A3 = Qwen3-8B     (no LoRA, pure base)

Protocol:
    Round 0: each agent answers independently (temperature high, e.g. 0.9)
    Round r >= 1: each agent sees the OTHER agents' round-(r-1) reasoning
                  and answers (anonymously, in random order as "Peer A" /
                  "Peer B"), and produces an updated answer (temperature
                  lower, e.g. 0.3).
    Aggregation: majority vote over the final round's answers using
                 grader.normalize_answer for comparison; tie-break by
                 preferring A3's answer.

Public API:
    run_mad_for_problem(problem, callers, ...) -> MADRecord
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jca.src.agents import AGENT_IDS
from jca.src.inference import response_attempts
from jca.src.data import MuSiQueProblem, format_problem_as_prompt
from jca.src.grader import normalize_answer


PROMPT_ROUND0_PATH = Path(__file__).parent / "prompts" / "round0.md"
PROMPT_DEBATE_PATH = Path(__file__).parent / "prompts" / "debate.md"


LLMCaller = Callable[[List[Dict[str, str]]], str]


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class AgentTurn:
    """One agent's output at one round."""

    agent_id: str
    round_idx: int
    reasoning: str
    answer: str
    raw_output: str
    parse_ok: bool
    retried: bool = False
    raw_outputs: List[str] = field(default_factory=list)


@dataclass
class MADRecord:
    """Full record of one problem run under MAD."""

    problem_id: str
    n_rounds: int
    turns: List[AgentTurn] = field(default_factory=list)
    final_answer: Optional[str] = None
    voter_answers: Dict[str, str] = field(default_factory=dict)
    tie_break: bool = False
    error: Optional[str] = None

    def turn_at(self, round_idx: int, agent_id: str) -> Optional[AgentTurn]:
        for turn in self.turns:
            if turn.round_idx == round_idx and turn.agent_id == agent_id:
                return turn
        return None


# ============================================================================
# Prompt loading
# ============================================================================


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ============================================================================
# Message builders
# ============================================================================


def _build_round0_messages(problem: MuSiQueProblem) -> List[Dict[str, str]]:
    """System + user for the independent round-0 turn (same for all agents)."""
    return [
        {"role": "system", "content": _load_prompt(PROMPT_ROUND0_PATH)},
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]


def _format_peer_block(peer_label: str, turn: AgentTurn) -> str:
    return (
        f"## {peer_label}\n"
        f"Reasoning: {turn.reasoning}\n"
        f"Answer: {turn.answer}"
    )


def _build_debate_messages(
    problem: MuSiQueProblem,
    own_prev: AgentTurn,
    peer_turns: List[AgentTurn],
    rng: random.Random,
) -> List[Dict[str, str]]:
    """System + user for a debate-round turn.

    Peers are presented anonymously as 'Peer A' / 'Peer B' in random order
    so the model cannot infer which peer is which base model.
    """
    shuffled = list(peer_turns)
    rng.shuffle(shuffled)
    peer_labels = ["Peer A", "Peer B"]
    peer_blocks = [
        _format_peer_block(label, turn) for label, turn in zip(peer_labels, shuffled)
    ]

    user_content = (
        f"{format_problem_as_prompt(problem)}\n\n"
        f"# Your Previous Answer\n"
        f"Reasoning: {own_prev.reasoning}\n"
        f"Answer: {own_prev.answer}\n\n"
        f"# Other Agents' Answers\n"
        + "\n\n".join(peer_blocks)
        + "\n\nProduce your updated JSON answer for this round."
    )
    return [
        {"role": "system", "content": _load_prompt(PROMPT_DEBATE_PATH)},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# JSON extraction
# ============================================================================


def _parse_mad_json(raw_output: str) -> Optional[Tuple[str, str]]:
    """Return (reasoning, answer) if parseable, else None."""
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()

    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                payload, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]

    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str):
            answer_s = answer.strip()
            if answer_s:
                return reasoning.strip(), answer_s

    return _regex_fallback(text)


def _regex_fallback(text: str) -> Optional[Tuple[str, str]]:
    """Extract fields via regex when JSON is malformed."""
    reasoning_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    answer_m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if reasoning_m and answer_m:
        reasoning = reasoning_m.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
        answer = answer_m.group(1).encode("utf-8").decode("unicode_escape", errors="replace").strip()
        if answer:
            return reasoning.strip(), answer
    return None


def _retry_instruction() -> str:
    return (
        "Your previous output could not be parsed. Return ONLY a valid JSON "
        'object of the form {"reasoning": "...", "answer": "..."} '
        "with a non-empty answer field. No prose, no code fences."
    )


# ============================================================================
# Single-agent single-round execution
# ============================================================================


def _run_one_turn(
    caller: LLMCaller,
    agent_id: str,
    round_idx: int,
    base_messages: List[Dict[str, str]],
) -> AgentTurn:
    raw_output = caller([dict(m) for m in base_messages])
    raw_outputs = response_attempts(raw_output)
    parsed = _parse_mad_json(raw_output)
    retried = False

    if parsed is None:
        retry_messages = [dict(m) for m in base_messages]
        retry_messages.append({"role": "user", "content": _retry_instruction()})
        raw_output_retry = caller(retry_messages)
        raw_outputs.extend(response_attempts(raw_output_retry))
        parsed_retry = _parse_mad_json(raw_output_retry)
        retried = True
        if parsed_retry is not None:
            reasoning, answer = parsed_retry
            return AgentTurn(
                agent_id=agent_id,
                round_idx=round_idx,
                reasoning=reasoning,
                answer=answer,
                raw_output=raw_output_retry,
                parse_ok=True,
                retried=True,
                raw_outputs=raw_outputs,
            )
        return AgentTurn(
            agent_id=agent_id,
            round_idx=round_idx,
            reasoning="",
            answer="",
            raw_output=raw_output_retry,
            parse_ok=False,
            retried=True,
            raw_outputs=raw_outputs,
        )

    reasoning, answer = parsed
    return AgentTurn(
        agent_id=agent_id,
        round_idx=round_idx,
        reasoning=reasoning,
        answer=answer,
        raw_output=raw_output,
        parse_ok=True,
        retried=retried,
        raw_outputs=raw_outputs,
    )


# ============================================================================
# Aggregation
# ============================================================================


def _majority_vote(
    answers_by_agent: Dict[str, str],
    tie_break_priority: List[str],
) -> Tuple[Optional[str], bool]:
    """Return (final_answer, tie_break_used).

    - Non-empty answers only.
    - Compare using normalize_answer, keep the raw string with the highest
      count. Ties broken by tie_break_priority order (first match wins).
    - final_answer is the raw (un-normalized) string from the highest-priority
      agent whose normalized answer matches the winning bucket.
    """
    non_empty = [
        (agent_id, ans)
        for agent_id, ans in answers_by_agent.items()
        if ans and normalize_answer(ans)
    ]
    if not non_empty:
        return None, False

    buckets: Dict[str, List[str]] = {}
    for agent_id, ans in non_empty:
        key = normalize_answer(ans)
        buckets.setdefault(key, []).append(agent_id)

    max_count = max(len(v) for v in buckets.values())
    winners = [key for key, v in buckets.items() if len(v) == max_count]
    tie_break_used = len(winners) > 1

    if tie_break_used:
        for preferred in tie_break_priority:
            for key in winners:
                if preferred in buckets[key]:
                    for agent_id, ans in non_empty:
                        if agent_id == preferred and normalize_answer(ans) == key:
                            return ans, True
        winning_key = winners[0]
    else:
        winning_key = winners[0]

    for preferred in tie_break_priority:
        for agent_id, ans in non_empty:
            if agent_id == preferred and normalize_answer(ans) == winning_key:
                return ans, tie_break_used

    for agent_id, ans in non_empty:
        if normalize_answer(ans) == winning_key:
            return ans, tie_break_used

    return None, False


# ============================================================================
# Full problem run
# ============================================================================


def run_mad_for_problem(
    problem: MuSiQueProblem,
    callers_round0: Dict[str, LLMCaller],
    callers_debate: Dict[str, LLMCaller],
    *,
    n_rounds: int = 3,
    tie_break_priority: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> MADRecord:
    """Run n_rounds of MAD debate for one MuSiQue problem.

    Two caller dicts are required because round 0 typically uses a higher
    temperature (for independent diversity) than debate rounds (for
    convergence). Each dict maps agent_id -> LLMCaller.

    Rounds are executed sequentially, but the 3 agents within a round
    run concurrently via a small ThreadPoolExecutor.
    """
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    for agent_id in AGENT_IDS:
        if agent_id not in callers_round0:
            raise ValueError(f"Missing round-0 caller for agent {agent_id}")
        if agent_id not in callers_debate:
            raise ValueError(f"Missing debate caller for agent {agent_id}")

    tie_break_priority = tie_break_priority or ["A3", "A2", "A1"]
    rng = random.Random(seed if seed is not None else hash(problem.id) & 0xFFFFFFFF)

    record = MADRecord(problem_id=problem.id, n_rounds=n_rounds)

    try:
        prev_turns: Dict[str, AgentTurn] = {}
        for round_idx in range(n_rounds):
            if round_idx == 0:
                base_messages_by_agent = {
                    agent_id: _build_round0_messages(problem) for agent_id in AGENT_IDS
                }
                round_callers = callers_round0
            else:
                base_messages_by_agent = {}
                for agent_id in AGENT_IDS:
                    own_prev = prev_turns[agent_id]
                    peer_turns = [prev_turns[a] for a in AGENT_IDS if a != agent_id]
                    base_messages_by_agent[agent_id] = _build_debate_messages(
                        problem, own_prev, peer_turns, rng
                    )
                round_callers = callers_debate

            with ThreadPoolExecutor(max_workers=len(AGENT_IDS)) as executor:
                futures = {
                    agent_id: executor.submit(
                        _run_one_turn,
                        round_callers[agent_id],
                        agent_id,
                        round_idx,
                        base_messages_by_agent[agent_id],
                    )
                    for agent_id in AGENT_IDS
                }
                new_turns: Dict[str, AgentTurn] = {
                    agent_id: future.result() for agent_id, future in futures.items()
                }

            for agent_id in AGENT_IDS:
                record.turns.append(new_turns[agent_id])
            prev_turns = new_turns
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    final_round = record.n_rounds - 1
    final_answers: Dict[str, str] = {}
    for agent_id in AGENT_IDS:
        turn = record.turn_at(final_round, agent_id)
        if turn is not None and turn.answer:
            final_answers[agent_id] = turn.answer
    record.voter_answers = final_answers

    voted, tie_break_used = _majority_vote(final_answers, tie_break_priority)
    record.tie_break = tie_break_used
    if voted is not None:
        record.final_answer = f"\\boxed{{{voted}}}"
    return record


# ============================================================================
# Serialization
# ============================================================================


def mad_record_to_dict(record: MADRecord) -> Dict[str, Any]:
    return {
        "problem_id": record.problem_id,
        "n_rounds": record.n_rounds,
        "turns": [asdict(turn) for turn in record.turns],
        "final_answer": record.final_answer,
        "voter_answers": record.voter_answers,
        "tie_break": record.tie_break,
        "error": record.error,
    }
