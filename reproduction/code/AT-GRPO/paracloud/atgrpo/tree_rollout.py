"""Tree-structured sampling -- the mechanism that makes AT-GRPO work.

Reference: "Stronger Together: On-Policy Reinforcement Learning for
Collaborative LLMs" (Zhao et al., arXiv:2510.11062, ICLR 2026), whose algorithm
is AT-GRPO. Code: github.com/pettingllms-ai/PettingLLMs.

THE PROBLEM IT SOLVES. GRPO's advantage needs several outputs sampled from the
*identical* prompt. Sample K independent full trajectories instead and, at any
turn t > 0, no two samples share a prompt -- history has diverged -- so every
group has size 1, the group baseline is the sample itself, the advantage is
identically zero, and GRPO silently stops reducing variance.

THE FIX. Branch per turn rather than per episode: freeze the conversation state
at turn t, sample K candidate actions for the agent that acts at t, and score
each by continuing the episode to the end. All K candidates then share a
byte-identical prompt, so the (agent, turn) group is a valid GRPO group.

Two branching modes:

* ``spine`` (default) -- roll one trajectory; at each turn branch K ways from
  the spine's state, continue each to the end, then advance the spine along one
  sampled child. Cost is ``T * K`` continuations per prompt, and every turn gets
  exactly one group of size K. This is the shape ``mas_grpo_probe/08`` already
  implements for single-agent frozen-context resampling.
* ``full_tree`` -- branch every child at every level: ``K^T`` leaves. Faithful to
  a literal reading of "branches at each turn" but explosive (K=4, T=4 -> 256
  episodes per prompt). Available, warned about, not the default.

The paper does not pin down which of the two it uses; ``branch_mode`` records
our choice rather than hiding it.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

from .protocol import AGENT_IDS, answer_of, format_joint_observation, repair_instruction
from .tasks.base import Problem, TaskAdapter
from .transport import ConcurrencyLimitedCaller


def _derive_seed(base: int, *parts: int) -> int:
    value = base
    for part in parts:
        value = (value * 1_000_003 + part) % (2**31 - 1)
    return value


def prompt_fingerprint(messages: Sequence[dict[str, str]]) -> str:
    """Stable hash of a rendered prompt.

    Used to *assert* the defining invariant: every candidate in an (agent, turn)
    group must have been generated from the same prompt. MAGRPO had to drop this
    check because joint rollouts diverge; AT-GRPO requires it.
    """
    blob = json.dumps(list(messages), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Candidate:
    """One sampled action for one agent at one turn, plus its episode outcome."""

    agent: str
    turn: int
    candidate_idx: int
    group_id: str
    prompt_messages: list[dict[str, str]]
    prompt_fingerprint: str
    response: str
    parsed: dict[str, Any] | None = None
    protocol_error: str | None = None
    answer: str | None = None
    # Outcome of continuing the episode after this candidate.
    outcome_answer: str | None = None
    reward: float | None = None
    advantage: float | None = None
    reward_detail: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TreeRollout:
    """Every candidate produced for one prompt."""

    problem_id: str
    branch_mode: str
    round_num: int
    agents: tuple[str, ...]
    candidates: list[Candidate] = field(default_factory=list)
    spine_answer: str | None = None
    error: str | None = None

    def groups(self) -> dict[str, list[Candidate]]:
        """Group by ``group_id`` -- one group per (agent, turn, branch point)."""
        out: dict[str, list[Candidate]] = {}
        for candidate in self.candidates:
            out.setdefault(candidate.group_id, []).append(candidate)
        for group in out.values():
            group.sort(key=lambda c: c.candidate_idx)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "branch_mode": self.branch_mode,
            "round_num": self.round_num,
            "agents": list(self.agents),
            "candidates": [c.to_dict() for c in self.candidates],
            "spine_answer": self.spine_answer,
            "error": self.error,
        }


def acting_agent(turn: int, agents: Sequence[str]) -> str:
    """Round-robin schedule: agent ``turn % n`` acts at turn ``turn``.

    AT-GRPO groups by (agent, turn), which presupposes a known actor per turn --
    a role-based workflow, as in the paper's Coder/Tester setup. Round-robin over
    our heterogeneous trio is the minimal such schedule.
    """
    return agents[turn % len(agents)]


def _call(
    caller: Any,
    task: TaskAdapter,
    messages: list[dict[str, str]],
    *,
    agent: str,
    turn: int,
    seed: int,
    step_retries: int,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    schema = task.response_schema(agent)
    attempt_messages = list(messages)
    raw, parsed, error, finish = "", None, None, None
    for attempt in range(step_retries + 1):
        completion = caller.generate(
            attempt_messages,
            seed=_derive_seed(seed, turn, attempt),
            response_format=schema,
        )
        raw, finish = completion.text, completion.finish_reason
        parsed, error = task.parse(raw, agent=agent, turn=turn)
        if parsed is not None:
            return raw, parsed, None, finish
        if attempt < step_retries:
            attempt_messages = list(messages) + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_instruction(error or "invalid")},
            ]
    return raw, None, error, finish


def _build_prompt(
    task: TaskAdapter,
    problem: Problem,
    agent: str,
    history: Sequence[str],
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": task.system_prompt(agent, joint_mode="sequential")},
        {"role": "user", "content": task.user_prompt(problem)},
    ]
    for block in history:
        messages.append({"role": "user", "content": block})
    return messages


def _continue_episode(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    agents: Sequence[str],
    history: list[str],
    start_turn: int,
    round_num: int,
    seed: int,
    step_retries: int,
) -> tuple[str | None, list[str]]:
    """Run turns ``start_turn..round_num-1`` and return the final answer.

    This is the "score the candidate by its consequences" half of tree sampling:
    a candidate's reward is the outcome of the episode it leads to, not of the
    candidate in isolation.
    """
    local_history = list(history)
    last_answer: str | None = None
    for turn in range(start_turn, round_num):
        agent = acting_agent(turn, agents)
        prompt = _build_prompt(task, problem, agent, local_history)
        raw, parsed, _, _ = _call(
            callers[agent], task, prompt,
            agent=agent, turn=turn,
            seed=_derive_seed(seed, turn), step_retries=step_retries,
        )
        answer = answer_of(parsed) if parsed else None
        if answer:
            last_answer = answer
        local_history.append(
            format_joint_observation({agent: parsed}, turn=turn)
        )
    return last_answer, local_history


def rollout_spine(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    agents: Sequence[str],
    round_num: int,
    group_size: int,
    seed: int,
    iteration: int = 0,
    step_retries: int = 1,
    max_workers: int = 8,
) -> TreeRollout:
    """Spine branching: one group of ``group_size`` candidates per turn.

    At each turn the acting agent's prompt is frozen, K candidates are sampled
    from it, each is continued to the end for its reward, and the spine advances
    along candidate 0 so the next turn branches from a real state.
    """
    rollout = TreeRollout(
        problem_id=problem.problem_id,
        branch_mode="spine",
        round_num=round_num,
        agents=tuple(agents),
    )
    history: list[str] = []

    for turn in range(round_num):
        agent = acting_agent(turn, agents)
        prompt = _build_prompt(task, problem, agent, history)
        fingerprint = prompt_fingerprint(prompt)
        group_id = f"{problem.problem_id}#it{iteration}#t{turn}#{agent}"

        def sample(index: int) -> Candidate:
            raw, parsed, error, finish = _call(
                callers[agent], task, prompt,
                agent=agent, turn=turn,
                # The seed is the ONLY thing that differs across candidates --
                # the prompt is byte-identical by construction.
                seed=_derive_seed(seed, iteration, turn, index),
                step_retries=step_retries,
            )
            return Candidate(
                agent=agent, turn=turn, candidate_idx=index, group_id=group_id,
                prompt_messages=prompt, prompt_fingerprint=fingerprint,
                response=raw, parsed=parsed, protocol_error=error,
                answer=answer_of(parsed) if parsed else None,
                finish_reason=finish,
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, group_size)) as pool:
            group = list(pool.map(sample, range(group_size)))

        # Each candidate is scored by the episode it produces.
        def finish_one(candidate: Candidate) -> None:
            child_history = history + [
                format_joint_observation({agent: candidate.parsed}, turn=turn)
            ]
            outcome, _ = _continue_episode(
                task, problem, callers,
                agents=agents, history=child_history,
                start_turn=turn + 1, round_num=round_num,
                seed=_derive_seed(seed, iteration, turn, candidate.candidate_idx, 7),
                step_retries=step_retries,
            )
            # A candidate at the last turn IS the outcome.
            candidate.outcome_answer = outcome or candidate.answer

        with ThreadPoolExecutor(max_workers=min(max_workers, group_size)) as pool:
            list(pool.map(finish_one, group))

        rollout.candidates.extend(group)
        # Advance the spine along candidate 0.
        history.append(format_joint_observation({agent: group[0].parsed}, turn=turn))
        rollout.spine_answer = group[0].outcome_answer

    return rollout


def rollout_full_tree(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    agents: Sequence[str],
    round_num: int,
    group_size: int,
    seed: int,
    iteration: int = 0,
    step_retries: int = 1,
    max_workers: int = 8,
) -> TreeRollout:
    """Branch every node: ``group_size ** round_num`` leaves.

    Faithful to the literal reading, but the cost is exponential in the number
    of turns. Guard before calling.
    """
    rollout = TreeRollout(
        problem_id=problem.problem_id,
        branch_mode="full_tree",
        round_num=round_num,
        agents=tuple(agents),
    )
    # (history, path) frontier; path makes each branch point's group id unique.
    frontier: list[tuple[list[str], str]] = [([], "root")]

    for turn in range(round_num):
        agent = acting_agent(turn, agents)
        next_frontier: list[tuple[list[str], str]] = []
        for history, path in frontier:
            prompt = _build_prompt(task, problem, agent, history)
            fingerprint = prompt_fingerprint(prompt)
            group_id = f"{problem.problem_id}#it{iteration}#t{turn}#{agent}#{path}"

            def sample(index: int) -> Candidate:
                raw, parsed, error, finish = _call(
                    callers[agent], task, prompt,
                    agent=agent, turn=turn,
                    seed=_derive_seed(seed, iteration, turn, index, hash(path) % 9973),
                    step_retries=step_retries,
                )
                return Candidate(
                    agent=agent, turn=turn, candidate_idx=index, group_id=group_id,
                    prompt_messages=prompt, prompt_fingerprint=fingerprint,
                    response=raw, parsed=parsed, protocol_error=error,
                    answer=answer_of(parsed) if parsed else None,
                    finish_reason=finish,
                )

            with ThreadPoolExecutor(max_workers=min(max_workers, group_size)) as pool:
                group = list(pool.map(sample, range(group_size)))
            rollout.candidates.extend(group)
            for candidate in group:
                child = history + [
                    format_joint_observation({agent: candidate.parsed}, turn=turn)
                ]
                # At the leaf level the candidate's own answer is the outcome.
                if turn == round_num - 1:
                    candidate.outcome_answer = candidate.answer
                next_frontier.append((child, f"{path}.{candidate.candidate_idx}"))
        frontier = next_frontier

    # Non-leaf candidates inherit the outcome of their first descendant leaf.
    leaves = {c.group_id: c for c in rollout.candidates if c.turn == round_num - 1}
    for candidate in rollout.candidates:
        if candidate.outcome_answer is None:
            candidate.outcome_answer = candidate.answer
    return rollout


def run_tree(
    task: TaskAdapter,
    problems: Sequence[Problem],
    callers: dict[str, Any],
    *,
    agents: Sequence[str],
    round_num: int,
    group_size: int,
    seed: int,
    iteration: int = 0,
    branch_mode: str = "spine",
    step_retries: int = 1,
    max_workers: int = 8,
) -> list[TreeRollout]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    runner = rollout_spine if branch_mode == "spine" else rollout_full_tree
    semaphore = threading.BoundedSemaphore(max_workers)
    limited_callers = {
        agent: ConcurrencyLimitedCaller(caller, semaphore)
        for agent, caller in callers.items()
    }

    def one(indexed: tuple[int, Problem]) -> TreeRollout:
        index, problem = indexed
        try:
            return runner(
                task, problem, limited_callers,
                agents=agents, round_num=round_num, group_size=group_size,
                seed=_derive_seed(seed, index), iteration=iteration,
                step_retries=step_retries, max_workers=max_workers,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, batch survives
            return TreeRollout(
                problem_id=problem.problem_id, branch_mode=branch_mode,
                round_num=round_num, agents=tuple(agents),
                error=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(problems) or 1)) as pool:
        return list(pool.map(one, enumerate(problems)))


def score_candidates(
    task: TaskAdapter,
    items: Sequence[tuple[Problem, TreeRollout]],
) -> None:
    """Reward every candidate by its episode outcome, in one batched call."""
    flat: list[tuple[Problem, str | None]] = []
    refs: list[Candidate] = []
    for problem, rollout in items:
        for candidate in rollout.candidates:
            flat.append((problem, candidate.outcome_answer))
            refs.append(candidate)
    if not flat:
        return
    for candidate, (reward, detail) in zip(refs, task.score_answers(flat)):
        candidate.reward = float(reward)
        candidate.reward_detail = detail
