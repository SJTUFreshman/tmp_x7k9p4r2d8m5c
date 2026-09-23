"""The debate rollout engine.

All ``agent_num`` agents act at every turn. Turn 0 is independent drafting from
an identical prompt; from turn 1 each agent sees its peers' previous-turn
responses via :func:`maporl.debate.construct_message_multi_agent`.

One consequence of ``reward_feedback=False`` is worth stating, because it is
what makes the whole pipeline cheap: no reward text ever enters the context
(``ppov2_trainer_multi_different_model.py:1420-1438`` is inside an ``if``), and
early stopping is disabled. So correctness is never needed *during* a rollout --
every ``agent_num x round_num x B`` grading defers to one batched call
afterwards. That is what lets MultiPL-E run one docker invocation per iteration
instead of nine.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any, Sequence

from .debate import DebateContext, construct_message_multi_agent
from .protocol import answer_of, repair_instruction
from .tasks.base import Problem, TaskAdapter
from .trajectory import DebateTrajectory, TurnRecord
from .transport import Completion


@dataclass
class AttemptResult:
    completion: Completion
    prompt_messages: list[dict[str, str]]
    parsed: dict[str, Any] | None
    error: str | None
    attempts: int


def _derive_seed(base: int, *parts: int) -> int:
    value = base
    for part in parts:
        value = (value * 1_000_003 + part) % (2**31 - 1)
    return value


def _call_with_retry(
    caller: Any,
    task: TaskAdapter,
    messages: list[dict[str, str]],
    *,
    agent: str,
    turn: int,
    seed: int,
    step_retries: int,
) -> AttemptResult:
    """Generate, parse, retry once with a repair nudge on malformed output."""
    schema = task.response_schema(agent)
    if step_retries < 0:
        raise ValueError("step_retries must be nonnegative")
    attempt_messages = [dict(message) for message in messages]
    for attempt in range(step_retries + 1):
        completion = caller.generate(
            attempt_messages,
            seed=_derive_seed(seed, turn, attempt),
            response_format=schema,
        )
        parsed, error = task.parse(completion.text, agent=agent, turn=turn)
        if parsed is not None:
            return AttemptResult(completion, attempt_messages, parsed, None, attempt + 1)
        if attempt < step_retries:
            attempt_messages = [dict(message) for message in messages] + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": repair_instruction(error or "invalid")},
            ]
    return AttemptResult(completion, attempt_messages, None, error, step_retries + 1)


def run_debate_one(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str | tuple[int, str], Any],
    *,
    agents: Sequence[str],
    round_num: int,
    seed: int,
    step_retries: int = 1,
) -> DebateTrajectory:
    """One question's full debate. All agents act at every turn, in parallel."""
    traj = DebateTrajectory(
        problem_id=problem.problem_id, round_num=round_num, agents=tuple(agents)
    )
    user_prompt = task.user_prompt(problem)
    # Turn 0: every agent gets the SAME seed prompt. Diversity comes from
    # sampling (and, for us, from the three models being different sizes).
    contexts = {
        agent: DebateContext(
            f"{task.system_prompt(agent)}\n\n{user_prompt}", reward_feedback=False
        )
        for agent in agents
    }

    for turn in range(round_num):
        if turn > 0:
            for agent in agents:
                peers = [contexts[other].messages for other in agents if other != agent]
                contexts[agent].add_debate_message(
                    construct_message_multi_agent(
                        peers, user_prompt, turn, reward_feedback=False
                    )
                )

        prompts = {agent: contexts[agent].prompt_for(turn) for agent in agents}
        # Simultaneous action: no agent sees this turn's peers.
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {
                agent: pool.submit(
                    _call_with_retry,
                    callers[(turn, agent)] if (turn, agent) in callers else callers[agent],
                    task,
                    prompts[agent],
                    agent=agent,
                    turn=turn,
                    seed=_derive_seed(seed, agents.index(agent)),
                    step_retries=step_retries,
                )
                for agent in agents
            }
            outcomes = {agent: future.result() for agent, future in futures.items()}

        for agent in agents:
            outcome = outcomes[agent]
            completion = outcome.completion
            traj.put(
                TurnRecord(
                    turn=turn,
                    agent=agent,
                    prompt_messages=outcome.prompt_messages,
                    response=completion.text,
                    parsed=outcome.parsed,
                    protocol_error=outcome.error,
                    answer=answer_of(outcome.parsed) if outcome.parsed else None,
                    finish_reason=completion.finish_reason,
                    prompt_token_ids=completion.prompt_token_ids,
                    response_token_ids=completion.response_token_ids,
                    response_logprobs=completion.response_logprobs,
                    model_name=completion.model_name,
                    adapter_version=completion.adapter_version,
                    generation_config=completion.generation_config,
                    guided_decoding=completion.guided_decoding,
                    stop_reason=completion.stop_reason,
                    generation_attempts=outcome.attempts,
                )
            )
            # The peer view quotes the raw assistant text, as upstream does.
            contexts[agent].add_response(completion.text)

    traj.final_answer = task.aggregate_answers(
        {r.agent: r.answer for r in traj.turn_records(round_num - 1)}
    )
    return traj


def run_debate(
    task: TaskAdapter,
    problems: Sequence[Problem],
    callers: dict[str | tuple[int, str], Any],
    *,
    agents: Sequence[str],
    round_num: int,
    seed: int,
    iteration: int = 0,
    step_retries: int = 1,
    max_workers: int = 8,
    fail_fast: bool = False,
) -> list[DebateTrajectory]:
    """Roll out a batch, recording isolated errors unless fail_fast is enabled."""

    def one(index_problem: tuple[int, Problem]) -> DebateTrajectory:
        index, problem = index_problem
        try:
            return run_debate_one(
                task, problem, callers,
                agents=agents, round_num=round_num,
                seed=_derive_seed(seed, iteration, index),
                step_retries=step_retries,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            if fail_fast:
                raise
            return DebateTrajectory(
                problem_id=problem.problem_id,
                round_num=round_num,
                agents=tuple(agents),
                error=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(problems) or 1)) as pool:
        return list(pool.map(one, enumerate(problems)))


def score_trajectories(
    task: TaskAdapter,
    items: Sequence[tuple[Problem, DebateTrajectory]],
    *,
    binarize: str = "strict",
    threshold: float = 0.5,
) -> None:
    """Grade every ``(turn, agent)`` cell in one batched call.

    Writes ``correctness`` (ground truth, feeds ``bonus_rule``) and ``score``
    (feeds ``score_rule``). With ``reward_feedback=False`` the official code
    makes these the same tensor, so ``binarize="strict"`` -- the default -- keeps
    both binary even for Conifer, whose raw ``hard_score`` is continuous.

    ``binarize="split"`` keeps the continuous value in ``score`` while
    ``correctness`` stays binary. That carries more signal but decouples the two
    tables in a way ``reward_feedback=False`` never does upstream, so it is a
    documented variant rather than the default.
    """
    flat: list[tuple[Problem, str | None]] = []
    keys: list[tuple[int, tuple[int, str]]] = []
    for index, (problem, traj) in enumerate(items):
        for key, record in sorted(traj.records.items()):
            flat.append((problem, record.answer))
            keys.append((index, key))
    if not flat:
        return

    scored = task.score_answers(flat)
    if len(scored) != len(flat):
        raise RuntimeError(
            f"grader returned {len(scored)} scores for {len(flat)} answers"
        )
    if any(not math.isfinite(float(value)) for value, _ in scored):
        raise RuntimeError("grader returned a nonfinite score")
    for (index, key), (value, detail) in zip(keys, scored):
        record = items[index][1].records[key]
        binary = 1.0 if float(value) >= threshold else 0.0
        record.correctness = binary
        record.score = binary if binarize == "strict" else float(value)
        record.score_detail = detail
