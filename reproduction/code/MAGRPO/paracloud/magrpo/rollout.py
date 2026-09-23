"""Joint rollout engine.

``synchronous`` is the default because it is what MAGRPO's Dec-POMDP describes:
all N agents act at every timestep from their own observation, and the
environment transitions on the *joint* action. Turn 0 is independent drafting;
each later turn shows every agent all three previous drafts.

``sequential`` reproduces the handoff-over-one-shared-thread shape used by the
existing MAS arms in this repo. It is a different Dec-POMDP (N-1 agents emit
nothing per timestep) and is provided for a controlled comparison where the only
variable is the learning algorithm.

Every turn stores the conversation snapshot taken *before* the call; that is what
the trainer encodes, and it cannot be faithfully reconstructed afterwards.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .protocol import (
    AGENT_IDS,
    answer_of,
    format_joint_observation,
    render_assistant_message,
    repair_instruction,
)
from .tasks.base import Problem, TaskAdapter
from .trajectory import AgentTurn, JointTrajectory


def _derive_seed(base_seed: int, *parts: int) -> int:
    value = base_seed
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
) -> tuple[
    str, dict[str, Any] | None, str | None, str | None, list[dict[str, str]]
]:
    """Generate, parse, and retry once with a repair nudge on malformed output."""
    schema = task.response_schema(agent)
    attempt_messages = [dict(message) for message in messages]
    raw = ""
    parsed = None
    error = None
    finish_reason = None
    for attempt in range(step_retries + 1):
        completion = caller.generate(
            attempt_messages,
            seed=_derive_seed(seed, turn, attempt),
            response_format=schema,
        )
        raw = completion.text
        finish_reason = completion.finish_reason
        parsed, error = task.parse(raw, agent=agent, turn=turn)
        if parsed is not None:
            return raw, parsed, None, finish_reason, attempt_messages
        if attempt < step_retries:
            attempt_messages = [dict(message) for message in messages] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_instruction(error or "invalid")},
            ]
    return raw, None, error, finish_reason, attempt_messages


def run_synchronous(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    rollout_idx: int,
    group_id: str,
    t_max: int,
    seed: int,
    step_retries: int = 1,
    shaping: bool = False,
) -> JointTrajectory:
    """All agents act every turn; the observation is the previous joint response."""
    traj = JointTrajectory(
        problem_id=problem.problem_id,
        group_id=group_id,
        rollout_idx=rollout_idx,
        joint_mode="synchronous",
    )
    user_prompt = task.user_prompt(problem)
    history: list[str] = []

    for turn in range(t_max):
        prompts: dict[str, list[dict[str, str]]] = {}
        for agent in AGENT_IDS:
            messages = [
                {"role": "system", "content": task.system_prompt(agent, joint_mode="synchronous")},
                {"role": "user", "content": user_prompt},
            ]
            for block in history:
                messages.append({"role": "user", "content": block})
            prompts[agent] = messages

        # Agents act simultaneously: no agent sees this turn's peers.
        with ThreadPoolExecutor(max_workers=len(AGENT_IDS)) as pool:
            futures = {
                agent: pool.submit(
                    _call_with_retry,
                    callers[agent],
                    task,
                    prompts[agent],
                    agent=agent,
                    turn=turn,
                    seed=_derive_seed(seed, rollout_idx, AGENT_IDS.index(agent)),
                    step_retries=step_retries,
                )
                for agent in AGENT_IDS
            }
            outcomes = {agent: future.result() for agent, future in futures.items()}

        round_parsed: dict[str, dict[str, Any] | None] = {}
        for agent in AGENT_IDS:
            raw, parsed, error, finish_reason, snapshot = outcomes[agent]
            step = AgentTurn(
                agent=agent,
                turn=turn,
                prompt_messages=snapshot,
                response=raw,
                parsed=parsed,
                protocol_error=error,
                finish_reason=finish_reason,
            )
            if shaping and parsed is not None:
                step.step_reward = task.step_reward(problem, answer_of(parsed))
            traj.turns.append(step)
            round_parsed[agent] = parsed

        history.append(format_joint_observation(round_parsed, turn=turn))
        if all(value is None for value in round_parsed.values()):
            traj.terminated_by = "invalid_response"
            break
    else:
        traj.terminated_by = "turn_limit"

    traj.final_answer = task.aggregate(traj)
    return traj


def run_sequential(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    rollout_idx: int,
    group_id: str,
    t_max: int,
    seed: int,
    start_agent: str = "A1",
    step_retries: int = 1,
    shaping: bool = False,
) -> JointTrajectory:
    """One shared thread; on handoff only the system prompt is swapped."""
    traj = JointTrajectory(
        problem_id=problem.problem_id,
        group_id=group_id,
        rollout_idx=rollout_idx,
        joint_mode="sequential",
    )
    active = start_agent
    messages = [
        {"role": "system", "content": task.system_prompt(active, joint_mode="sequential")},
        {"role": "user", "content": task.user_prompt(problem)},
    ]

    for turn in range(t_max):
        snapshot = [dict(m) for m in messages]
        raw, parsed, error, finish_reason, snapshot = _call_with_retry(
            callers[active],
            task,
            snapshot,
            agent=active,
            turn=turn,
            seed=_derive_seed(seed, rollout_idx, turn),
            step_retries=step_retries,
        )
        step = AgentTurn(
            agent=active,
            turn=turn,
            prompt_messages=snapshot,
            response=raw,
            parsed=parsed,
            protocol_error=error,
            finish_reason=finish_reason,
        )
        if shaping and parsed is not None:
            step.step_reward = task.step_reward(problem, answer_of(parsed))
        traj.turns.append(step)

        if parsed is None:
            traj.terminated_by = "invalid_response"
            break
        messages.append({"role": "assistant", "content": render_assistant_message(parsed)})
        if parsed.get("action") == "confirm_stop":
            traj.terminated_by = "stop"
            break
        target = parsed.get("handoff_target")
        if target not in AGENT_IDS:
            traj.terminated_by = "invalid_response"
            break
        active = target
        messages[0] = {
            "role": "system",
            "content": task.system_prompt(active, joint_mode="sequential"),
        }
    else:
        traj.terminated_by = "turn_limit"

    traj.final_answer = task.aggregate(traj)
    return traj


def run_group(
    task: TaskAdapter,
    problem: Problem,
    callers: dict[str, Any],
    *,
    group_size: int,
    iteration: int,
    t_max: int,
    seed: int,
    joint_mode: str = "synchronous",
    step_retries: int = 1,
    shaping: bool = False,
    max_workers: int = 8,
) -> list[JointTrajectory]:
    """Sample ``group_size`` joint rollouts from one prompt -- the GRPO group."""
    group_id = f"{problem.problem_id}#{iteration}"
    runner = run_synchronous if joint_mode == "synchronous" else run_sequential

    def one(index: int) -> JointTrajectory:
        try:
            return runner(
                task,
                problem,
                callers,
                rollout_idx=index,
                group_id=group_id,
                t_max=t_max,
                seed=_derive_seed(seed, iteration, index),
                step_retries=step_retries,
                shaping=shaping,
            )
        except Exception as exc:  # noqa: BLE001 - one bad rollout must not kill the group
            traj = JointTrajectory(
                problem_id=problem.problem_id,
                group_id=group_id,
                rollout_idx=index,
                joint_mode=joint_mode,
                terminated_by="exception",
                error=f"{type(exc).__name__}: {exc}",
            )
            return traj

    with ThreadPoolExecutor(max_workers=min(max_workers, group_size)) as pool:
        return list(pool.map(one, range(group_size)))


def score_groups(
    task: TaskAdapter,
    groups: Sequence[tuple[Problem, list[JointTrajectory]]],
) -> None:
    """Attach team rewards in one batched grader call (MultiPL-E needs this)."""
    items: list[tuple[Problem, JointTrajectory]] = []
    for problem, group in groups:
        for traj in group:
            items.append((problem, traj))
    if not items:
        return
    for (_, traj), (reward, detail) in zip(items, task.team_reward_batch(items)):
        traj.team_reward = float(reward)
        traj.reward_detail = detail
