"""Group-relative team advantages -- the centralized half of MAGRPO's CTDE.

A group is the ``G`` joint rollouts sampled from one prompt.  Each rollout earns a
single team reward, and the group-normalized advantage of that rollout is written
onto *every* turn of *every* agent inside it.  The "shared advantage" that makes
this multi-agent rather than three independent GRPO runs is therefore a data-level
broadcast, not anything the loss function has to know about.

Note the deliberate difference from ``mas_grpo_probe/09_build_grpo_agent_dataset.py``:
that pipeline asserts every group member shares an identical prompt, which is right
for its frozen-context resampling. MAGRPO groups must *not* assert that -- joint
rollouts diverge after turn 0, so only ``problem_id`` is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Literal, Sequence

from .trajectory import JointTrajectory

Granularity = Literal["episode", "turn"]


@dataclass
class AdvantageReport:
    n_groups: int = 0
    n_degenerate: int = 0
    n_turns_assigned: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    advantage_abs_mean: float = 0.0
    per_agent_turns: dict[str, int] = field(default_factory=dict)

    @property
    def degenerate_frac(self) -> float:
        return self.n_degenerate / self.n_groups if self.n_groups else 0.0


def group_advantages(
    returns: Sequence[float],
    *,
    clip: float = 4.0,
    std_floor: float = 1e-8,
) -> tuple[list[float], bool]:
    """Standardize ``returns`` within the group.

    Returns ``(advantages, informative)``. A group whose rewards are all equal
    carries no learning signal; it yields all-zero advantages and
    ``informative=False`` so the caller can count and drop it.
    """
    if not returns:
        return [], False
    if len(returns) == 1:
        return [0.0], False
    mu = fmean(returns)
    sigma = pstdev(returns)
    if sigma <= std_floor:
        return [0.0] * len(returns), False
    out = []
    for value in returns:
        adv = (value - mu) / sigma
        out.append(max(-clip, min(clip, adv)))
    return out, True


def turn_returns(step_rewards: Sequence[float], *, gamma: float = 1.0) -> list[float]:
    """Discounted return from each turn onward: ``G_t = sum_{t'>=t} gamma^(t'-t) r_t'``."""
    out = [0.0] * len(step_rewards)
    running = 0.0
    for i in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[i] + gamma * running
        out[i] = running
    return out


def _episode_returns(group: Sequence[JointTrajectory]) -> list[float]:
    return [float(t.team_reward if t.team_reward is not None else 0.0) for t in group]


def assign_advantages(
    group: Sequence[JointTrajectory],
    *,
    granularity: Granularity = "episode",
    gamma: float = 1.0,
    clip: float = 4.0,
    std_floor: float = 1e-8,
) -> tuple[bool, list[float]]:
    """Write ``.advantage`` onto every turn of every rollout in ``group``.

    All members must share a ``problem_id``; their prompts legitimately differ.
    Returns ``(informative, episode_advantages)``.
    """
    if not group:
        return False, []
    problem_ids = {t.problem_id for t in group}
    if len(problem_ids) != 1:
        raise ValueError(f"group spans multiple problems: {sorted(problem_ids)}")

    episode_adv, informative = group_advantages(
        _episode_returns(group), clip=clip, std_floor=std_floor
    )

    if granularity == "episode" or not informative:
        for traj, adv in zip(group, episode_adv):
            for turn in traj.turns:
                turn.advantage = adv
        return informative, episode_adv

    # Turn granularity: normalize within (prompt, turn index) across the rollouts
    # that actually reached that turn. With terminal-only reward and gamma=1 this
    # is identical to episode granularity, so it only bites once step_reward is
    # non-zero (reward_shaping).
    returns_by_traj: list[list[float]] = []
    for traj in group:
        steps = [t.step_reward for t in traj.turns]
        if steps:
            steps[-1] += float(traj.team_reward or 0.0)
        returns_by_traj.append(turn_returns(steps, gamma=gamma))

    max_turns = max((len(r) for r in returns_by_traj), default=0)
    for turn_idx in range(max_turns):
        members = [i for i, r in enumerate(returns_by_traj) if turn_idx < len(r)]
        values = [returns_by_traj[i][turn_idx] for i in members]
        advs, turn_informative = group_advantages(values, clip=clip, std_floor=std_floor)
        for slot, member in enumerate(members):
            # Too few survivors (or no spread) at this depth: fall back to the
            # rollout's episode-level advantage rather than emitting a zero.
            value = advs[slot] if turn_informative else episode_adv[member]
            group[member].turns[turn_idx].advantage = value
    return informative, episode_adv


def summarize_groups(
    groups: Sequence[Sequence[JointTrajectory]],
    *,
    granularity: Granularity = "episode",
    gamma: float = 1.0,
    clip: float = 4.0,
    std_floor: float = 1e-8,
) -> tuple[list[JointTrajectory], AdvantageReport]:
    """Assign advantages across many groups and return the informative rollouts."""
    report = AdvantageReport(n_groups=len(groups))
    kept: list[JointTrajectory] = []
    all_rewards: list[float] = []
    all_advs: list[float] = []

    for group in groups:
        informative, _ = assign_advantages(
            group, granularity=granularity, gamma=gamma, clip=clip, std_floor=std_floor
        )
        all_rewards.extend(_episode_returns(group))
        if not informative:
            report.n_degenerate += 1
            continue
        for traj in group:
            kept.append(traj)
            for turn in traj.turns:
                report.per_agent_turns[turn.agent] = (
                    report.per_agent_turns.get(turn.agent, 0) + 1
                )
                report.n_turns_assigned += 1
                if turn.advantage is not None:
                    all_advs.append(abs(turn.advantage))

    if all_rewards:
        report.reward_mean = fmean(all_rewards)
        report.reward_std = pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
        report.reward_min = min(all_rewards)
        report.reward_max = max(all_rewards)
    if all_advs:
        report.advantage_abs_mean = fmean(all_advs)
    return kept, report
