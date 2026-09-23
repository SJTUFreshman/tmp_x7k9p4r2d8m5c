"""Agent- and turn-wise group-relative advantages.

AT-GRPO's first component (the second is tree sampling, in ``tree_rollout.py``).
Every ``(agent, turn, branch point)`` gets its own advantage group, normalized
within the group:

    A_k = clip( (R_k - mean_k R) / std_k R , +-clip )

The contrast with the MAGRPO baseline in this repo is exact and worth stating,
because the two make opposite demands:

* MAGRPO groups G *joint rollouts* of one prompt and broadcasts one team
  advantage to every agent and turn. Prompts inside a group necessarily diverge
  after turn 0, so it must **not** check prompt identity.
* AT-GRPO groups K *candidate actions* sampled from one frozen state, so every
  member shares a byte-identical prompt. That identity is the whole point --
  it is what makes the group a valid GRPO group -- so here it **is** asserted.

Getting that backwards is silent: you would still get numbers, they just would
not be group-relative to anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Sequence

from .tree_rollout import Candidate


@dataclass
class ATAdvantageReport:
    n_groups: int = 0
    n_degenerate: int = 0
    n_candidates: int = 0
    n_assigned: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    advantage_abs_mean: float = 0.0
    per_agent: dict[str, int] = field(default_factory=dict)
    per_turn: dict[int, int] = field(default_factory=dict)
    # Groups whose members disagreed on the prompt. Must stay zero.
    n_fingerprint_violations: int = 0

    @property
    def degenerate_frac(self) -> float:
        return self.n_degenerate / self.n_groups if self.n_groups else 0.0


def group_advantages(
    rewards: Sequence[float],
    *,
    clip: float = 4.0,
    std_floor: float = 1e-8,
) -> tuple[list[float], bool]:
    """Standardize within a group. ``informative=False`` when the group is flat."""
    if not rewards:
        return [], False
    if len(rewards) == 1:
        # The single-member case AT-GRPO exists to avoid: the baseline is the
        # sample itself, so the advantage is necessarily zero.
        return [0.0], False
    mu = fmean(rewards)
    sigma = pstdev(rewards)
    if sigma <= std_floor:
        return [0.0] * len(rewards), False
    return [max(-clip, min(clip, (r - mu) / sigma)) for r in rewards], True


def assign_group(
    group: Sequence[Candidate],
    *,
    clip: float = 4.0,
    std_floor: float = 1e-8,
    strict_fingerprint: bool = True,
) -> tuple[bool, bool]:
    """Write ``.advantage`` onto one (agent, turn) group.

    Returns ``(informative, fingerprint_ok)``.
    """
    if not group:
        return False, True

    agents = {c.agent for c in group}
    turns = {c.turn for c in group}
    if len(agents) != 1 or len(turns) != 1:
        raise ValueError(
            f"group spans multiple (agent, turn) cells: agents={sorted(agents)} "
            f"turns={sorted(turns)}"
        )

    fingerprints = {c.prompt_fingerprint for c in group}
    fingerprint_ok = len(fingerprints) == 1
    if not fingerprint_ok and strict_fingerprint:
        raise ValueError(
            f"group {group[0].group_id!r} has {len(fingerprints)} distinct prompts; "
            "AT-GRPO groups must be sampled from one frozen state, otherwise the "
            "group-relative advantage compares incomparable things"
        )

    rewards = [float(c.reward if c.reward is not None else 0.0) for c in group]
    advantages, informative = group_advantages(rewards, clip=clip, std_floor=std_floor)
    for candidate, advantage in zip(group, advantages):
        candidate.advantage = advantage
    return informative, fingerprint_ok


def summarize(
    rollouts: Sequence,
    *,
    clip: float = 4.0,
    std_floor: float = 1e-8,
    strict_fingerprint: bool = True,
) -> tuple[list[Candidate], ATAdvantageReport]:
    """Assign advantages across every group and return the trainable candidates."""
    report = ATAdvantageReport()
    kept: list[Candidate] = []
    all_rewards: list[float] = []
    all_advantages: list[float] = []

    for rollout in rollouts:
        for group in rollout.groups().values():
            report.n_groups += 1
            report.n_candidates += len(group)
            all_rewards.extend(
                float(c.reward if c.reward is not None else 0.0) for c in group
            )
            informative, fingerprint_ok = assign_group(
                group, clip=clip, std_floor=std_floor,
                strict_fingerprint=strict_fingerprint,
            )
            if not fingerprint_ok:
                report.n_fingerprint_violations += 1
            if not informative:
                report.n_degenerate += 1
                continue
            for candidate in group:
                # Unparseable turns have no usable training target.
                if candidate.parsed is None or candidate.advantage is None:
                    continue
                kept.append(candidate)
                report.n_assigned += 1
                report.per_agent[candidate.agent] = (
                    report.per_agent.get(candidate.agent, 0) + 1
                )
                report.per_turn[candidate.turn] = (
                    report.per_turn.get(candidate.turn, 0) + 1
                )
                all_advantages.append(abs(candidate.advantage))

    if all_rewards:
        report.reward_mean = fmean(all_rewards)
        report.reward_std = pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
        report.reward_min = min(all_rewards)
        report.reward_max = max(all_rewards)
    if all_advantages:
        report.advantage_abs_mean = fmean(all_advantages)
    return kept, report


def rows_by_agent(
    candidates: Sequence[Candidate], agents: Sequence[str]
) -> dict[str, list[dict]]:
    """Partition trainable candidates into per-agent training rows."""
    out: dict[str, list[dict]] = {agent: [] for agent in agents}
    for candidate in candidates:
        out.setdefault(candidate.agent, []).append(
            {
                "prompt_messages": candidate.prompt_messages,
                "response": candidate.response,
                "advantage": candidate.advantage,
            }
        )
    return out
