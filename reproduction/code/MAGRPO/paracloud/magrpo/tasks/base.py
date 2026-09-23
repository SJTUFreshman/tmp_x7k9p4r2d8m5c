"""The uniform interface every dataset plugs into.

Five datasets with very different graders (numeric tolerance, symbolic
equivalence, token F1, constraint checking, containerized unit tests) are reduced
to one contract so ``rollout.py`` and ``loop.py`` never branch on dataset.

``team_reward_batch`` is batched rather than per-item purely so MultiPL-E can
amortize one docker invocation across a whole iteration; the other four implement
it as a loop.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..protocol import AGENT_IDS, answer_of, parse_protocol
from ..trajectory import JointTrajectory

Split = Literal["train", "eval"]


@dataclass(frozen=True)
class Problem:
    problem_id: str
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    meta: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class EvalResult:
    problem_id: str
    final_answer: str | None
    score: float
    detail: dict[str, Any] = field(default_factory=dict)


class TaskAdapter(abc.ABC):
    """One dataset: how to load it, prompt it, parse it, and score it."""

    name: str = "base"
    headline_metric: str = "score"
    # MultiPL-E overrides this; it renames two protocol fields.
    allow_field_aliases: bool = False

    def __init__(self, *, data_root: str | None = None, **options: Any) -> None:
        self.data_root = data_root
        self.options = options

    # -- data ---------------------------------------------------------------
    @abc.abstractmethod
    def load(self, split: Split) -> list[Problem]: ...

    # -- prompting ----------------------------------------------------------
    @abc.abstractmethod
    def system_prompt(self, agent: str, *, joint_mode: str) -> str: ...

    @abc.abstractmethod
    def user_prompt(self, problem: Problem) -> str: ...

    def response_schema(self, agent: str) -> dict[str, Any] | None:
        """Guided-decoding schema, when the dataset supplies one."""
        return None

    # -- parsing ------------------------------------------------------------
    def parse(
        self, raw: str, *, agent: str, turn: int
    ) -> tuple[dict[str, Any] | None, str | None]:
        return parse_protocol(
            raw, active_agent=agent, allow_aliases=self.allow_field_aliases
        )

    # -- reduction ----------------------------------------------------------
    def aggregate(self, traj: JointTrajectory) -> str | None:
        """Reduce the final joint response to a single answer string.

        Default: majority vote over the last turn's answers under
        :meth:`normalize_for_vote`, breaking ties toward the largest model (A3).
        Conifer overrides this -- voting on free-form prose is meaningless.
        """
        if not traj.turns:
            return None
        if traj.joint_mode == "sequential":
            # The thread ends when some agent confirms; that answer is the output.
            for turn in reversed(traj.turns):
                if turn.parsed:
                    return answer_of(turn.parsed)
            return None

        last_round = max(t.turn for t in traj.turns)
        finals = {
            t.agent: answer_of(t.parsed)
            for t in traj.turns
            if t.turn == last_round and t.parsed
        }
        candidates = {a: v for a, v in finals.items() if v}
        if not candidates:
            return None

        buckets: dict[str, list[str]] = {}
        for agent, answer in candidates.items():
            buckets.setdefault(self.normalize_for_vote(answer), []).append(agent)
        best = max(
            buckets.items(),
            # Most votes wins; ties go to the bucket containing the largest model.
            key=lambda kv: (len(kv[1]), max(AGENT_IDS.index(a) for a in kv[1])),
        )
        winning_agent = max(best[1], key=AGENT_IDS.index)
        return candidates[winning_agent]

    def normalize_for_vote(self, answer: str) -> str:
        """Equivalence key used when counting votes."""
        return " ".join(str(answer or "").split()).casefold()

    # -- reward and metric --------------------------------------------------
    @abc.abstractmethod
    def team_reward_batch(
        self, items: Sequence[tuple[Problem, JointTrajectory]]
    ) -> list[tuple[float, dict[str, Any]]]:
        """Shared team reward in [0, 1] per joint rollout, plus audit detail."""

    def step_reward(self, problem: Problem, answer: str) -> float:
        """Optional per-turn shaping. Zero unless reward_shaping is enabled."""
        return 0.0

    @abc.abstractmethod
    def eval_metric(self, results: Sequence[EvalResult]) -> dict[str, Any]: ...


def build_task(name: str, **options: Any) -> TaskAdapter:
    from . import TASK_REGISTRY

    if name not in TASK_REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASK_REGISTRY)}")
    if "data_root" not in options:
        data_root = os.environ.get(f"MAGRPO_{name.upper()}_DATA_ROOT")
        if data_root:
            options["data_root"] = data_root
    return TASK_REGISTRY[name](**options)
