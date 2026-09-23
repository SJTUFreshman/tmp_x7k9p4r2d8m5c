"""The uniform interface every dataset plugs into.

Five datasets with very different graders (numeric tolerance, symbolic
equivalence, token F1, constraint checking, containerized unit tests) are reduced
to one contract so ``rollout.py`` and ``loop.py`` never branch on dataset.

``score_answers`` is batched rather than per-item purely so MultiPL-E can
amortize one docker invocation across a whole iteration; the other four implement
it as a loop.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..protocol import AGENT_IDS, answer_of, parse_protocol

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
    def system_prompt(self, agent: str) -> str: ...

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
    def aggregate_answers(self, answers: dict[str, str | None]) -> str | None:
        """Reduce one turn's answers, keyed by agent, to a single team answer.

        Majority vote under :meth:`normalize_for_vote`, breaking ties toward the
        largest model (A3). Deliberately identical in behaviour to the MAGRPO
        baseline's ``aggregate`` so the two baselines' team numbers stay
        comparable; only the input shape differs (MAPoRL has a per-(turn, agent)
        table rather than a step list). Conifer overrides it -- majority voting
        on free-form prose is meaningless.
        """
        candidates = {a: v for a, v in answers.items() if v}
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
    def score_answers(
        self, items: Sequence[tuple[Problem, str | None]]
    ) -> list[tuple[float, dict[str, Any]]]:
        """Grade a flat list of (problem, answer) pairs.

        MAPoRL needs a score for every (turn, agent) cell, not one per rollout,
        so this is the primary scoring entry point. Batched for the same reason
        ``score_answers`` is batched: MultiPL-E amortizes one docker invocation over
        the whole list.

        Duplicate ``problem_id`` values are expected and must be handled --
        ``agent_num * round_num`` answers share each problem.
        """
        return [self._score(problem, answer) for problem, answer in items]

    def _score(
        self, problem: Problem, answer: str | None
    ) -> tuple[float, dict[str, Any]]:  # pragma: no cover - subclasses override
        raise NotImplementedError(
            f"{type(self).__name__} must implement _score or override score_answers"
        )


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
        data_root = os.environ.get(f"MAPORL_{name.upper()}_DATA_ROOT")
        if data_root:
            options["data_root"] = data_root
    return TASK_REGISTRY[name](**options)
