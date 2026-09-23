"""MuSiQue multi-hop QA, graded by the official EM/F1 normalization."""
from __future__ import annotations

from statistics import fmean
from typing import Any, Sequence

from vendor.musique_data import format_problem_as_prompt, load_musique
from vendor.musique_grader import compute_em_f1, extract_boxed_answer, normalize_answer
from vendor.musique_prompt import render_old_system_prompt

from ..trajectory import JointTrajectory
from .base import EvalResult, Problem, Split, TaskAdapter

DEFAULT_ROOT = "/data/wangyuheng/jca/musique_data"
# The canonical eval run forces EVAL_PROTOCOL=old_sft against the dev split.
SPLIT_NAMES = {"train": "train", "eval": "dev"}


class MuSiQueTask(TaskAdapter):
    name = "musique"
    headline_metric = "em"

    def __init__(self, *, data_root: str | None = None, **options: Any) -> None:
        super().__init__(data_root=data_root or DEFAULT_ROOT, **options)
        self._problems: dict[str, Any] = {}

    def load(self, split: Split) -> list[Problem]:
        problems = load_musique(SPLIT_NAMES[split], data_dir=self.data_root)
        out = []
        for problem in problems:
            self._problems[problem.id] = problem
            out.append(
                Problem(
                    problem_id=problem.id,
                    raw={"question": problem.question, "answer": problem.answer},
                    meta={"split": split, "hop": problem.hop},
                )
            )
        return out

    def system_prompt(self, agent: str, *, joint_mode: str) -> str:
        return render_old_system_prompt(agent)

    def user_prompt(self, problem: Problem) -> str:
        return format_problem_as_prompt(self._problems[problem.problem_id])

    def normalize_for_vote(self, answer: str) -> str:
        return normalize_answer(extract_boxed_answer(answer) or answer)

    def _score(self, problem: Problem, answer: str | None) -> tuple[float, dict]:
        if not answer:
            return 0.0, {"reason": "no answer"}
        em, f1 = compute_em_f1(answer, self._problems[problem.problem_id])
        return float(em), {"em": float(em), "f1": float(f1)}

    def team_reward_batch(
        self, items: Sequence[tuple[Problem, JointTrajectory]]
    ) -> list[tuple[float, dict[str, Any]]]:
        return [self._score(p, t.final_answer) for p, t in items]

    def step_reward(self, problem: Problem, answer: str) -> float:
        return 0.2 * self._score(problem, answer)[0]

    def eval_metric(self, results: Sequence[EvalResult]) -> dict[str, Any]:
        if not results:
            return {"em": 0.0, "n": 0}
        return {
            "em": fmean(r.score for r in results),
            "f1": fmean(float(r.detail.get("f1", 0.0)) for r in results),
            "n": len(results),
            "answered": sum(1 for r in results if r.final_answer),
        }
