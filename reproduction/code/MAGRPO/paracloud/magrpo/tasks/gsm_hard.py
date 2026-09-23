"""GSM-Hard: numeric answers graded by tolerance-based exact match."""
from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from vendor.gsm_agents import render_system_prompt
from vendor.gsm_data import extract_number, format_problem_as_prompt, load_gsm_hard
from vendor.gsm_grader import compute_em_f1

from ..trajectory import JointTrajectory
from .base import EvalResult, Problem, Split, TaskAdapter

DEFAULT_ROOT = "/data/wangyuheng/jca/Math/data/GSM-HARD"
SPLIT_FILES = {
    "train": "splits/gsmhardv2_train.jsonl",
    "eval": "splits/gsmhardv2_dev.jsonl",
}


class GSMHardTask(TaskAdapter):
    name = "gsm_hard"
    headline_metric = "em"

    def __init__(self, *, data_root: str | None = None, **options: Any) -> None:
        super().__init__(data_root=data_root or DEFAULT_ROOT, **options)
        self._problems: dict[str, Any] = {}

    def load(self, split: Split) -> list[Problem]:
        path = Path(self.data_root) / SPLIT_FILES[split]
        if not path.is_file():
            raise FileNotFoundError(f"GSM-Hard {split} split not found: {path}")
        out = []
        for problem in load_gsm_hard(str(path)):
            self._problems[problem.id] = problem
            out.append(
                Problem(
                    problem_id=problem.id,
                    raw={"question": problem.question, "answer": problem.answer_str},
                    meta={"split": split},
                )
            )
        return out

    def system_prompt(self, agent: str, *, joint_mode: str) -> str:
        return render_system_prompt(agent)

    def user_prompt(self, problem: Problem) -> str:
        return format_problem_as_prompt(self._problems[problem.problem_id])

    def normalize_for_vote(self, answer: str) -> str:
        value = extract_number(answer)
        # Bucket by parsed value so "42", "42.0" and "The answer is 42" agree.
        return "nan" if value is None else repr(round(float(value), 9))

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
