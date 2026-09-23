"""Conifer instruction-following, scored by the project's local rule.

Per the user's instruction this uses ``check_constraints``'s ``hard_score``
(``0.75 * explicit_score + 0.25 * requirement_coverage``) and never invokes the
LLM judge -- at train time or eval time.

Two consequences worth keeping in view, both surfaced in the metrics:

* The reward is *dense* in [0, 1], unlike the other four datasets' binary EM, so
  Conifer is the least likely to produce degenerate groups. Prescreening is off.
* ``requirement_coverage`` is lexical overlap and is therefore gameable by
  keyword stuffing. ``eval_metric`` reports ``explicit_score`` and
  ``requirement_coverage`` separately, and tracks answer length, so a divergence
  between the two components -- the signature of reward hacking -- is visible.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from vendor.conifer_protocol import (
    format_problem_as_prompt,
    render_system_prompt,
    response_schema,
)
from vendor.conifer_scoring import check_constraints

from ..protocol import answer_of
from .base import EvalResult, Problem, Split, TaskAdapter

DEFAULT_ROOT = "/data/wangyuheng/jca/conifer_training_hub/01_dataset/processed"
SPLIT_FILES = {"train": "train.jsonl", "eval": "test.jsonl"}


class ConiferTask(TaskAdapter):
    name = "conifer"
    headline_metric = "mean_final_hard_score"

    def __init__(self, *, data_root: str | None = None, **options: Any) -> None:
        super().__init__(data_root=data_root or DEFAULT_ROOT, **options)
        self._rows: dict[str, dict[str, Any]] = {}

    def load(self, split: Split) -> list[Problem]:
        path = Path(self.data_root) / SPLIT_FILES[split]
        if not path.is_file():
            raise FileNotFoundError(f"Conifer {split} split not found: {path}")
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = str(row["problem_id"])
                self._rows[pid] = row
                out.append(
                    Problem(
                        problem_id=pid,
                        raw=row,
                        meta={
                            "split": split,
                            "source_type": row.get("source_type"),
                            "difficulty": row.get("difficulty"),
                        },
                    )
                )
        return out

    def system_prompt(self, agent: str) -> str:
        return render_system_prompt(agent)

    def user_prompt(self, problem: Problem) -> str:
        # include_reference MUST stay False -- the gold answer must never leak
        # into the prompt.
        return format_problem_as_prompt(self._rows[problem.problem_id], include_reference=False)

    def response_schema(self, agent: str) -> dict[str, Any] | None:
        return response_schema(agent, strict=True)

    def aggregate_answers(self, answers: dict[str, str | None]) -> str | None:
        """Take A3's answer; majority voting on free-form prose is meaningless."""
        # A3 is the largest model and has seen A1/A2's drafts; fall back down the
        # ladder only if it produced nothing.
        for agent in ("A3", "A2", "A1"):
            if answers.get(agent):
                return answers[agent]
        return None

    def _score(self, problem: Problem, answer: str | None) -> tuple[float, dict]:
        checks = check_constraints(self._rows[problem.problem_id], answer or "")
        return float(checks["hard_score"]), {
            "hard_score": float(checks["hard_score"]),
            "explicit_score": float(checks["explicit_score"]),
            "requirement_coverage": float(checks["requirement_coverage"]),
            "all_explicit_passed": bool(checks["all_explicit_passed"]),
            "word_count": int(checks["word_count"]),
        }


    def step_reward(self, problem: Problem, answer: str) -> float:
        return 0.2 * self._score(problem, answer)[0]

    def eval_metric(self, results: Sequence[EvalResult]) -> dict[str, Any]:
        if not results:
            return {"mean_final_hard_score": 0.0, "n": 0}

        def avg(key: str) -> float:
            return fmean(float(r.detail.get(key, 0.0)) for r in results)

        return {
            "mean_final_hard_score": fmean(r.score for r in results),
            # Reported separately so keyword stuffing (coverage climbing while
            # explicit stalls) is legible rather than hidden inside hard_score.
            "mean_explicit_score": avg("explicit_score"),
            "mean_requirement_coverage": avg("requirement_coverage"),
            "all_explicit_pass_rate": fmean(
                1.0 if r.detail.get("all_explicit_passed") else 0.0 for r in results
            ),
            "mean_word_count": avg("word_count"),
            "n": len(results),
            "answered": sum(1 for r in results if r.final_answer),
        }
