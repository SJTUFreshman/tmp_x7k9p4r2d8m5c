"""MATH competition problems, graded by math_eval_symbolic_v4.

Module is ``math_comp`` rather than ``math`` on purpose: a package named ``math``
shadows the stdlib module for anything importing it (the repo already has this
problem in ``analysis/math``).

The train split comes from the parquet tree; the canonical eval set is
``shard_04.jsonl``, drawn from the *test* parquet, so the two never overlap.
``load_math_problems`` only reads parquet, so the shard is materialized back into
a parquet tree by the vendored ``build_eval_shard_root.py``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from vendor.math_eval_v4 import (
    MATH_EVAL_VERSION,
    compute_math_em,
    extract_last_boxed,
    fallback_answer,
    format_math_problem_as_prompt,
    load_math_problems,
    math_soft_f1,
    normalize_math_answer,
    render_math_mas_system_prompt,
)

from ..trajectory import JointTrajectory
from .base import EvalResult, Problem, Split, TaskAdapter

DEFAULT_ROOT = "/data/wangyuheng/jca/Math/data/MATH"
DEFAULT_EVAL_SHARD = "splits/test_10x500_seed42/shard_04.jsonl"
VENDOR_BUILDER = Path(__file__).resolve().parents[2] / "vendor" / "build_eval_shard_root.py"


class MathTask(TaskAdapter):
    name = "math"
    headline_metric = "em"
    coerce_numeric_answers = True

    def __init__(
        self,
        *,
        data_root: str | None = None,
        eval_shard: str | None = None,
        shard_cache: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(data_root=data_root or DEFAULT_ROOT, **options)
        self.eval_shard = eval_shard or DEFAULT_EVAL_SHARD
        self.shard_cache = Path(
            shard_cache
            or Path(__file__).resolve().parents[2] / "runs" / "_shared" / "math_eval_root"
        )
        self._problems: dict[str, Any] = {}

    def _materialize_eval_root(self) -> Path:
        """Rebuild the shard's parquet tree once, then reuse it across runs."""
        root = Path(self.data_root)
        shard = root / self.eval_shard
        if not shard.is_file():
            raise FileNotFoundError(f"MATH eval shard not found: {shard}")
        marker = self.shard_cache / ".complete"
        if marker.is_file() and marker.read_text().strip() == self.eval_shard:
            return self.shard_cache
        # The builder refuses to write into an existing output root, so hand it a
        # fresh path and only publish on success. This also makes a half-built
        # tree from an interrupted run impossible to mistake for a complete one.
        shutil.rmtree(self.shard_cache, ignore_errors=True)
        self.shard_cache.parent.mkdir(parents=True, exist_ok=True)
        staging = self.shard_cache.with_name(self.shard_cache.name + ".building")
        shutil.rmtree(staging, ignore_errors=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(VENDOR_BUILDER),
                "--source-root", str(root),
                "--shard-file", str(shard),
                "--output-root", str(staging),
                "--manifest", str(staging / "manifest.json"),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"MATH shard materialization failed ({proc.returncode}):\n"
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        staging.replace(self.shard_cache)
        marker.write_text(self.eval_shard, encoding="utf-8")
        return self.shard_cache

    def load(self, split: Split) -> list[Problem]:
        if split == "train":
            problems = load_math_problems(Path(self.data_root), split="train")
        else:
            problems = load_math_problems(self._materialize_eval_root(), split="test")
        out = []
        for problem in problems:
            self._problems[problem.problem_id] = problem
            out.append(
                Problem(
                    problem_id=problem.problem_id,
                    raw={"problem": problem.prompt, "gold_answer": problem.gold_answer},
                    meta={
                        "split": split,
                        "subject": problem.subject,
                        "level": problem.level,
                    },
                )
            )
        return out

    def system_prompt(self, agent: str, *, joint_mode: str) -> str:
        return render_math_mas_system_prompt(agent)

    def user_prompt(self, problem: Problem) -> str:
        return format_math_problem_as_prompt(self._problems[problem.problem_id])

    def normalize_for_vote(self, answer: str) -> str:
        extracted = extract_last_boxed(answer) or fallback_answer(answer) or answer
        return normalize_math_answer(extracted)

    def _score(self, problem: Problem, answer: str | None) -> tuple[float, dict]:
        if not answer:
            return 0.0, {"reason": "no answer"}
        gold = self._problems[problem.problem_id].gold_answer
        # compute_math_em handles boxed input itself, but extracting first makes
        # the audit trail show what was actually compared.
        extracted = extract_last_boxed(answer) or fallback_answer(answer) or answer
        em = compute_math_em(extracted, gold)
        # Soft F1 gives partial credit on numeric answers via a sign-aware
        # min/max ratio; non-numeric golds (prose, intervals) are excluded from
        # the numeric subset rather than scored as zero.
        soft_f1, numeric_eligible, numeric_score = math_soft_f1(extracted, gold)
        # The reward stays EM: partial credit on a wrong number would teach the
        # policy that being close is good enough.
        return float(em), {
            "em": float(em),
            "f1": float(soft_f1),
            "numeric_eligible": bool(numeric_eligible),
            "numeric_f1": numeric_score,
            "extracted": extracted,
            "gold": gold,
            "grader": MATH_EVAL_VERSION,
        }

    def team_reward_batch(
        self, items: Sequence[tuple[Problem, JointTrajectory]]
    ) -> list[tuple[float, dict[str, Any]]]:
        return [self._score(p, t.final_answer) for p, t in items]

    def step_reward(self, problem: Problem, answer: str) -> float:
        return 0.2 * self._score(problem, answer)[0]

    def eval_metric(self, results: Sequence[EvalResult]) -> dict[str, Any]:
        if not results:
            return {"em": 0.0, "n": 0}
        by_subject: dict[str, list[float]] = {}
        by_level: dict[str, list[float]] = {}
        for result in results:
            problem = self._problems.get(result.problem_id)
            if problem is not None:
                by_subject.setdefault(str(problem.subject), []).append(result.score)
                by_level.setdefault(str(problem.level), []).append(result.score)
        numeric = [
            float(r.detail["numeric_f1"])
            for r in results
            if r.detail.get("numeric_eligible") and r.detail.get("numeric_f1") is not None
        ]
        return {
            "em": fmean(r.score for r in results),
            "f1": fmean(float(r.detail.get("f1", 0.0)) for r in results),
            # Restricted to problems whose gold answer is a finite real scalar,
            # which is the only subset where partial credit is meaningful.
            "numeric_f1": fmean(numeric) if numeric else 0.0,
            "numeric_n": len(numeric),
            "n": len(results),
            "answered": sum(1 for r in results if r.final_answer),
            "grader": MATH_EVAL_VERSION,
            "em_by_subject": {k: fmean(v) for k, v in sorted(by_subject.items())},
            "em_by_level": {k: fmean(v) for k, v in sorted(by_level.items())},
        }
