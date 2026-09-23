"""MultiPL-E 8-language code generation, scored by real unit-test execution.

Two things make this adapter different from the other four:

* The protocol fields are renamed (``tentative_completion`` /
  ``confirmed_completion``); :attr:`allow_field_aliases` maps them back so the
  strict validator still applies.
* The reward requires running the tests in docker. ``team_reward_batch`` writes
  every completion in an iteration and makes ONE ``evaluate_multipl_e_parallel.py``
  call, which is why the interface is batched.

``reward_mode="proxy"`` exists only for when docker is unavailable. It is
strictly weaker than the real signal, so every artifact produced under it is
stamped, and the stamp is meant to stop the number being quoted as comparable.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from vendor.multipl_e_completion_adapter import build_chat_user_prompt, normalize_completion

from ..protocol import AGENT_IDS
from ..trajectory import JointTrajectory
from .base import EvalResult, Problem, Split, TaskAdapter

DEFAULT_ROOT = (
    "/data/wangyuheng/jca/Code/multipl_e_8lang_benchmark/splits/"
    "unified_train70_test30_seed7658190907657085414"
)
EVALUATOR = "/data/wangyuheng/jca/Code/MultiPL-E/scripts/evaluate_multipl_e_parallel.py"
DEFAULT_IMAGE = "multipl-e-evaluation:jca-current"
SPLIT_FILES = {"train": "train.jsonl", "eval": "test.jsonl"}

SYSTEM_PROMPT = """You are agent {agent} collaborating with {others} to write one correct program.

You will be shown a function signature and its tests. Produce the function body that
passes the tests.

Reply with ONLY one JSON object with exactly these keys:
  "reasoning"             - brief analysis of the task and of the other agents' drafts
  "tentative_completion"  - your current best completion (the code continuation)
  "action"                - "handoff" or "confirm_stop"
  "handoff_target"        - another agent id when action is "handoff", else null
  "handoff_note"          - short note to that agent when handing off, else null
  "confirmed_completion"  - the final completion when action is "confirm_stop", else null

Emit code only inside the completion fields. No prose outside the JSON object.
"""


class MultiplETask(TaskAdapter):
    name = "multipl_e"
    headline_metric = "weighted_pass_at_1"
    allow_field_aliases = True

    def __init__(
        self,
        *,
        data_root: str | None = None,
        reward_mode: str = "execution",
        image: str = DEFAULT_IMAGE,
        workers: int = 16,
        scratch_dir: str | None = None,
        evaluator: str = EVALUATOR,
        docker_exec: str = "docker",
        **options: Any,
    ) -> None:
        super().__init__(data_root=data_root or DEFAULT_ROOT, **options)
        if reward_mode not in {"execution", "proxy"}:
            raise ValueError(f"reward_mode must be execution|proxy, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.image = image
        self.workers = workers
        self.scratch_dir = scratch_dir
        self.evaluator = evaluator
        self.docker_exec = docker_exec
        self._rows: dict[str, dict[str, Any]] = {}

    # -- data ---------------------------------------------------------------
    def load(self, split: Split) -> list[Problem]:
        path = Path(self.data_root) / SPLIT_FILES[split]
        if not path.is_file():
            raise FileNotFoundError(f"MultiPL-E {split} split not found: {path}")
        out = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # name repeats across languages, so the id must include both.
                pid = f"{row['language']}::{row['name']}"
                self._rows[pid] = row
                out.append(
                    Problem(
                        problem_id=pid,
                        raw=row,
                        meta={
                            "split": split,
                            "language": row["language"],
                            "dataset": row["dataset"],
                            "name": row["name"],
                        },
                    )
                )
        return out

    # -- prompting ----------------------------------------------------------
    def system_prompt(self, agent: str, *, joint_mode: str) -> str:
        others = " and ".join(a for a in AGENT_IDS if a != agent)
        return SYSTEM_PROMPT.format(agent=agent, others=others)

    def user_prompt(self, problem: Problem) -> str:
        row = self._rows[problem.problem_id]
        return build_chat_user_prompt(
            row["language"], row["prompt"], row["tests"], row.get("stop_tokens") or []
        )

    def _normalize(self, problem: Problem, answer: str | None) -> str:
        if not answer:
            return ""
        row = self._rows[problem.problem_id]
        return normalize_completion(
            answer,
            row["prompt"],
            row["tests"],
            row.get("stop_tokens") or [],
            row["language"],
        )

    def normalize_for_vote(self, answer: str) -> str:
        return " ".join(str(answer or "").split())

    # -- reward -------------------------------------------------------------
    def team_reward_batch(
        self, items: Sequence[tuple[Problem, JointTrajectory]]
    ) -> list[tuple[float, dict[str, Any]]]:
        if not items:
            return []
        if self.reward_mode == "proxy":
            return [self._proxy_score(p, t.final_answer) for p, t in items]
        return self._execution_scores(items)

    def _execution_scores(
        self, items: Sequence[tuple[Problem, JointTrajectory]]
    ) -> list[tuple[float, dict[str, Any]]]:
        """Write every completion, run docker once, read the results back."""
        if self.scratch_dir:
            Path(self.scratch_dir).mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="magrpo_multipl_e_", dir=self.scratch_dir))
        input_dir = scratch / "completions"
        output_dir = scratch / "evaluated"
        try:
            index: list[tuple[int, Path]] = []
            for slot, (problem, traj) in enumerate(items):
                row = self._rows[problem.problem_id]
                completion = self._normalize(problem, traj.final_answer)
                # One directory per item: `name` collides across rollouts of the
                # same problem, and the evaluator keys results off the filename.
                item_dir = input_dir / f"item_{slot:05d}" / row["dataset"] / row["language"]
                item_dir.mkdir(parents=True, exist_ok=True)
                target = item_dir / f"{row['name']}.json.gz"
                payload = {
                    "name": row["name"],
                    "language": row["language"],
                    "prompt": row["prompt"],
                    "tests": row["tests"],
                    "stop_tokens": row.get("stop_tokens") or [],
                    "completions": [completion],
                }
                with gzip.open(target, "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                index.append((slot, target))

            result = subprocess.run(
                [
                    sys.executable, self.evaluator,
                    "--input-dir", str(input_dir),
                    "--output-dir", str(output_dir),
                    "--image", self.image,
                    "--docker-exec", self.docker_exec,
                    "--shards", str(self.workers),
                    "--inner-workers", "1",
                ],
                capture_output=True,
                text=True,
                timeout=self.options.get("exec_timeout_s", 3600),
            )
            if result.returncode:
                raise RuntimeError(
                    f"MultiPL-E evaluator failed ({result.returncode}): "
                    f"{result.stderr[-4000:] or result.stdout[-4000:]}"
                )

            out: list[tuple[float, dict[str, Any]]] = []
            for slot, target in index:
                out.append(self._read_result(output_dir / target.relative_to(input_dir)))
            return out
        finally:
            if self.scratch_dir is None:
                shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def _read_result(completion_path: Path) -> tuple[float, dict[str, Any]]:
        results_path = completion_path.with_name(
            completion_path.name.replace(".json.gz", ".results.json.gz")
        )
        if not results_path.is_file():
            raise RuntimeError(f"MultiPL-E evaluator omitted results: {results_path}")
        with gzip.open(results_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results") or []
        if not results:
            raise RuntimeError(f"MultiPL-E evaluator returned empty results: {results_path}")
        first = results[0]
        # The official predicate from summarize_multipl_e_sas.py:118-121.
        passed = first.get("status") == "OK" and first.get("exit_code") == 0
        return (1.0 if passed else 0.0), {
            "executed": True,
            "status": first.get("status"),
            "exit_code": first.get("exit_code"),
        }

    def _proxy_score(
        self, problem: Problem, answer: str | None
    ) -> tuple[float, dict[str, Any]]:
        """Cheap syntactic stand-in. Never comparable to pass@1 -- always stamped."""
        completion = self._normalize(problem, answer)
        detail: dict[str, Any] = {"reward_mode": "proxy", "executed": False}
        if not completion.strip():
            return 0.0, {**detail, "reason": "empty completion"}
        score = 0.5
        opens = sum(completion.count(c) for c in "([{")
        closes = sum(completion.count(c) for c in ")]}")
        if opens == closes:
            score += 0.3
        if self._rows[problem.problem_id]["language"] == "py":
            import ast

            try:
                ast.parse(self._rows[problem.problem_id]["prompt"] + completion)
                score += 0.2
            except SyntaxError:
                pass
        else:
            score += 0.2
        return min(1.0, score), {**detail, "score": score}

    # -- metric -------------------------------------------------------------
    def eval_metric(self, results: Sequence[EvalResult]) -> dict[str, Any]:
        if not results:
            return {"weighted_pass_at_1": 0.0, "n": 0}
        by_language: dict[str, list[float]] = {}
        for result in results:
            row = self._rows.get(result.problem_id)
            language = str(row["language"]) if row else "unknown"
            by_language.setdefault(language, []).append(result.score)
        per_language = {k: fmean(v) for k, v in sorted(by_language.items())}
        out = {
            # Weighted = per-problem mean; macro = mean over languages.
            "weighted_pass_at_1": fmean(r.score for r in results),
            "macro_language_pass_at_1": fmean(per_language.values()),
            "pass_at_1_by_language": per_language,
            "n": len(results),
            "reward_mode": self.reward_mode,
        }
        if self.reward_mode == "proxy":
            out["WARNING"] = "proxy reward: NOT comparable to official pass@1"
        return out
