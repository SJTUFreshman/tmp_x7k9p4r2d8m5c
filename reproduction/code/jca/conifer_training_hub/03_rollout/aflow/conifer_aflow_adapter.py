"""Configure the MuSiQue AFlow implementation for Conifer.

The MuSiQue AFlow sandbox (`workflow.py`) and MCTS driver (`optimizer.py`) are
task-agnostic; only the operator prompts, the problem view, and the scoring are
MuSiQue-specific.  This module monkeypatches those three seams, exactly like
`baseline/GSM-Hard/AFlow/aflow_gsmhard_adapter.py` does for GSM-Hard, so that
none of the shared AFlow code has to be edited.

Scoring parity with the other datasets (see AFLOW_SEARCH_MIGRATION.md 3.1):

    dev_em  <- check_constraints(...)["hard_score"]            drives UCB
    dev_f1  <- check_constraints(...)["reference_lexical_f1"]  tie-break only

`hard_score` is Conifer's reported metric, so the search optimizes exactly what
is reported, the same way MuSiQue/GSM-Hard/MATH search on EM.  No composite
score, no compensation for the fact that `hard_score` saturates.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Type

THIS_DIR = Path(__file__).resolve().parent
HUB_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = HUB_ROOT.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_AFLOW_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "AFlow"

for _path in (
    PACKAGE_PARENT,
    PROJECT_ROOT,
    MUSIQUE_AFLOW_DIR,
    HUB_ROOT / "02_protocol",
    HUB_ROOT / "04_judge",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from conifer_scoring import check_constraints  # noqa: E402

METRIC_NAME = "conifer_hard_score_v1 (em=hard_score, f1=reference_lexical_f1)"

INITIAL_WORKFLOW = THIS_DIR / "workflows" / "round_00_initial.py"


# ---------------------------------------------------------------------------
# Problem view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConiferProblem:
    """The view a workflow and the scorer get for one Conifer row.

    `row` is kept because `check_constraints` needs `constraints` and
    `reference_answer`, which the workflow itself must never see.
    """

    id: str
    question: str
    rendered_text: str
    row: Dict[str, Any]
    # `optimizer.evaluate_workflow`'s exception path reads `problem.answer`.
    # Conifer has no short gold answer, and leaking `reference_answer` into a
    # record that the optimizer prompt might quote would be a contamination
    # risk, so the attribute exists but stays empty.
    answer: str = ""

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ConiferProblem":
        from conifer_protocol import format_problem_as_prompt

        return cls(
            id=str(row.get("problem_id") or ""),
            question=str(row.get("question") or ""),
            # include_reference defaults to False: the workflow never sees the
            # reference answer, only the scorer does.
            rendered_text=format_problem_as_prompt(row),
            row=row,
        )


def _lexical_f1(checks: Dict[str, Any]) -> float:
    value = checks.get("reference_lexical_f1")
    return float(value) if isinstance(value, (int, float)) else 0.0


# ---------------------------------------------------------------------------
# Scoring seam
# ---------------------------------------------------------------------------


def _failure_samples_conifer(records: List[Dict[str, Any]], k: int = 5) -> str:
    """Failure digest for the optimizer prompt.

    MuSiQue prints `gold=<short entity>`; Conifer's reference answers are long
    paragraphs, so we print which explicit constraints failed instead.  That is
    both shorter and far more actionable for a workflow rewrite.
    """
    failed = [record for record in records if not record.get("correct")]
    if not failed:
        return "(every dev problem passed all explicit checks)"
    lines: List[str] = []
    for record in failed[:k]:
        checks = record.get("hard_checks") or {}
        missed: List[str] = []
        for name, ok in (checks.get("format") or {}).items():
            if not ok:
                missed.append(f"format:{name}")
        for name, ok in (checks.get("limits") or {}).items():
            if not ok:
                missed.append(f"limit:{name}")
        for name, ok in (checks.get("required_terms") or {}).items():
            if not ok:
                missed.append(f"required_term:{name}")
        if not checks.get("answer_present", True):
            missed.append("answer_present")
        line = (
            f"- problem={record.get('problem_id')}"
            f" hard_score={float(record.get('em', 0.0)):.3f}"
            f" coverage={float(checks.get('requirement_coverage', 0.0)):.3f}"
            f" words={checks.get('word_count')}"
            f" failed=[{', '.join(missed) if missed else 'none (only coverage is low)'}]"
        )
        if record.get("error"):
            line += f" error={record['error']}"
        lines.append(line)
    return "\n".join(lines)


def _make_eval_one(optimizer_module, workflow_module):
    from dataclasses import asdict

    run_workflow_on_problem = workflow_module.run_workflow_on_problem

    def _eval_one_conifer(workflow_obj, problem, callers) -> Dict[str, Any]:
        record = run_workflow_on_problem(workflow_obj, problem, callers)
        # No \boxed{} wrapping: Conifer answers are prose and the format checks
        # in check_constraints would be corrupted by a wrapper.
        prediction = record.final_answer or ""
        checks = check_constraints(problem.row, prediction)
        return {
            "problem_id": problem.id,
            "pred": prediction,
            "em": float(checks["hard_score"]),
            "f1": _lexical_f1(checks),
            "correct": bool(checks.get("all_explicit_passed")),
            "metric": METRIC_NAME,
            "hard_checks": checks,
            "n_ops": len(record.op_calls),
            "op_kinds": [call.op for call in record.op_calls],
            "op_records": [asdict(call) for call in record.op_calls],
            "error": record.error,
            "wall_time_s": record.wall_time_s,
        }

    return _eval_one_conifer


# ---------------------------------------------------------------------------
# Sandbox compatibility shims (both carried over from the GSM-Hard adapter)
# ---------------------------------------------------------------------------


def _install_sandbox_shims(workflow_module, operators_module) -> None:
    base_restricted_builtins = workflow_module._restricted_builtins

    def restricted_builtins_with_safe_import():
        builtins = base_restricted_builtins()

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top_level = name.split(".", 1)[0]
            if level or top_level not in workflow_module.ALLOWED_STDLIB_MODULES:
                raise ImportError(f"Workflow import is not allowed: {name!r}")
            return __import__(name, globals, locals, fromlist, level)

        builtins["__import__"] = safe_import
        return builtins

    workflow_module._restricted_builtins = restricted_builtins_with_safe_import

    async def call_with_owned_executor(self, caller, messages):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return await loop.run_in_executor(executor, caller, messages)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    operators_module.Ops._acall = call_with_owned_executor


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def configure() -> Dict[str, object]:
    """Patch the shared AFlow modules for Conifer and return them."""
    import operators
    import optimizer
    import workflow

    _install_sandbox_shims(workflow, operators)

    @classmethod
    def from_conifer(cls: Type, row):
        if isinstance(row, ConiferProblem):
            return cls(id=row.id, question=row.question, rendered_text=row.rendered_text)
        problem = ConiferProblem.from_row(row)
        return cls(
            id=problem.id,
            question=problem.question,
            rendered_text=problem.rendered_text,
        )

    workflow.WorkflowProblem.from_musique = from_conifer

    optimizer._eval_one = _make_eval_one(optimizer, workflow)
    optimizer._failure_samples_for_prompt = _failure_samples_conifer

    operators.PROMPT_SOLVE_PATH = THIS_DIR / "prompts" / "op_solve.md"
    operators.PROMPT_ENSEMBLE_PATH = THIS_DIR / "prompts" / "op_ensemble.md"
    operators.PROMPT_ANSWER_PATH = THIS_DIR / "prompts" / "op_answer.md"
    optimizer.PROMPT_OPTIMIZER_PATH = THIS_DIR / "prompts" / "optimizer_propose.md"

    return {
        "operators": operators,
        "optimizer": optimizer,
        "workflow": workflow,
        "ConiferProblem": ConiferProblem,
        "metric_name": METRIC_NAME,
    }
