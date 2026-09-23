"""Configure the MuSiQue AFlow implementation for GSM-Hard."""

from __future__ import annotations

import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Type


THIS_DIR = Path(__file__).resolve().parent
GSM_BASELINE_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "AFlow"
for path in (PACKAGE_PARENT, GSM_BASELINE_DIR, MUSIQUE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gsmhard_common import (  # noqa: E402
    normalize_numeric_answer,
    patch_musique_numeric_normalizer,
)
from jca.gsm.src.data import format_problem_as_prompt  # noqa: E402
from jca.gsm.src.grader import compute_em_f1, is_correct  # noqa: E402


def configure() -> Dict[str, object]:
    import operators
    import optimizer
    import workflow

    patch_musique_numeric_normalizer()

    base_restricted_builtins = workflow._restricted_builtins

    def restricted_builtins_with_safe_import():
        builtins = base_restricted_builtins()

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top_level = name.split(".", 1)[0]
            if level or top_level not in workflow.ALLOWED_STDLIB_MODULES:
                raise ImportError(f"Workflow import is not allowed: {name!r}")
            return __import__(name, globals, locals, fromlist, level)

        builtins["__import__"] = safe_import
        return builtins

    workflow._restricted_builtins = restricted_builtins_with_safe_import

    async def call_with_owned_executor(self, caller, messages):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return await loop.run_in_executor(executor, caller, messages)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    operators.Ops._acall = call_with_owned_executor

    @classmethod
    def from_gsm(cls: Type, problem):
        return cls(
            id=problem.id,
            question=problem.question,
            rendered_text=format_problem_as_prompt(problem),
        )

    workflow.WorkflowProblem.from_musique = from_gsm
    optimizer.compute_em_f1 = compute_em_f1
    optimizer.is_correct = is_correct

    operators.PROMPT_SOLVE_PATH = THIS_DIR / "prompts" / "op_solve.md"
    operators.PROMPT_ENSEMBLE_PATH = THIS_DIR / "prompts" / "op_ensemble.md"
    operators.PROMPT_ANSWER_PATH = THIS_DIR / "prompts" / "op_answer.md"
    optimizer.PROMPT_OPTIMIZER_PATH = THIS_DIR / "prompts" / "optimizer_propose.md"

    return {
        "operators": operators,
        "optimizer": optimizer,
        "workflow": workflow,
        "normalize_numeric_answer": normalize_numeric_answer,
    }
