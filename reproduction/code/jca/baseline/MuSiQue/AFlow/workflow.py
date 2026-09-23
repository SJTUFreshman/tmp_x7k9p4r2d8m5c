"""Workflow loading, sandboxed execution, and per-problem run.

A workflow is a Python source file (or string) that defines
`async def workflow(problem, ops) -> str`. We load it via a
whitelist-only AST check and exec() into a restricted namespace, then
call it against an `Ops` bundle. The workflow never sees the network,
filesystem, or subprocess.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure this directory is on sys.path so sibling `operators` imports work
# regardless of who imports us (search runner, eval runner, etc.).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from operators import LLMCaller, Ops, OpCallStat


# ---------------------------------------------------------------------------
# AST whitelist
# ---------------------------------------------------------------------------


ALLOWED_STDLIB_MODULES = {
    "asyncio",
    "math",
    "random",
    "json",
    "re",
    "typing",
}


FORBIDDEN_NAMES = {
    # I/O and process
    "open", "exec", "eval", "compile", "input",
    "__import__", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    # Subprocess-like
    "os", "sys", "subprocess", "socket", "urllib", "http",
    "requests", "shutil", "pickle", "marshal",
}


class WorkflowSyntaxError(Exception):
    pass


def _check_ast(source: str) -> None:
    """Whitelist AST check. Raises WorkflowSyntaxError on any violation."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise WorkflowSyntaxError(f"Python syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names]
            module = getattr(node, "module", None)
            targets = names if isinstance(node, ast.Import) else [module or ""]
            for t in targets:
                top = (t or "").split(".", 1)[0]
                if top and top not in ALLOWED_STDLIB_MODULES:
                    raise WorkflowSyntaxError(f"Forbidden import: {t!r}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise WorkflowSyntaxError(f"Forbidden name reference: {node.id!r}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_NAMES:
                raise WorkflowSyntaxError(f"Forbidden attribute base: {node.value.id!r}")


def _find_workflow_function(namespace: Dict[str, Any]) -> Callable[..., Any]:
    fn = namespace.get("workflow")
    if fn is None or not callable(fn):
        raise WorkflowSyntaxError("workflow module must define `async def workflow(problem, ops)`")
    if not asyncio.iscoroutinefunction(fn):
        raise WorkflowSyntaxError("`workflow` must be declared `async def`")
    return fn


# ---------------------------------------------------------------------------
# Workflow container
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    """A loaded workflow: source code + compiled callable + a stable ID."""

    source: str
    fn: Callable[..., Any]
    round_id: int
    name: str
    parent_round: Optional[int] = None
    origin_path: Optional[Path] = None

    @property
    def code_hash(self) -> str:
        return hashlib.md5(self.source.encode("utf-8")).hexdigest()[:10]


def load_workflow_from_source(
    source: str,
    *,
    round_id: int,
    name: str = "",
    parent_round: Optional[int] = None,
    origin_path: Optional[Path] = None,
) -> Workflow:
    _check_ast(source)
    namespace: Dict[str, Any] = {"__builtins__": _restricted_builtins()}
    exec(compile(source, filename=f"<workflow_round_{round_id}>", mode="exec"),
         namespace, namespace)
    fn = _find_workflow_function(namespace)
    return Workflow(
        source=source,
        fn=fn,
        round_id=round_id,
        name=name or f"round_{round_id:02d}",
        parent_round=parent_round,
        origin_path=origin_path,
    )


def load_workflow_from_file(path: Path, round_id: int) -> Workflow:
    source = Path(path).read_text(encoding="utf-8")
    stem_match = re.match(r"round_(\d+)_(.+)", Path(path).stem)
    name = stem_match.group(2) if stem_match else Path(path).stem
    return load_workflow_from_source(
        source, round_id=round_id, name=name, origin_path=Path(path)
    )


def _restricted_builtins() -> Dict[str, Any]:
    """Minimum builtins so workflow code can construct lists/loops/etc."""
    safe = {
        "range", "len", "min", "max", "sum", "abs", "sorted", "enumerate",
        "zip", "map", "filter", "list", "dict", "tuple", "set", "str", "int",
        "float", "bool", "any", "all", "print",
        "True", "False", "None",
        "isinstance", "issubclass",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    }
    import builtins as _b
    restricted = {name: getattr(_b, name) for name in safe if hasattr(_b, name)}
    restricted["__import__"] = _restricted_import
    return restricted


def _restricted_import(
    name: str,
    globals: Optional[Dict[str, Any]] = None,
    locals: Optional[Dict[str, Any]] = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    if level != 0 or name not in ALLOWED_STDLIB_MODULES:
        raise ImportError(f"Workflow import is not allowed: {name!r}")
    import builtins as _b
    return _b.__import__(name, globals, locals, fromlist, level)


# ---------------------------------------------------------------------------
# Problem adapter
# ---------------------------------------------------------------------------


@dataclass
class WorkflowProblem:
    """View passed into workflows. Hides raw MuSiQueProblem internals."""

    id: str
    question: str
    rendered_text: str

    @classmethod
    def from_musique(cls, problem: Any) -> "WorkflowProblem":
        from jca.src.data import format_problem_as_prompt
        return cls(
            id=problem.id,
            question=problem.question,
            rendered_text=format_problem_as_prompt(problem),
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class WorkflowRunRecord:
    problem_id: str
    round_id: int
    final_answer: Optional[str]
    op_calls: List[OpCallStat] = field(default_factory=list)
    error: Optional[str] = None
    wall_time_s: float = 0.0
    max_new_tokens_per_call: Optional[int] = None


@dataclass(frozen=True)
class ExecutorCallers:
    """Model pool used by an AFlow workflow execution."""

    solve_callers: List[LLMCaller]
    judge_caller: LLMCaller
    solve_caller_ids: List[str] = field(default_factory=list)
    judge_caller_id: str = ""

    @classmethod
    def single(cls, caller: LLMCaller, caller_id: str = "executor") -> "ExecutorCallers":
        return cls([caller], caller, [caller_id], caller_id)


async def _run_workflow_async(
    workflow: Workflow, problem: WorkflowProblem, callers: ExecutorCallers | Callable
) -> WorkflowRunRecord:
    if callable(callers):
        callers = ExecutorCallers.single(callers)
    ops = Ops(
        callers.solve_callers,
        callers.judge_caller,
        callers.solve_caller_ids,
        callers.judge_caller_id,
    )
    started = time.monotonic()
    try:
        result = await workflow.fn(problem, ops)
        if not isinstance(result, str):
            result = str(result)
        return WorkflowRunRecord(
            problem_id=problem.id,
            round_id=workflow.round_id,
            final_answer=result,
            op_calls=list(ops.calls),
            wall_time_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        return WorkflowRunRecord(
            problem_id=problem.id,
            round_id=workflow.round_id,
            final_answer=None,
            op_calls=list(ops.calls),
            error=f"{type(exc).__name__}: {exc}",
            wall_time_s=round(time.monotonic() - started, 3),
        )


def run_workflow_on_problem(
    workflow: Workflow, problem: Any, callers: ExecutorCallers | Callable
) -> WorkflowRunRecord:
    """Sync entry point: runs one workflow on one MuSiQue problem."""
    wp = WorkflowProblem.from_musique(problem)
    return asyncio.run(_run_workflow_async(workflow, wp, callers))
