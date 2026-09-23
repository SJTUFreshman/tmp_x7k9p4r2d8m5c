"""AFlow MCTS-lite optimizer.

Simplified single-line-of-history variant of AFlow's search:

    workflows = [initial]
    scores[initial] = eval(initial, dev)
    for iter in [0, N):
        parent = ucb_select(workflows, scores)
        failed = get_failed_examples(parent)
        proposal_source = optimizer_14b.propose(parent.source, failed)
        try:
            new_wf = load_workflow_from_source(proposal_source)
        except WorkflowSyntaxError: skip
        s = evaluate(new_wf, dev)
        workflows.append(new_wf); scores[new_wf] = s
        persist(...)

State is stored on disk (workflows/round_NN_*.py + state.json) so runs
resume cleanly.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

from jca.src.inference import response_attempts

# Ensure the AFlow package directory is on sys.path so `from operators import ...`
# style loads work when the workflow module is exec'd. (Only needed when
# optimizer.py is imported by a runner that hasn't already fixed sys.path.)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from jca.src.grader import compute_em_f1, is_correct
from workflow import (
    ExecutorCallers,
    Workflow,
    WorkflowSyntaxError,
    load_workflow_from_source,
    run_workflow_on_problem,
)


LLMCaller = Callable[[List[Dict[str, str]]], str]


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


@dataclass
class WorkflowNode:
    round_id: int
    name: str
    parent_round: Optional[int]
    source_file: str
    dev_em: float
    dev_f1: float
    dev_records: List[Dict[str, Any]] = field(default_factory=list)
    visits: int = 0  # for UCB-lite parent selection
    proposed_by: str = "manual"  # "manual" for round 0, "optimizer" otherwise
    parse_ok: bool = True
    reject_reason: Optional[str] = None
    optimizer_caller_id: str = ""
    optimizer_messages: List[Dict[str, str]] = field(default_factory=list)
    optimizer_raw_outputs: List[str] = field(default_factory=list)


def _state_path(root: Path) -> Path:
    return root / "state.json"


def load_state(root: Path) -> List[WorkflowNode]:
    p = _state_path(root)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [WorkflowNode(**item) for item in raw]


def save_state(root: Path, nodes: List[WorkflowNode]) -> None:
    p = _state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([asdict(n) for n in nodes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_one(workflow: Workflow, problem: Any, callers: ExecutorCallers) -> Dict[str, Any]:
    rec = run_workflow_on_problem(workflow, problem, callers)
    from jca.src.grader import extract_boxed_answer  # noqa: F401  (kept for parity)
    pred = rec.final_answer or ""
    if pred and "\\boxed" not in pred:
        pred_wrapped = f"\\boxed{{{pred}}}"
    else:
        pred_wrapped = pred
    em, f1 = compute_em_f1(pred_wrapped, problem)
    correct = is_correct(pred_wrapped, problem)
    return {
        "problem_id": problem.id,
        "gold": problem.answer,
        "pred": pred,
        "em": em,
        "f1": f1,
        "correct": correct,
        "error": rec.error,
        "n_ops": len(rec.op_calls),
        "op_kinds": [c.op for c in rec.op_calls],
        "op_records": [asdict(c) for c in rec.op_calls],
        "wall_time_s": rec.wall_time_s,
    }


def evaluate_workflow(
    workflow: Workflow,
    problems: List[Any],
    callers: ExecutorCallers,
    *,
    max_concurrency: int,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Return (mean_em, mean_f1, per-problem records)."""
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_eval_one, workflow, p, callers): p for p in problems
        }
        for fut in as_completed(futures):
            try:
                records.append(fut.result())
            except Exception as exc:  # pragma: no cover
                records.append({
                    "problem_id": futures[fut].id,
                    "gold": futures[fut].answer,
                    "pred": None,
                    "em": 0.0,
                    "f1": 0.0,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_ops": 0,
                    "op_kinds": [],
                    "op_records": [],
                    "wall_time_s": 0.0,
                })
    if not records:
        return 0.0, 0.0, records
    return (
        mean(r["em"] for r in records),
        mean(r["f1"] for r in records),
        records,
    )


# ---------------------------------------------------------------------------
# Parent selection (UCB-lite)
# ---------------------------------------------------------------------------


def _ucb_scores(nodes: List[WorkflowNode], total_visits: int, c: float = 1.4) -> List[float]:
    scores: List[float] = []
    for n in nodes:
        exploit = n.dev_em
        explore = c * math.sqrt(math.log(max(total_visits, 1) + 1) / max(n.visits, 1))
        scores.append(exploit + explore)
    return scores


def select_parent(nodes: List[WorkflowNode], rng: random.Random) -> WorkflowNode:
    total = sum(n.visits for n in nodes)
    if total == 0:
        return nodes[0]
    ucb = _ucb_scores(nodes, total_visits=total)
    max_score = max(ucb)
    winners = [n for n, s in zip(nodes, ucb) if abs(s - max_score) < 1e-9]
    return rng.choice(winners)


# ---------------------------------------------------------------------------
# Optimizer prompt & parsing
# ---------------------------------------------------------------------------


PROMPT_OPTIMIZER_PATH = Path(__file__).parent / "prompts" / "optimizer_propose.md"


def _load_optimizer_prompt() -> str:
    return PROMPT_OPTIMIZER_PATH.read_text(encoding="utf-8").strip()


def _failure_samples_for_prompt(records: List[Dict[str, Any]], k: int = 5) -> str:
    failed = [r for r in records if not r.get("correct")]
    if not failed:
        return "(no failures on dev sample)"
    lines: List[str] = []
    for r in failed[:k]:
        pred = str(r.get("pred") or "").strip().replace("\n", " ")
        if len(pred) > 100:
            pred = pred[:100] + "..."
        gold = str(r.get("gold") or "").strip()
        err = r.get("error")
        line = f"- problem={r['problem_id']} gold={gold!r} pred={pred!r}"
        if err:
            line += f" error={err}"
        lines.append(line)
    return "\n".join(lines)


def _fill_optimizer_prompt(
    parent: WorkflowNode,
    parent_source: str,
    parent_records: List[Dict[str, Any]],
    dev_size: int,
) -> str:
    template = _load_optimizer_prompt()
    return (
        template
        .replace("{CURRENT_WORKFLOW_CODE}", parent_source)
        .replace("{DEV_SIZE}", str(dev_size))
        .replace("{CURRENT_EM}", f"{parent.dev_em:.3f}")
        .replace("{FAILURE_SAMPLES}", _failure_samples_for_prompt(parent_records))
    )


def _extract_python_source(raw_output: str) -> str:
    """Strip Qwen thinking and fences around an optimizer proposal."""
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def propose_workflow_source(
    optimizer_caller: LLMCaller,
    parent: WorkflowNode,
    parent_source: str,
    parent_records: List[Dict[str, Any]],
    dev_size: int,
) -> Tuple[str, List[Dict[str, str]], str]:
    system = "You output only Python source code for a workflow function."
    user = _fill_optimizer_prompt(parent, parent_source, parent_records, dev_size)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = optimizer_caller(messages)
    return _extract_python_source(raw), messages, raw


# ---------------------------------------------------------------------------
# Validation guardrails on proposals
# ---------------------------------------------------------------------------


class ProposalRejected(Exception):
    """Raised when a proposal fails a policy check (before syntax check)."""


def _count_ops_calls(source: str) -> int:
    """Rough count of ops.* calls (used to reject trivial workflows)."""
    return len(re.findall(r"\bops\.(solve|ensemble|answer_generate)\b", source))


def _validate_proposal(source: str) -> None:
    if len(source) > 8000:
        raise ProposalRejected(f"source too large: {len(source)} chars")
    if _count_ops_calls(source) < 2:
        raise ProposalRejected("workflow must call ops at least 2 times")
    if not re.search(r"async\s+def\s+workflow\s*\(", source):
        raise ProposalRejected("missing `async def workflow(...)`")


# ---------------------------------------------------------------------------
# Main MCTS loop
# ---------------------------------------------------------------------------


def _write_workflow_source(root: Path, round_id: int, name: str, source: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:40] or "workflow"
    filename = f"round_{round_id:02d}_{safe_name}.py"
    path = root / "workflows" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return str(path)


def run_mcts(
    *,
    root: Path,
    initial_source_path: Path,
    dev_problems: List[Any],
    executor_callers: ExecutorCallers,
    optimizer_caller: LLMCaller,
    max_iterations: int,
    max_concurrency: int,
    rng_seed: int = 20260810,
    optimizer_caller_id: str = "optimizer",
) -> List[WorkflowNode]:
    """Run MCTS search. Idempotent: resumes from state.json if present."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "workflows").mkdir(parents=True, exist_ok=True)

    nodes = load_state(root)
    rng = random.Random(rng_seed)

    if not nodes:
        # Bootstrap round 0 from the hand-written initial workflow.
        source = initial_source_path.read_text(encoding="utf-8")
        wf = load_workflow_from_source(source, round_id=0, name="initial")
        stored_path = _write_workflow_source(root, 0, "initial", source)
        em, f1, records = evaluate_workflow(
            wf, dev_problems, executor_callers, max_concurrency=max_concurrency,
        )
        node = WorkflowNode(
            round_id=0,
            name="initial",
            parent_round=None,
            source_file=stored_path,
            dev_em=em,
            dev_f1=f1,
            dev_records=records,
            visits=1,
            proposed_by="manual",
            parse_ok=True,
        )
        nodes.append(node)
        save_state(root, nodes)
        print(f"[iter 0/init] EM={em:.3f} F1={f1:.3f} (bootstrap)")

    start_iter = max(n.round_id for n in nodes if n.parse_ok) + 1

    for iter_id in range(start_iter, max_iterations + 1):
        # Only consider parse_ok nodes for parent selection.
        valid_parents = [n for n in nodes if n.parse_ok]
        parent = select_parent(valid_parents, rng)
        parent.visits += 1
        parent_source = Path(parent.source_file).read_text(encoding="utf-8")

        print(f"\n[iter {iter_id}/{max_iterations}] parent=round_{parent.round_id:02d} "
              f"(EM={parent.dev_em:.3f})  proposing…")
        t0 = time.monotonic()
        proposal_source = ""
        optimizer_messages: List[Dict[str, str]] = []
        optimizer_raw_outputs: List[str] = []
        try:
            proposal_source, optimizer_messages, optimizer_raw = propose_workflow_source(
                optimizer_caller, parent, parent_source, parent.dev_records,
                dev_size=len(dev_problems),
            )
            optimizer_raw_outputs.extend(response_attempts(optimizer_raw))
            _validate_proposal(proposal_source)
            wf = load_workflow_from_source(
                proposal_source,
                round_id=iter_id,
                name="proposed",
                parent_round=parent.round_id,
            )
        except (ProposalRejected, WorkflowSyntaxError, SyntaxError) as exc:
            reject_reason = f"{type(exc).__name__}: {exc}"
            print(f"  proposal REJECTED at parse/policy: {reject_reason}")
            # Persist as a parse-failed node so we don't retry the same parent
            # loop indefinitely (visits still incremented above).
            try:
                stored_path = _write_workflow_source(
                    root, iter_id, "rejected", proposal_source
                )
            except Exception:
                stored_path = ""
            nodes.append(WorkflowNode(
                round_id=iter_id,
                name="rejected",
                parent_round=parent.round_id,
                source_file=stored_path,
                dev_em=0.0,
                dev_f1=0.0,
                dev_records=[],
                visits=0,
                proposed_by="optimizer",
                parse_ok=False,
                reject_reason=reject_reason,
                optimizer_caller_id=optimizer_caller_id,
                optimizer_messages=optimizer_messages,
                optimizer_raw_outputs=optimizer_raw_outputs,
            ))
            save_state(root, nodes)
            continue

        stored_path = _write_workflow_source(
            root, iter_id, "proposed", proposal_source
        )
        em, f1, records = evaluate_workflow(
            wf, dev_problems, executor_callers, max_concurrency=max_concurrency,
        )
        elapsed = time.monotonic() - t0
        delta = em - parent.dev_em
        print(f"  new EM={em:.3f} F1={f1:.3f}  (Δ={delta:+.3f} vs parent) "
              f"elapsed={elapsed:.1f}s")

        nodes.append(WorkflowNode(
            round_id=iter_id,
            name="proposed",
            parent_round=parent.round_id,
            source_file=stored_path,
            dev_em=em,
            dev_f1=f1,
            dev_records=records,
            visits=1,
            proposed_by="optimizer",
            parse_ok=True,
            optimizer_caller_id=optimizer_caller_id,
            optimizer_messages=optimizer_messages,
            optimizer_raw_outputs=optimizer_raw_outputs,
        ))
        save_state(root, nodes)

    return nodes


def best_node(nodes: List[WorkflowNode]) -> Optional[WorkflowNode]:
    valid = [n for n in nodes if n.parse_ok]
    if not valid:
        return None
    return max(valid, key=lambda n: (n.dev_em, n.dev_f1))
