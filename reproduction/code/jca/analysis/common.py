"""Common data loading for baseline output analysis.

WHAT THIS FILE DOES
===================
Every baseline in `jca/baseline/MuSiQue/<X>/outputs/*.jsonl` uses a
slightly different per-record schema — MAD stores agent turns, AFlow
stores op_calls, GPTSwarm stores node_outputs, etc. This module
normalizes all of them into a single dataclass so downstream analysis
code doesn't have to care which baseline the record came from.

WHAT IT EXPOSES
===============
- `ProblemRecord` — one problem's worth of normalized data:
    * problem_id, question, gold_answer, hop
    * em, f1, correct, final_answer (already-graded outcome)
    * generations: List[Generation] — every LLM output emitted while
      answering this problem (both intermediate reasoning and the
      final answer producer)
    * output_texts: List[str] — complete raw response from every LLM call;
      token-cost metrics use this instead of parsed generations
    * n_agents_reasoning_at_final_layer: List[str] — only populated
      for methods with a "final round of parallel agents"; used by
      redundancy metric (see metric_redundancy.py)

- `Generation` — one LLM output blob:
    * kind: "reasoning" | "answer" | "raw"
    * text: str
    * role_tag: str — free-form label ("A1", "cot_0", "iter1/A2", etc.)
      only used for grouping in the redundancy metric

- `load_records(baseline: str, jsonl_path: Path) -> List[ProblemRecord]`
  — dispatches to per-baseline parsers.

- `count_tokens(text, tokenizer_name=DEFAULT_TOKENIZER)` — Qwen3
  tokenizer via HuggingFace. Cached at module level. Falls back to a
  4-char-per-token heuristic if transformers isn't available (so the
  analysis code stays runnable on laptops without GPU deps).

WHY WE DO IT THIS WAY
=====================
The 5 baselines were written by different iterations of pair-programming
and their jsonl schemas diverged. Rather than force retroactive schema
migration, we absorb the differences here. Adding a new baseline means
writing one new `_load_<baseline>` function.

BASELINES SUPPORTED
===================
- "single"      — Single-agent CoT (whatever schema you used; see fallback)
- "mad"         — jca/baseline/MuSiQue/MAD
- "agentverse"  — jca/baseline/MuSiQue/AgentVerse
- "aflow"       — jca/baseline/MuSiQue/AFlow
- "gptswarm"    — jca/baseline/MuSiQue/GPTSwarm
- "jca"         — the trained JCA runs (SFT+RL evaluation jsonl)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


_LOCAL_QWEN3_TOKENIZER = Path("/data/wangyuheng/models/Qwen3-1.7B")
DEFAULT_TOKENIZER = os.environ.get(
    "JCA_ANALYSIS_TOKENIZER",
    str(_LOCAL_QWEN3_TOKENIZER)
    if _LOCAL_QWEN3_TOKENIZER.joinpath("tokenizer.json").exists()
    else "Qwen/Qwen2.5-1.5B",
)


# ============================================================================
# Public types
# ============================================================================


@dataclass
class Generation:
    """One LLM output blob emitted somewhere during a problem."""

    kind: str            # "reasoning" | "answer" | "raw"
    text: str
    role_tag: str = ""   # e.g. "A1", "cot_0", "iter1/A2" — only for grouping


@dataclass
class ProblemRecord:
    """Normalized view of one problem's baseline output."""

    baseline: str
    problem_id: str
    question: str
    gold_answer: str
    hop: Optional[int]

    correct: bool
    em: float
    f1: float
    final_answer: Optional[str]
    supporting_idx: List[int] = field(default_factory=list)

    generations: List[Generation] = field(default_factory=list)

    # Complete text returned by each LLM call. Cost metrics must use this
    # instead of `generations`, whose parsed reasoning/answer fields serve
    # semantic analyses and may omit JSON/action text or duplicate a vote.
    output_texts: List[str] = field(default_factory=list)
    output_texts_available: bool = False

    # For redundancy metric (only populated for multi-agent baselines):
    # list-of-lists where each inner list is the reasonings of the
    # agents that produced the "final round" of answers. For MAD this
    # is the round-(N-1) turns; for AgentVerse the last iteration's
    # 3 agent answers; for GPTSwarm the layer-1 IO/CoT nodes; for
    # JCA the last few turns before stop.
    final_round_reasonings: List[str] = field(default_factory=list)
    redundancy_cohort: Optional[str] = None

    wall_time_s: Optional[float] = None


# ============================================================================
# Tokenizer helper
# ============================================================================


@lru_cache(maxsize=4)
def _get_tokenizer(name: str):
    """Load a HF tokenizer once. Returns None if transformers missing."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    """Return token count of `text`. Falls back to len(text)//4."""
    if not text:
        return 0
    tok = _get_tokenizer(tokenizer_name)
    if tok is None:
        return max(1, len(text) // 4)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def count_output_tokens(
    rec: ProblemRecord,
    *,
    allow_final_answer_proxy: bool = False,
) -> int:
    """Count complete outputs, optionally falling back to final-answer proxy."""
    if not rec.output_texts_available:
        if allow_final_answer_proxy:
            return count_tokens(_strip_boxed(rec.final_answer))
        raise ValueError(
            f"{rec.baseline} record {rec.problem_id!r} does not contain complete "
            "per-call outputs; rerun that baseline with output logging enabled"
        )
    return sum(count_tokens(text) for text in rec.output_texts)


# ============================================================================
# Public entry point
# ============================================================================


def load_records(baseline: str, jsonl_path: Path) -> List[ProblemRecord]:
    """Load records for a given baseline. Dispatches to the right parser."""
    baseline = baseline.lower()
    loaders: Dict[str, Callable[[Path], List[ProblemRecord]]] = {
        "single": _load_single,
        "mad": _load_mad,
        "agentverse": _load_agentverse,
        "aflow": _load_aflow,
        "gptswarm": _load_gptswarm,
        "jca": _load_jca,
    }
    if baseline not in loaders:
        raise ValueError(
            f"Unknown baseline {baseline!r}. Known: {sorted(loaders)}"
        )
    return loaders[baseline](Path(jsonl_path))


# ============================================================================
# Shared jsonl reader + problem-level fields
# ============================================================================


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: bad JSON: {exc}") from exc


def _problem_fields(raw: dict) -> Dict[str, Any]:
    """Extract shared problem-level fields present in every baseline's jsonl."""
    p = raw.get("problem", {}) or {}
    return {
        "problem_id": p.get("id", ""),
        "question": p.get("question", ""),
        "gold_answer": p.get("answer", ""),
        "hop": p.get("hop"),
        "correct": bool(raw.get("correct", False)),
        "em": float(raw.get("em", 0.0)),
        "f1": float(raw.get("f1", 0.0)),
        "final_answer": raw.get("final_answer"),
        "supporting_idx": [
            int(idx) for idx in (p.get("supporting_idx") or [])
            if isinstance(idx, int)
        ],
        "wall_time_s": raw.get("wall_time_s"),
    }


def _strip_boxed(text: Optional[str]) -> str:
    """Best-effort strip of \\boxed{...} wrapping. Returns '' if text is None."""
    if not text:
        return ""
    s = str(text).strip()
    marker = r"\boxed{"
    if marker in s:
        i = s.find(marker) + len(marker)
        depth = 1
        j = i
        while j < len(s) and depth > 0:
            c = s[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j].strip()
            j += 1
    return s


# ============================================================================
# Per-baseline loaders
# ============================================================================


def _load_single(path: Path) -> List[ProblemRecord]:
    """Single-agent CoT: one LLM call per problem.

    Expected schema (permissive):
        record["reasoning"] or record["cot"]  -> reasoning text
        record["final_answer"]                -> final answer string
    Falls back gracefully if fields differ.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        reasoning = (raw.get("reasoning") or raw.get("cot")
                     or raw.get("model_output") or "")
        model_output = raw.get("raw_output") or raw.get("model_output")
        final_ans = _strip_boxed(fields["final_answer"])
        gens = []
        if reasoning:
            gens.append(Generation(kind="reasoning", text=reasoning, role_tag="A3"))
        if final_ans:
            gens.append(Generation(kind="answer", text=final_ans, role_tag="A3"))
        out.append(ProblemRecord(
            baseline="single",
            generations=gens,
            output_texts=[str(model_output)] if model_output else [],
            output_texts_available=model_output is not None,
            final_round_reasonings=[reasoning] if reasoning else [],
            **fields,
        ))
    return out


def _load_mad(path: Path) -> List[ProblemRecord]:
    """MAD: rounds of parallel agent turns.

    Each record has record["mad"]["turns"] — a flat list of AgentTurn
    dicts with (agent_id, round_idx, reasoning, answer). The redundancy
    metric wants the FINAL round's reasonings across A1/A2/A3.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        mad = raw.get("mad", {}) or {}
        turns = mad.get("turns", []) or []
        gens: List[Generation] = []
        output_texts: List[str] = []
        max_round = -1
        for t in turns:
            r_idx = int(t.get("round_idx", 0))
            max_round = max(max_round, r_idx)
            aid = t.get("agent_id", "?")
            reasoning = t.get("reasoning") or ""
            answer = t.get("answer") or ""
            tag = f"round{r_idx}/{aid}"
            raw_outputs = t.get("raw_outputs")
            if not isinstance(raw_outputs, list):
                raw_outputs = [t.get("raw_output")] if t.get("raw_output") else []
            output_texts.extend(str(text) for text in raw_outputs if text)
            if reasoning:
                gens.append(Generation(kind="reasoning", text=reasoning, role_tag=tag))
            if answer:
                gens.append(Generation(kind="answer", text=answer, role_tag=tag))
        # Final round reasonings (one per agent)
        final_round_reasonings = [
            (t.get("reasoning") or "")
            for t in turns
            if int(t.get("round_idx", -1)) == max_round and t.get("reasoning")
        ]
        final_ans = _strip_boxed(fields["final_answer"])
        if final_ans:
            gens.append(Generation(kind="answer", text=final_ans, role_tag="team"))
        out.append(ProblemRecord(
            baseline="mad",
            generations=gens,
            output_texts=output_texts,
            output_texts_available=all(
                isinstance(t.get("raw_outputs"), list)
                or t.get("raw_output") is not None
                for t in turns
            ),
            final_round_reasonings=final_round_reasonings,
            redundancy_cohort="parallel_final_round",
            **fields,
        ))
    return out


def _load_agentverse(path: Path) -> List[ProblemRecord]:
    """AgentVerse: iterations of {3 agents + 1 evaluator}.

    Each record has record["agentverse"]["iterations"] — list of iter
    dicts. Each iter's "answers" is a list of 3 per-agent dicts. The
    redundancy metric wants the LAST iteration's 3 agent reasonings.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        av = raw.get("agentverse", {}) or {}
        iters = av.get("iterations", []) or []
        gens: List[Generation] = []
        output_texts: List[str] = []
        output_texts_available = (
            isinstance(av.get("recruit_raw_outputs"), list)
            or av.get("recruit_raw_output") is not None
        )
        recruit_outputs = av.get("recruit_raw_outputs")
        if not isinstance(recruit_outputs, list):
            recruit_outputs = ([av.get("recruit_raw_output")]
                               if av.get("recruit_raw_output") else [])
        output_texts.extend(str(text) for text in recruit_outputs if text)
        for i, it in enumerate(iters):
            for a in it.get("answers", []) or []:
                aid = a.get("agent_id", "?")
                reasoning = a.get("reasoning") or ""
                answer = a.get("answer") or ""
                tag = f"iter{i}/{aid}"
                raw_outputs = a.get("raw_outputs")
                if not isinstance(raw_outputs, list):
                    raw_outputs = [a.get("raw_output")] if a.get("raw_output") else []
                output_texts.extend(str(text) for text in raw_outputs if text)
                output_texts_available = (
                    output_texts_available
                    and (
                        isinstance(a.get("raw_outputs"), list)
                        or a.get("raw_output") is not None
                    )
                )
                if reasoning:
                    gens.append(Generation(kind="reasoning", text=reasoning, role_tag=tag))
                if answer:
                    gens.append(Generation(kind="answer", text=answer, role_tag=tag))
            ev = it.get("evaluation") or {}
            ev_outputs = ev.get("raw_outputs")
            if not isinstance(ev_outputs, list):
                ev_outputs = [ev.get("raw_output")] if ev.get("raw_output") else []
            output_texts.extend(str(text) for text in ev_outputs if text)
            output_texts_available = (
                output_texts_available
                and (
                    isinstance(ev.get("raw_outputs"), list)
                    or ev.get("raw_output") is not None
                )
            )
            fb = ev.get("feedback") or ""
            if fb:
                gens.append(Generation(kind="reasoning", text=fb,
                                        role_tag=f"iter{i}/evaluator"))
        final_round_reasonings: List[str] = []
        if iters:
            last = iters[-1]
            final_round_reasonings = [
                (a.get("reasoning") or "")
                for a in (last.get("answers") or []) if a.get("reasoning")
            ]
        final_ans = _strip_boxed(fields["final_answer"])
        if final_ans:
            gens.append(Generation(kind="answer", text=final_ans, role_tag="team"))
        out.append(ProblemRecord(
            baseline="agentverse",
            generations=gens,
            output_texts=output_texts,
            output_texts_available=output_texts_available,
            final_round_reasonings=final_round_reasonings,
            redundancy_cohort="parallel_final_round",
            **fields,
        ))
    return out


def _load_aflow(path: Path) -> List[ProblemRecord]:
    """AFlow eval jsonl with complete per-op raw outputs.

    Legacy files containing only n_ops/op_kinds remain usable for metrics
    unrelated to cost, but token metrics reject them as incomplete.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        gens: List[Generation] = []
        output_texts: List[str] = []
        output_texts_available = "op_records" in raw
        final_ans = _strip_boxed(fields["final_answer"])
        if final_ans:
            gens.append(Generation(kind="answer", text=final_ans, role_tag="workflow"))
        for op in raw.get("op_records", []) or []:
            raw_outputs = op.get("raw_outputs")
            if not isinstance(raw_outputs, list):
                raw_outputs = [op.get("raw_output")] if op.get("raw_output") else []
            output_texts.extend(str(text) for text in raw_outputs if text)
            output_texts_available = output_texts_available and (
                isinstance(op.get("raw_outputs"), list)
                or op.get("raw_output") is not None
            )
            for output_idx, text in enumerate(raw_outputs):
                reasoning = _extract_json_string_field(str(text), "reasoning")
                if reasoning:
                    role = op.get("caller_id") or op.get("op") or "op"
                    gens.append(Generation(
                        kind="reasoning",
                        text=reasoning,
                        role_tag=f"{role}/attempt{output_idx + 1}",
                    ))
        out.append(ProblemRecord(
            baseline="aflow",
            generations=gens,
            output_texts=output_texts,
            output_texts_available=output_texts_available,
            final_round_reasonings=[],  # not available at this granularity
            redundancy_cohort=None,
            **fields,
        ))
    return out


def _extract_json_string_field(text: str, field_name: str) -> str:
    """Extract a string field from JSON, including a safely recoverable partial object."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        value = payload.get(field_name)
        return value.strip() if isinstance(value, str) else ""

    # Some model responses contain a complete JSON string value but omit a
    # trailing object delimiter. Decode only that complete quoted value.
    match = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*("(?:\\.|[^"\\])*")',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    return value.strip() if isinstance(value, str) else ""


def _load_gptswarm(path: Path) -> List[ProblemRecord]:
    """GPTSwarm: fixed 7-node DAG.

    Each record has record["swarm"]["node_outputs"] — 7 NodeOutput
    dicts. The redundancy metric wants the layer-1 (IO/CoT parallel)
    reasonings, i.e. nodes with no predecessors.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        swarm = raw.get("swarm", {}) or {}
        nodes = swarm.get("node_outputs", []) or []
        layers = swarm.get("layers", []) or []
        gens: List[Generation] = []
        output_texts: List[str] = []
        for n in nodes:
            reasoning = n.get("reasoning") or ""
            answer = n.get("answer") or ""
            tag = f"{n.get('node_type', '?')}/{n.get('node_name', '?')}"
            raw_outputs = n.get("raw_outputs")
            if not isinstance(raw_outputs, list):
                raw_outputs = [n.get("raw_output")] if n.get("raw_output") else []
            output_texts.extend(str(text) for text in raw_outputs if text)
            if reasoning:
                gens.append(Generation(kind="reasoning", text=reasoning, role_tag=tag))
            if answer:
                gens.append(Generation(kind="answer", text=answer, role_tag=tag))
        # Layer 0 = source layer = parallel agents that don't see peers
        source_layer = set(layers[0]) if layers else set()
        final_round_reasonings = [
            (n.get("reasoning") or "")
            for n in nodes
            if n.get("node_name") in source_layer and n.get("reasoning")
        ]
        out.append(ProblemRecord(
            baseline="gptswarm",
            generations=gens,
            output_texts=output_texts,
            output_texts_available=all(
                isinstance(n.get("raw_outputs"), list)
                or n.get("raw_output") is not None
                for n in nodes
            ),
            final_round_reasonings=final_round_reasonings,
            redundancy_cohort="parallel_source_layer",
            **fields,
        ))
    return out


def _load_jca(path: Path) -> List[ProblemRecord]:
    """JCA trained-eval jsonl (v1 old_sft protocol schema).

    Assumes the same top-level shape as run_sft_old_protocol.py:
        record["trajectory"]["steps"] — list of OldTrajectoryStep dicts
        with (turn, active_agent, reasoning, tentative_answer, ...).
    Adjust here if the current protocol changed.
    """
    out: List[ProblemRecord] = []
    for raw in _iter_jsonl(path):
        fields = _problem_fields(raw)
        traj = raw.get("trajectory", {}) or {}
        steps = traj.get("steps", []) or []
        gens: List[Generation] = []
        output_texts: List[str] = []
        for s in steps:
            reasoning = s.get("reasoning") or ""
            tent = s.get("tentative_answer") or ""
            aid = s.get("active_agent", "?")
            turn = s.get("turn", 0)
            tag = f"turn{turn}/{aid}"
            raw_outputs = s.get("raw_outputs")
            if not isinstance(raw_outputs, list):
                raw_outputs = [s.get("raw_output")] if s.get("raw_output") else []
            output_texts.extend(str(text) for text in raw_outputs if text)
            if reasoning:
                gens.append(Generation(kind="reasoning", text=reasoning, role_tag=tag))
            if tent:
                gens.append(Generation(kind="answer", text=tent, role_tag=tag))
        final_ans = _strip_boxed(fields["final_answer"])
        if final_ans:
            gens.append(Generation(kind="answer", text=final_ans, role_tag="team"))
        # Final round = last 3 distinct agents' reasonings (best proxy)
        by_agent: Dict[str, str] = {}
        for s in steps:
            aid = s.get("active_agent")
            r = s.get("reasoning") or ""
            if aid and r:
                by_agent[aid] = r  # later turns overwrite earlier ones
        final_round_reasonings = list(by_agent.values())
        out.append(ProblemRecord(
            baseline="jca",
            generations=gens,
            output_texts=output_texts,
            output_texts_available=all(
                isinstance(s.get("raw_outputs"), list)
                or s.get("raw_output") is not None
                for s in steps
            ),
            final_round_reasonings=final_round_reasonings,
            redundancy_cohort="sequential_last_per_agent",
            **fields,
        ))
    return out
