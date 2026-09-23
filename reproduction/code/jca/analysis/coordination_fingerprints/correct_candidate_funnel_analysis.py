"""Measure correct-candidate discovery and retention across supported benchmarks.

The protocol-specific candidate extraction deliberately reuses
``analysis.gsm_hard.failure_modes.intermediate_answers`` so that GSM-Hard
matches section 7.2 of the existing failure-mode report exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.gsm_hard.failure_modes import (  # noqa: E402
    extract_number,
    intermediate_answers,
    numeric_match,
)
from src.grader import is_correct as musique_is_correct  # noqa: E402
from src.math_eval import compute_math_em  # noqa: E402
from analysis.math.common import (  # noqa: E402
    MATH_JCA_TURN_BUDGET,
    math_jca_answer_projection,
)

try:
    from .benchmark_paths import (
        MATH_AFLOW,
        MATH_AGENTVERSE,
        MATH_GPTSWARM,
        MATH_JCA,
        MATH_MAD,
        MATH_ZERO_SHOT,
        MULTIPL_E_AFLOW,
        MULTIPL_E_AGENTVERSE,
        MULTIPL_E_GPTSWARM,
        MULTIPL_E_JCA,
        MULTIPL_E_MAD,
        MULTIPL_E_SELF_RL,
        MULTIPL_E_ZERO_SHOT,
        MULTIPL_E_ZERO_SHOT_COMPLETION_ROOT,
        MUSIQUE_AFLOW,
        MUSIQUE_AGENTVERSE,
        MUSIQUE_GPTSWARM,
        MUSIQUE_JCA,
        MUSIQUE_MAD,
        GSM_JCA,
        CONIFER_JCA,
        CONIFER_MAD,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_AFLOW,
    )
except ImportError:
    from benchmark_paths import (
        MATH_AFLOW,
        MATH_AGENTVERSE,
        MATH_GPTSWARM,
        MATH_JCA,
        MATH_MAD,
        MATH_ZERO_SHOT,
        MULTIPL_E_AFLOW,
        MULTIPL_E_AGENTVERSE,
        MULTIPL_E_GPTSWARM,
        MULTIPL_E_JCA,
        MULTIPL_E_MAD,
        MULTIPL_E_SELF_RL,
        MULTIPL_E_ZERO_SHOT,
        MULTIPL_E_ZERO_SHOT_COMPLETION_ROOT,
        MUSIQUE_AFLOW,
        MUSIQUE_AGENTVERSE,
        MUSIQUE_GPTSWARM,
        MUSIQUE_JCA,
        MUSIQUE_MAD,
        GSM_JCA,
        CONIFER_JCA,
        CONIFER_MAD,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_AFLOW,
    )


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUTPUT_DIR / "correct_candidate_funnel_report.md"

DEFAULT_MUSIQUE_JCA = MUSIQUE_JCA
# 2026-09-14：MuSiQue 四个 baseline 的路径改为直接取自 benchmark_paths，不再在本文件里
# 重复硬编码一份。此前这里写死的是 2026-09-11 重跑之前的旧 run，导致同一个方法在本报告
# 与 model_call_count_report / weighted_output_token_14b_report 里用的不是同一次运行。
# 2026-09-14 前原值逐字如下（仓库没有可用 git，保留以便比对）：
#   DEFAULT_MUSIQUE_MAD = (
#       ROOT / "baseline/MuSiQue/MAD/logs/20260811_104018_mad_dev_start0_n2417_r3/results.jsonl"
#   )                                                              # 943/2417 = 39.02%
#   DEFAULT_MUSIQUE_AGENTVERSE = (
#       ROOT / "baseline/MuSiQue/AgentVerse/logs/agentverse_meta8b_dev_n2417/results.jsonl"
#   )                                                              # 1014/2417 = 41.95%
#   DEFAULT_MUSIQUE_AFLOW = (
#       ROOT / "baseline/MuSiQue/AFlow/outputs/aflow_hetero_search20_dev20_full2417.jsonl"
#   )                                                              # 1019/2417 = 42.16%
#   DEFAULT_MUSIQUE_GPTSWARM = (
#       ROOT / "baseline/MuSiQue/GPTSwarm/logs/musique_gptswarm_hetero_full_v1/results.jsonl"
#   )                                                              # 885/2417 = 36.62%
DEFAULT_MUSIQUE_MAD = MUSIQUE_MAD
DEFAULT_MUSIQUE_AGENTVERSE = MUSIQUE_AGENTVERSE
DEFAULT_MUSIQUE_AFLOW = MUSIQUE_AFLOW
DEFAULT_MUSIQUE_GPTSWARM = MUSIQUE_GPTSWARM

DEFAULT_GSM_ROOT = ROOT / "logs/gsm_hard_baselines/gsmhard_latest_roles_raw8192_20260816"
DEFAULT_GSM_JCA = GSM_JCA
DEFAULT_GSM_MAD = DEFAULT_GSM_ROOT / "01_mad/results.jsonl"
DEFAULT_GSM_AGENTVERSE = DEFAULT_GSM_ROOT / "02_agentverse/results.jsonl"
DEFAULT_GSM_AFLOW = DEFAULT_GSM_ROOT / "03_aflow/eval_results.jsonl"
DEFAULT_GSM_GPTSWARM = DEFAULT_GSM_ROOT / "04_gptswarm/results.jsonl"
DEFAULT_MATH_JCA = MATH_JCA

METHOD_ORDER = ("JCA", "MAD", "AgentVerse", "AFlow", "GPTSwarm")
METHOD_KEYS = {
    "JCA": "jca",
    "MAD": "mad",
    "AgentVerse": "agentverse",
    "AFlow": "aflow",
    "GPTSwarm": "gptswarm",
}
MULTIPL_REPLAY_ROOT = ROOT / "analysis/multipl-e-8lang/results/candidate_replay"
MULTIPL_ZERO_SHOT_REPLAY_ROOT = ROOT / "analysis/multipl-e-8lang/results/zero_shot_candidate_replay"
MULTIPL_JCA_REPLAY_ROOT = MULTIPL_REPLAY_ROOT / "jca_mpe0912_120_r08"
CANDIDATE_SOURCES = {
    "JCA": "`trajectory.steps[].tentative_answer`",
    "MAD": "`mad.turns[].answer`",
    "AgentVerse": "`agentverse.iterations[].answers[].answer`",
    "AFlow": "成功的 `op_records[op=solve]` 输出中的 `answer`",
    "GPTSwarm": "`swarm.node_outputs[].answer`，排除最终 `Aggregator`",
}
MULTIPL_METHOD_ORDER = ("zero-shot", "JCA", "MAD", "AgentVerse", "AFlow", "GPTSwarm", "self-RL")
MULTIPL_CANDIDATE_SOURCES = {
    "zero-shot": "MultiPL-E candidate replay 中的 `tentative_completion`",
    "JCA": "MultiPL-E candidate replay 中的 `tentative_completion`",
    "MAD": "MultiPL-E candidate replay 中的 MAD 中间 completion",
    "AgentVerse": "MultiPL-E candidate replay 中的 AgentVerse solver completion",
    "AFlow": "MultiPL-E candidate replay 中成功 solve op 的 completion",
    "GPTSwarm": "MultiPL-E candidate replay 中排除 Aggregator 的 node completion",
    "self-RL": "n/a（SAS 是 one-shot，没有中间候选）",
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    method: str
    path: Path
    replay_path: Path | None = None
    applicable: bool = True
    note: str = ""

    @property
    def method_key(self) -> str:
        return METHOD_KEYS[self.method]


@dataclass(frozen=True)
class FunnelResult:
    dataset: str
    method: str
    path: Path
    n_problems: int
    n_final_correct: int
    with_parsed_candidate: int
    with_correct_candidate: int
    correct_candidate_retained: int
    correct_candidate_lost: int
    rescued_without_correct_candidate: int
    no_correct_candidate_final_wrong: int
    final_grader_disagreements: int
    applicable: bool = True
    note: str = ""
    full_trajectory_with_parsed_candidate: int | None = None
    full_trajectory_with_correct_candidate: int | None = None
    full_trajectory_correct_candidate_retained: int | None = None

    @property
    def final_accuracy(self) -> float:
        return self.n_final_correct / self.n_problems if self.n_problems else 0.0

    @property
    def discovery_rate(self) -> float:
        return self.with_correct_candidate / self.n_problems if self.n_problems else 0.0

    @property
    def retention_rate(self) -> float | None:
        if not self.with_correct_candidate:
            return None
        return self.correct_candidate_retained / self.with_correct_candidate

    @property
    def parsed_candidate_rate(self) -> float:
        return self.with_parsed_candidate / self.n_problems if self.n_problems else 0.0


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def _musique_problem(record: dict[str, Any]) -> SimpleNamespace:
    problem = record.get("problem") or {}
    return SimpleNamespace(
        answer=str(problem.get("answer") or ""),
        answer_aliases=[
            str(alias) for alias in (problem.get("answer_aliases") or [])
        ],
    )


def _candidate_value_is_parsed(dataset: str, candidate: object) -> bool:
    if dataset == "MuSiQue":
        return bool(str(candidate or "").strip())
    return extract_number(candidate) is not None


def _answer_is_correct(
    dataset: str,
    candidate: object,
    record: dict[str, Any],
) -> bool:
    if dataset == "MuSiQue":
        text = str(candidate or "").strip()
        return bool(text and musique_is_correct(text, _musique_problem(record)))
    gold = extract_number((record.get("problem") or {}).get("answer"))
    prediction = extract_number(candidate)
    return numeric_match(prediction, gold)


def _persisted_final_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record.get("correct"))
    return float(record.get("em", 0.0) or 0.0) >= 1.0


def _math_final_answer(method: str, record: dict[str, Any]) -> str:
    if method == "JCA":
        return str(math_jca_answer_projection(record)["projected_answer"] or "").strip()
    if method == "AFlow":
        return str(record.get("prediction") or "").strip()
    return str(record.get("final_answer") or "").strip()


def _math_final_correct(method: str, record: dict[str, Any]) -> bool:
    if method == "JCA":
        return _math_answer_is_correct(_math_final_answer(method, record), record)
    return _persisted_final_correct(record)


def _conifer_candidates(method: str, record: dict[str, Any]) -> list[tuple[str, bool]]:
    """Extract Conifer intermediate candidates with their all_explicit_passed status.

    Returns list of (tentative_answer, all_explicit_passed) tuples.
    """
    steps = (record.get("trajectory") or {}).get("steps", []) or []
    candidates = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tentative = step.get("tentative_answer")
        if tentative in (None, ""):
            continue
        hard_checks = step.get("hard_checks") or {}
        all_explicit = hard_checks.get("all_explicit_passed", False)
        candidates.append((str(tentative).strip(), all_explicit))
    return candidates


def _conifer_final_correct(record: dict[str, Any]) -> bool:
    """Check if Conifer final answer passes all explicit checks."""
    final_checks = record.get("final_checks") or {}
    return bool(final_checks.get("all_explicit_passed", False))


def _as_list(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _math_candidates(method: str, record: dict[str, Any]) -> list[str]:
    if method in {"zero-shot", "JCA"}:
        steps = (record.get("trajectory") or {}).get("steps", []) or []
        if method == "JCA":
            steps = steps[:MATH_JCA_TURN_BUDGET]
        return [
            str(step["tentative_answer"]).strip()
            for step in steps
            if isinstance(step, dict) and step.get("tentative_answer") not in (None, "")
        ]
    if method == "MAD":
        return [str(turn["answer"]).strip() for turn in _as_list(record.get("turns")) if turn.get("answer") not in (None, "")]
    if method == "AgentVerse":
        iterations = record.get("iterations") or {}
        return [
            str(answer["answer"]).strip()
            for iteration in _as_list(iterations)
            for answer in _as_list(iteration.get("answers"))
            if answer.get("answer") not in (None, "")
        ]
    if method == "GPTSwarm":
        outputs = record.get("node_outputs") or {}
        return [
            str(node["answer"]).strip()
            for node in _as_list(outputs)
            if str(node.get("node_type") or "").lower() != "aggregator"
            and node.get("answer") not in (None, "")
        ]
    if method == "AFlow":
        try:
            from analysis.math.common import component_answer
        except ImportError:
            from analysis.math.common import component_answer
        return [
            answer
            for operation in _as_list(record.get("op_records"))
            if str(operation.get("op") or "").lower() == "solve"
            and operation.get("success", operation.get("ok", False))
            and (answer := component_answer(operation))
        ]
    return []


def _math_answer_is_correct(candidate: object, record: dict[str, Any]) -> bool:
    problem = record.get("problem") or {}
    gold_answer = record.get("gold_answer") or problem.get("gold_answer")
    return bool(
        compute_math_em(
            str(candidate or ""),
            str(gold_answer or ""),
        )
    )


def _analyze_math_run(spec: RunSpec) -> FunnelResult:
    counts = {
        "n_problems": 0,
        "n_final_correct": 0,
        "with_parsed_candidate": 0,
        "with_correct_candidate": 0,
        "correct_candidate_retained": 0,
        "correct_candidate_lost": 0,
        "rescued_without_correct_candidate": 0,
        "no_correct_candidate_final_wrong": 0,
        "final_grader_disagreements": 0,
    }
    full_trajectory_with_parsed_candidate = 0
    full_trajectory_with_correct_candidate = 0
    full_trajectory_correct_candidate_retained = 0
    for record in iter_jsonl(spec.path):
        counts["n_problems"] += 1
        candidates = _math_candidates(spec.method, record)
        has_correct = any(_math_answer_is_correct(candidate, record) for candidate in candidates)
        final_correct = _math_final_correct(spec.method, record)
        final_value = _math_final_answer(spec.method, record)
        grader_final = _math_answer_is_correct(final_value, record)
        counts["with_parsed_candidate"] += int(bool(candidates))
        counts["with_correct_candidate"] += int(has_correct)
        counts["n_final_correct"] += int(final_correct)
        counts["final_grader_disagreements"] += int(final_correct != grader_final)
        if has_correct and final_correct:
            counts["correct_candidate_retained"] += 1
        elif has_correct:
            counts["correct_candidate_lost"] += 1
        elif final_correct:
            counts["rescued_without_correct_candidate"] += 1
        else:
            counts["no_correct_candidate_final_wrong"] += 1
        if spec.method == "JCA":
            raw_steps = (record.get("trajectory") or {}).get("steps", []) or []
            raw_candidates = [
                str(step["tentative_answer"]).strip()
                for step in raw_steps
                if isinstance(step, dict)
                and step.get("tentative_answer") not in (None, "")
            ]
            raw_has_correct = any(
                _math_answer_is_correct(candidate, record)
                for candidate in raw_candidates
            )
            full_trajectory_with_parsed_candidate += int(bool(raw_candidates))
            full_trajectory_with_correct_candidate += int(raw_has_correct)
            full_trajectory_correct_candidate_retained += int(
                raw_has_correct and final_correct
            )
    return FunnelResult(
        dataset=spec.dataset,
        method=spec.method,
        path=spec.path,
        full_trajectory_with_parsed_candidate=(
            full_trajectory_with_parsed_candidate if spec.method == "JCA" else None
        ),
        full_trajectory_with_correct_candidate=(
            full_trajectory_with_correct_candidate if spec.method == "JCA" else None
        ),
        full_trajectory_correct_candidate_retained=(
            full_trajectory_correct_candidate_retained if spec.method == "JCA" else None
        ),
        **counts,
    )


def _analyze_conifer_run(spec: RunSpec) -> FunnelResult:
    """Analyze Conifer run using all_explicit_passed as binary correctness criterion."""
    counts = {
        "n_problems": 0,
        "n_final_correct": 0,
        "with_parsed_candidate": 0,
        "with_correct_candidate": 0,
        "correct_candidate_retained": 0,
        "correct_candidate_lost": 0,
        "rescued_without_correct_candidate": 0,
        "no_correct_candidate_final_wrong": 0,
        "final_grader_disagreements": 0,
    }
    for record in iter_jsonl(spec.path):
        counts["n_problems"] += 1
        candidates = _conifer_candidates(spec.method, record)
        # Candidate is "correct" if all_explicit_passed is True
        has_correct = any(all_explicit for _, all_explicit in candidates)
        final_correct = _conifer_final_correct(record)

        counts["with_parsed_candidate"] += int(bool(candidates))
        counts["with_correct_candidate"] += int(has_correct)
        counts["n_final_correct"] += int(final_correct)

        if has_correct and final_correct:
            counts["correct_candidate_retained"] += 1
        elif has_correct:
            counts["correct_candidate_lost"] += 1
        elif final_correct:
            counts["rescued_without_correct_candidate"] += 1
        else:
            counts["no_correct_candidate_final_wrong"] += 1

    return FunnelResult(
        dataset=spec.dataset,
        method=spec.method,
        path=spec.path,
        **counts,
    )


def _analyze_multipl_replay(spec: RunSpec) -> FunnelResult:
    if spec.replay_path is None or not spec.replay_path.is_file():
        raise FileNotFoundError(f"MultiPL-E candidate replay report missing: {spec.replay_path}")
    if spec.method == "zero-shot":
        try:
            from analysis.coordination_fingerprints.multipl_e_8lang_zero_shot_behavior_analysis import (
                load_candidate_statuses,
                load_final_statuses,
            )
            candidate_statuses = load_candidate_statuses(spec.replay_path.parent)
            final_statuses = load_final_statuses(MULTIPL_E_ZERO_SHOT_COMPLETION_ROOT)
            counts = {
                "n_problems": 0,
                "n_final_correct": 0,
                "with_parsed_candidate": 0,
                "with_correct_candidate": 0,
                "correct_candidate_retained": 0,
                "correct_candidate_lost": 0,
                "rescued_without_correct_candidate": 0,
                "no_correct_candidate_final_wrong": 0,
                "final_grader_disagreements": 0,
            }
            for record in iter_jsonl(spec.path):
                counts["n_problems"] += 1
                key = (
                    str(record.get("root_dataset") or ""),
                    str(record.get("language") or ""),
                    str(record.get("problem_id") or ""),
                )
                candidates = [
                    str(step.get("tentative_completion") or "").strip()
                    for step in (record.get("trajectory") or {}).get("steps", []) or []
                    if str(step.get("tentative_completion") or "").strip()
                ]
                statuses = candidate_statuses.get(key, [])
                has_correct = any(status is True for status in statuses)
                final_correct = final_statuses.get(key) is True
                counts["with_parsed_candidate"] += int(bool(candidates))
                counts["with_correct_candidate"] += int(has_correct)
                counts["n_final_correct"] += int(final_correct)
                if has_correct and final_correct:
                    counts["correct_candidate_retained"] += 1
                elif has_correct:
                    counts["correct_candidate_lost"] += 1
                elif final_correct:
                    counts["rescued_without_correct_candidate"] += 1
                else:
                    counts["no_correct_candidate_final_wrong"] += 1
            return FunnelResult(dataset=spec.dataset, method=spec.method, path=spec.path, **counts)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            raise

    report = json.loads(spec.replay_path.read_text(encoding="utf-8"))
    n_problems = int(report.get("n_tasks", 0))
    with_candidate = int(report.get("n_tasks_with_candidate", 0))
    discovered = int(report.get("candidate_oracle_tasks", 0))
    retained = int(report.get("correct_candidate_retained", 0))
    lost = int(report.get("correct_candidate_lost", 0))
    final_correct = sum(bool(row.get("final_pass")) for row in report.get("task_rows", []))
    if not final_correct:
        final_correct = retained + sum(
            bool(row.get("final_pass")) and not row.get("any_candidate_pass")
            for row in report.get("task_rows", [])
        )
    rescued = final_correct - retained
    no_correct_wrong = n_problems - retained - lost - rescued
    if min(n_problems, with_candidate, discovered, retained, lost, rescued, no_correct_wrong) < 0:
        raise ValueError(f"invalid MultiPL-E replay counts in {spec.replay_path}")
    return FunnelResult(
        dataset=spec.dataset,
        method=spec.method,
        path=spec.path,
        n_problems=n_problems,
        n_final_correct=final_correct,
        with_parsed_candidate=with_candidate,
        with_correct_candidate=discovered,
        correct_candidate_retained=retained,
        correct_candidate_lost=lost,
        rescued_without_correct_candidate=rescued,
        no_correct_candidate_final_wrong=no_correct_wrong,
        final_grader_disagreements=0,
    )


def analyze_run(spec: RunSpec) -> FunnelResult:
    if not spec.applicable:
        return FunnelResult(
            dataset=spec.dataset,
            method=spec.method,
            path=spec.path,
            n_problems=0,
            n_final_correct=0,
            with_parsed_candidate=0,
            with_correct_candidate=0,
            correct_candidate_retained=0,
            correct_candidate_lost=0,
            rescued_without_correct_candidate=0,
            no_correct_candidate_final_wrong=0,
            final_grader_disagreements=0,
            applicable=False,
            note=spec.note,
        )
    if spec.dataset == "MultiPL-E-8Lang":
        return _analyze_multipl_replay(spec)
    if spec.dataset == "MATH":
        return _analyze_math_run(spec)
    if spec.dataset == "Conifer":
        return _analyze_conifer_run(spec)

    n_problems = 0
    n_final_correct = 0
    with_parsed_candidate = 0
    with_correct_candidate = 0
    correct_candidate_retained = 0
    correct_candidate_lost = 0
    rescued_without_correct_candidate = 0
    no_correct_candidate_final_wrong = 0
    final_grader_disagreements = 0

    for record in iter_jsonl(spec.path):
        n_problems += 1
        candidates = intermediate_answers(spec.method_key, record)
        parsed_candidates = [
            candidate
            for candidate in candidates
            if _candidate_value_is_parsed(spec.dataset, candidate)
        ]
        has_correct_candidate = any(
            _answer_is_correct(spec.dataset, candidate, record)
            for candidate in parsed_candidates
        )
        final_correct = _persisted_final_correct(record)
        grader_final_correct = _answer_is_correct(
            spec.dataset,
            record.get("final_answer"),
            record,
        )

        with_parsed_candidate += int(bool(parsed_candidates))
        with_correct_candidate += int(has_correct_candidate)
        n_final_correct += int(final_correct)
        final_grader_disagreements += int(final_correct != grader_final_correct)

        if has_correct_candidate and final_correct:
            correct_candidate_retained += 1
        elif has_correct_candidate:
            correct_candidate_lost += 1
        elif final_correct:
            rescued_without_correct_candidate += 1
        else:
            no_correct_candidate_final_wrong += 1

    result = FunnelResult(
        dataset=spec.dataset,
        method=spec.method,
        path=spec.path,
        n_problems=n_problems,
        n_final_correct=n_final_correct,
        with_parsed_candidate=with_parsed_candidate,
        with_correct_candidate=with_correct_candidate,
        correct_candidate_retained=correct_candidate_retained,
        correct_candidate_lost=correct_candidate_lost,
        rescued_without_correct_candidate=rescued_without_correct_candidate,
        no_correct_candidate_final_wrong=no_correct_candidate_final_wrong,
        final_grader_disagreements=final_grader_disagreements,
    )
    _validate_result(result)
    return result


def _validate_result(result: FunnelResult) -> None:
    if result.correct_candidate_retained + result.correct_candidate_lost != result.with_correct_candidate:
        raise AssertionError(f"{result.dataset}/{result.method}: inconsistent discovered-candidate counts")
    if (
        result.correct_candidate_retained + result.rescued_without_correct_candidate
        != result.n_final_correct
    ):
        raise AssertionError(f"{result.dataset}/{result.method}: inconsistent final-correct counts")
    if (
        result.correct_candidate_retained
        + result.correct_candidate_lost
        + result.rescued_without_correct_candidate
        + result.no_correct_candidate_final_wrong
        != result.n_problems
    ):
        raise AssertionError(f"{result.dataset}/{result.method}: funnel does not cover all records")


def _verify_default_provenance(results: Sequence[FunnelResult]) -> None:
    expected = {
        ("MuSiQue", "JCA"): (2417, 1024, None, None),
        # 2026-09-14：MuSiQue 四个 baseline 改为 2026-09-11 的重跑（AFlow 取官方 5-roll
        # 最高的 run_01）。旧值依次是 AFlow 1019、MAD 943、AgentVerse 1014、GPTSwarm 885。
        ("MuSiQue", "MAD"): (2417, 955, None, None),
        ("MuSiQue", "AgentVerse"): (2417, 1015, None, None),
        ("MuSiQue", "AFlow"): (2417, 1011, None, None),
        ("MuSiQue", "GPTSwarm"): (2417, 875, None, None),
        ("GSM-Hard", "JCA"): (132, 95, 97, 95),
        ("GSM-Hard", "MAD"): (132, 88, 98, 88),
        ("GSM-Hard", "AgentVerse"): (132, 95, 99, 95),
        ("GSM-Hard", "AFlow"): (132, 90, 106, 90),
        ("GSM-Hard", "GPTSwarm"): (132, 97, 105, 97),
        # mpe0912 job 120 / r08；803 与该 roll mas_scores.json 的 correct_count 一致。
        ("MultiPL-E-8Lang", "JCA"): (1352, 803, 517, 508),
        ("MATH", "MAD"): (500, 362, 391, 362),
        ("MATH", "AgentVerse"): (500, 341, 382, 341),
        # 2026-09-14：补上 2026-09-11 MATH 重跑后的实测值（旧值为 AFlow (320, 374, 319)、
        # GPTSwarm (350, 377, 350)，对应换掉的 MODE=eval / TEMPERATURE=0.0 两个旧 run）。
        ("MATH", "AFlow"): (500, 325, 376, 325),
        ("MATH", "GPTSwarm"): (500, 346, 373, 346),
        ("MATH", "JCA"): (500, 392, 400, 392),
    }
    default_paths = {
        ("MuSiQue", "JCA"): DEFAULT_MUSIQUE_JCA,
        ("MuSiQue", "MAD"): DEFAULT_MUSIQUE_MAD,
        ("MuSiQue", "AgentVerse"): DEFAULT_MUSIQUE_AGENTVERSE,
        ("MuSiQue", "AFlow"): DEFAULT_MUSIQUE_AFLOW,
        ("MuSiQue", "GPTSwarm"): DEFAULT_MUSIQUE_GPTSWARM,
        ("GSM-Hard", "JCA"): DEFAULT_GSM_JCA,
        ("GSM-Hard", "MAD"): DEFAULT_GSM_MAD,
        ("GSM-Hard", "AgentVerse"): DEFAULT_GSM_AGENTVERSE,
        ("GSM-Hard", "AFlow"): DEFAULT_GSM_AFLOW,
        ("GSM-Hard", "GPTSwarm"): DEFAULT_GSM_GPTSWARM,
        ("MultiPL-E-8Lang", "JCA"): MULTIPL_E_JCA,
        ("MATH", "MAD"): MATH_MAD,
        ("MATH", "AgentVerse"): MATH_AGENTVERSE,
        ("MATH", "AFlow"): MATH_AFLOW,
        ("MATH", "GPTSwarm"): MATH_GPTSWARM,
        ("MATH", "JCA"): MATH_JCA,
    }
    for result in results:
        key = (result.dataset, result.method)
        if key not in expected or result.path.resolve() != default_paths[key].resolve():
            continue
        n_problems, n_correct, n_discovered, n_retained = expected[key]
        actual = (
            result.n_problems,
            result.n_final_correct,
            result.with_correct_candidate if n_discovered is not None else None,
            result.correct_candidate_retained if n_retained is not None else None,
        )
        wanted = (n_problems, n_correct, n_discovered, n_retained)
        if actual != wanted:
            raise AssertionError(f"{result.dataset}/{result.method}: expected {wanted}, got {actual}")


def _ratio(count: int, denominator: int) -> str:
    percentage = 100.0 * count / denominator if denominator else 0.0
    return f"{count}/{denominator} ({percentage:.2f}%)"


def _retention(result: FunnelResult) -> str:
    if result.retention_rate is None:
        return "—"
    return _ratio(result.correct_candidate_retained, result.with_correct_candidate)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _dataset_table(dataset: str, results: Sequence[FunnelResult]) -> list[str]:
    selected = [result for result in results if result.dataset == dataset]
    lines = [
        f"### {dataset}",
        "",
        "| 方法 | 有可解析候选 | 发现正答率 | 最终正确率 | 丢失正确候选 | 正答保留率 | 无正确候选时救回 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in selected:
        if not result.applicable:
            lines.append(f"| {result.method} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {result.method} | {_ratio(result.with_parsed_candidate, result.n_problems)} "
            f"| {_ratio(result.with_correct_candidate, result.n_problems)} "
            f"| {_ratio(result.n_final_correct, result.n_problems)} "
            f"| {result.correct_candidate_lost} | {_retention(result)} "
            f"| {result.rescued_without_correct_candidate} |"
        )
    return lines


def _result_for(
    results: Sequence[FunnelResult],
    dataset: str,
    method: str,
) -> FunnelResult:
    return next(
        result
        for result in results
        if result.dataset == dataset and result.method == method
    )


def render_report(results: Sequence[FunnelResult]) -> str:
    lines = [
        "# 正确候选发现与保留分析",
        "",
        "## 指标口径",
        "",
        "本报告严格沿用 GSM-Hard 既有报告 7.2 节的协作漏斗口径，并将同一口径扩展到 MuSiQue。对每一道题，先从协议日志中提取**最终答案选择之前**的中间候选：",
        "",
        "- **发现正答率**：至少一个可解析中间候选命中标准答案的题数 / 全部题数。",
        "- **正答保留率**：曾发现正确候选且最终答案正确的题数 / 曾发现正确候选的题数。",
        "- **丢失正确候选**：中间曾出现正确候选、但最终答案错误的题数。",
        "- **无正确候选时救回**：没有中间正确候选、但最终答案正确的题数。",
        "",
        "> “保留”只要求中间曾出现至少一个正确候选且最终答对，不要求所有中间候选都正确。MATH JCA 的最终正确性按三-turn 预算回退投影重算；其余运行使用日志中的 `correct`/`em`，并用数据集 grader 复核。",
        "",
        "**Conifer 不在本报告范围内，且不是因为数据缺失。** 这套漏斗口径要求中间候选有一个二值的“命中标准答案”判据；Conifer 没有 EM，正确性是连续的 `hard_score`（`conifer_hard_score_v1`），唯一的二值口径 `final_checks.all_explicit_passed` 只对最终答案定义，中间 tentative 上没有同构的判据。硬套会把“候选正确”偷换成“候选的 hard_score 达到某个人为阈值”，与另外四个数据集不可比。Conifer 的其余四份分析（W3/W4 token、模型调用、turn、就绪度）已于 2026-09-14 覆盖，见 `benchmark_readiness_report.md`。",
        "",
        "## 汇总结果",
        "",
    ]
    lines.extend(_dataset_table("MuSiQue", results))
    lines.extend([""])
    lines.extend(_dataset_table("GSM-Hard", results))

    lines.extend(["", "## MultiPL-E-8Lang 与 MATH", ""])
    lines.extend(_dataset_table("MultiPL-E-8Lang", results))
    lines.extend([""])
    lines.extend(_dataset_table("MATH", results))
    math_jca = _result_for(results, "MATH", "JCA")
    if math_jca.full_trajectory_with_correct_candidate is not None:
        raw_discovered = math_jca.full_trajectory_with_correct_candidate
        raw_retained = math_jca.full_trajectory_correct_candidate_retained or 0
        lines.extend([
            "",
            (
                "MATH JCA 主表只纳入前三个落盘 turn 的 tentative（包含强制 A3 格式转换首步）。完整 raw 轨迹审计会得到 "
                f"{raw_discovered}/500 题发现正答、{raw_retained}/{raw_discovered} 题保留；"
                "预算后的候选不进入论文主口径。"
            ),
        ])

    musique_jca = _result_for(results, "MuSiQue", "JCA")
    musique_aflow = _result_for(results, "MuSiQue", "AFlow")
    gsm_jca = _result_for(results, "GSM-Hard", "JCA")
    gsm_aflow = _result_for(results, "GSM-Hard", "AFlow")
    lines.extend([
        "",
        "## 直接观察",
        "",
        f"- MuSiQue 上 AFlow 的发现正答率最高（{100 * musique_aflow.discovery_rate:.2f}%），JCA 的正答保留率最高（{100 * (musique_jca.retention_rate or 0.0):.2f}%）。",
        f"- GSM-Hard 上同样是 AFlow 的发现正答率最高（{100 * gsm_aflow.discovery_rate:.2f}%），JCA 的正答保留率最高（{100 * (gsm_jca.retention_rate or 0.0):.2f}%）。",
        "- 因而这两个指标刻画的是不同阶段：发现正答率更接近候选生成覆盖，正答保留率更接近后续协作与最终选择是否破坏已有正确答案；这里只报告相关现象，不据此单独推断因果。",
    ])

    lines.extend([
        "",
        "## 候选字段映射",
        "",
        "| 方法 | 纳入的中间候选 |",
        "|---|---|",
    ])
    for method in METHOD_ORDER:
        lines.append(f"| {method} | {CANDIDATE_SOURCES[method]} |")
    for method in MULTIPL_METHOD_ORDER:
        lines.append(f"| MultiPL-E/{method} | {MULTIPL_CANDIDATE_SOURCES[method]} |")
    lines.extend([
        "| MATH/MAD, AgentVerse, AFlow, GPTSwarm | 各协议中的 answer 字段；AFlow 从成功 solve op 的 raw output 解析 |",
    ])

    lines.extend([
        "",
        "AFlow 只纳入成功 `solve` 操作的候选，不把最终 `answer_generate` 当成中间发现；GPTSwarm 排除最终 Aggregator，避免把最终答案重复计为中间候选。MuSiQue 使用项目 `src.grader.is_correct`（规范化 EM 与 aliases），GSM-Hard 使用既有 7.2 脚本的末尾数值抽取与 `1e-6` 绝对/相对容差。",
        "",
        "## 输入与复核",
        "",
        "| 数据集 | 方法 | 最终分数 | 输入日志 |",
        "|---|---|---:|---|",
    ])
    for result in results:
        if not result.applicable:
            continue
        lines.append(
            f"| {result.dataset} | {result.method} "
            f"| {_ratio(result.n_final_correct, result.n_problems)} "
            f"| `{_relative_path(result.path)}` |"
        )

    disagreement_total = sum(result.final_grader_disagreements for result in results if result.applicable)
    lines.extend([
        "",
        f"- 最终答案 grader 复核与日志 `correct`/`em` 的不一致数：**{disagreement_total}**。",
        "- GSM-Hard 四个 training-free baseline 已精确复现旧报告 7.2：MAD 98/132、AgentVerse 99/132、AFlow 106/132、GPTSwarm 105/132。",
        "- MultiPL-E 的候选漏斗直接读取已完成的 candidate replay 报告；replay 使用原 evaluator，不重新调用模型。",
        "- MATH 的 zero-shot（裸模型 + 零初始化 adapter）与四个 baseline 直接从 no-thinking trajectory 提取中间答案，并使用项目 MATH scorer；MATH JCA 使用最新正式 eval，超过三 turn 或第三 turn 后仍未成功 `stop` 时以首个 `tentative_answer` 作为最终 scorer 输入，候选发现只看前三个落盘 turn，并包含强制 A3 格式转换首步的 tentative。14B self-RL 仍无可比正式输入。",
        "- GSM-Hard JCA 使用论文正式 seed=43 路径，最终正确 95/132（71.97%），不是旧 7.2 使用的 93/132 raw-output 重跑。",
        "- MuSiQue 四个 baseline 均为 2026-09-11 的重跑，与 model_call_count_report / "
        "weighted_output_token_14b_report 同源；AFlow 取 `070_musique_aflow_repeat5` 官方 "
        "5-roll 中 EM 最高的 `run_01`（1011/2417 = 41.83%），替换掉此前的 1019/2417（42.16%）。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python \\",
        "  analysis/coordination_fingerprints/correct_candidate_funnel_analysis.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musique-jca", type=Path, default=DEFAULT_MUSIQUE_JCA)
    parser.add_argument("--musique-mad", type=Path, default=DEFAULT_MUSIQUE_MAD)
    parser.add_argument("--musique-agentverse", type=Path, default=DEFAULT_MUSIQUE_AGENTVERSE)
    parser.add_argument("--musique-aflow", type=Path, default=DEFAULT_MUSIQUE_AFLOW)
    parser.add_argument("--musique-gptswarm", type=Path, default=DEFAULT_MUSIQUE_GPTSWARM)
    parser.add_argument("--gsm-jca", type=Path, default=DEFAULT_GSM_JCA)
    parser.add_argument("--gsm-mad", type=Path, default=DEFAULT_GSM_MAD)
    parser.add_argument("--gsm-agentverse", type=Path, default=DEFAULT_GSM_AGENTVERSE)
    parser.add_argument("--gsm-aflow", type=Path, default=DEFAULT_GSM_AFLOW)
    parser.add_argument("--gsm-gptswarm", type=Path, default=DEFAULT_GSM_GPTSWARM)
    parser.add_argument("--multipl-e-zero-shot", type=Path, default=MULTIPL_E_ZERO_SHOT)
    parser.add_argument("--multipl-e-jca", type=Path, default=MULTIPL_E_JCA)
    parser.add_argument("--multipl-e-mad", type=Path, default=MULTIPL_E_MAD)
    parser.add_argument("--multipl-e-agentverse", type=Path, default=MULTIPL_E_AGENTVERSE)
    parser.add_argument("--multipl-e-aflow", type=Path, default=MULTIPL_E_AFLOW)
    parser.add_argument("--multipl-e-gptswarm", type=Path, default=MULTIPL_E_GPTSWARM)
    parser.add_argument("--multipl-e-self-rl", type=Path, default=MULTIPL_E_SELF_RL)
    parser.add_argument("--multipl-e-replay-root", type=Path, default=MULTIPL_REPLAY_ROOT)
    parser.add_argument("--multipl-e-zero-shot-replay-root", type=Path, default=MULTIPL_ZERO_SHOT_REPLAY_ROOT)
    parser.add_argument("--multipl-e-jca-replay-root", type=Path, default=MULTIPL_JCA_REPLAY_ROOT)
    parser.add_argument("--math-mad", type=Path, default=MATH_MAD)
    parser.add_argument("--math-agentverse", type=Path, default=MATH_AGENTVERSE)
    parser.add_argument("--math-aflow", type=Path, default=MATH_AFLOW)
    parser.add_argument("--math-gptswarm", type=Path, default=MATH_GPTSWARM)
    parser.add_argument("--math-jca", type=Path, default=DEFAULT_MATH_JCA)
    parser.add_argument("--math-zero-shot", type=Path, default=MATH_ZERO_SHOT)
    parser.add_argument("--conifer-jca", type=Path, default=CONIFER_JCA)
    parser.add_argument("--conifer-mad", type=Path, default=CONIFER_MAD)
    parser.add_argument("--conifer-agentverse", type=Path, default=CONIFER_AGENTVERSE)
    parser.add_argument("--conifer-gptswarm", type=Path, default=CONIFER_GPTSWARM)
    parser.add_argument("--conifer-aflow", type=Path, default=CONIFER_AFLOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        RunSpec("MuSiQue", "JCA", args.musique_jca),
        RunSpec("MuSiQue", "MAD", args.musique_mad),
        RunSpec("MuSiQue", "AgentVerse", args.musique_agentverse),
        RunSpec("MuSiQue", "AFlow", args.musique_aflow),
        RunSpec("MuSiQue", "GPTSwarm", args.musique_gptswarm),
        RunSpec("GSM-Hard", "JCA", args.gsm_jca),
        RunSpec("GSM-Hard", "MAD", args.gsm_mad),
        RunSpec("GSM-Hard", "AgentVerse", args.gsm_agentverse),
        RunSpec("GSM-Hard", "AFlow", args.gsm_aflow),
        RunSpec("GSM-Hard", "GPTSwarm", args.gsm_gptswarm),
        RunSpec(
            "MultiPL-E-8Lang", "zero-shot", args.multipl_e_zero_shot,
            args.multipl_e_zero_shot_replay_root / "candidate_replay_report.json",
        ),
        RunSpec(
            "MultiPL-E-8Lang", "JCA", args.multipl_e_jca,
            args.multipl_e_jca_replay_root / "candidate_replay_report.json",
        ),
        RunSpec("MultiPL-E-8Lang", "MAD", args.multipl_e_mad, args.multipl_e_replay_root / "mad/candidate_replay_report.json"),
        RunSpec("MultiPL-E-8Lang", "AgentVerse", args.multipl_e_agentverse, args.multipl_e_replay_root / "agentverse/candidate_replay_report.json"),
        RunSpec("MultiPL-E-8Lang", "AFlow", args.multipl_e_aflow, args.multipl_e_replay_root / "aflow/candidate_replay_report.json"),
        RunSpec("MultiPL-E-8Lang", "GPTSwarm", args.multipl_e_gptswarm, args.multipl_e_replay_root / "gptswarm/candidate_replay_report.json"),
        RunSpec("MultiPL-E-8Lang", "self-RL", args.multipl_e_self_rl, applicable=False, note="SAS one-shot，不存在 MAS 中间候选"),
        RunSpec("MATH", "zero-shot", args.math_zero_shot),
        RunSpec("MATH", "JCA", args.math_jca),
        RunSpec("MATH", "MAD", args.math_mad),
        RunSpec("MATH", "AgentVerse", args.math_agentverse),
        RunSpec("MATH", "AFlow", args.math_aflow),
        RunSpec("MATH", "GPTSwarm", args.math_gptswarm),
        RunSpec("Conifer", "JCA", args.conifer_jca),
        RunSpec("Conifer", "MAD", args.conifer_mad),
        RunSpec("Conifer", "AgentVerse", args.conifer_agentverse),
        RunSpec("Conifer", "GPTSwarm", args.conifer_gptswarm),
        RunSpec("Conifer", "AFlow", args.conifer_aflow),
    ]


def main() -> None:
    args = parse_args()
    results = [analyze_run(spec) for spec in build_specs(args)]
    _verify_default_provenance(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(results), encoding="utf-8")
    print(f"Wrote {args.output}")
    for result in results:
        print(
            f"{result.dataset:8s} {result.method:11s} "
            f"discovery={result.with_correct_candidate}/{result.n_problems} "
            f"retention={result.correct_candidate_retained}/{result.with_correct_candidate}"
        )


if __name__ == "__main__":
    main()
