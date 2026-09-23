"""Compare Qwen3-14B scores on its own rollouts and heterogeneous MAS turns.

The analysis uses the persisted judge outputs. It does not call a model. Every
valid score in each selected file is included; upstream selection of a score
pool is audited and reported as a limitation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for import_path in (ROOT, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from rl_rescore_rollout import parse_step as parse_musique_step  # noqa: E402
from src.data import MuSiQueProblem, load_musique as load_musique_problems  # noqa: E402
from src.grader import compute_em_f1  # noqa: E402


MUSIQUE_DATA_DIR = ROOT / "musique_data"
DEFAULT_MUSIQUE_SELF_PATH = (
    ROOT
    / "baseline/MuSiQue/sas_self_judged_14b_rl/runs/"
    "sas14b_self_rl_20260817_161921/sas_train_judged.jsonl"
)
DEFAULT_MUSIQUE_MAS_PATH = ROOT / "rl_data/rl07151947_rollout_train_0_1000_qwen14b.jsonl"
MUSIQUE_MAS_RAW_PATH = ROOT / "rl_data/rl07081658_rollout_train_0_1000_final.jsonl"
DEFAULT_GSM_SELF_PATH = (
    ROOT
    / "baseline/GSM-Hard/sas_self_judged_14b_rl/runs/"
    "gsm_sas14b_self_rl_v1/sas_train_judged.jsonl"
)
DEFAULT_GSM_MAS_PATH = (
    ROOT / "rl_data/gsm/judge_rl/v13_14b/source_rejudged.jsonl"
)
DEFAULT_GSM_SELECTED_PATH = (
    ROOT / "rl_data/gsm/judge_rl/gsm_judge_rl_v13_14b_role_c2c_1_05_1/train.jsonl"
)
DEFAULT_MULTIPL_E_SELF_PATH = ROOT / (
    "logs/multipl_e_8lang_baselines/sas_self_judged_14b_rl/"
    "multipl_e_8lang_sas_self_rl_normalized_v2/train_scored.jsonl"
)
DEFAULT_MULTIPL_E_MAS_PATH = ROOT / (
    "rl_data/clean/"
    "multipl_e_rl_a1a2_sft_a3base_0811_judged_qwen14b_perturn_alpha03_clean.jsonl"
)
MULTIPL_E_RAW_FULL_PATH = (
    ROOT / "rl_data/multipl_e_rl_a1a2_sft_a3base_0805_full.jsonl"
)
MULTIPL_E_RAW_REROLL_PATH = (
    ROOT / "rl_data/multipl_e_rl_a1a2_sft_a3base_0805_reroll_bad.jsonl"
)
MULTIPL_E_FULL_FILTER_STATS_PATH = ROOT / (
    "logs/multipl_e_8lang_rl_rollout/"
    "multipl_e_rl_a1a2_sft_a3base_0805_full/cleaning/"
    "clean_success_excl_reroll_shsalvage_stats.json"
)
MULTIPL_E_REROLL_FILTER_STATS_PATH = ROOT / (
    "logs/multipl_e_8lang_rl_rollout/"
    "multipl_e_rl_a1a2_sft_a3base_0805_reroll_bad/cleaning/"
    "repair_success_shsalvage_stats.json"
)
MULTIPL_E_FINAL_TRAIN_LOG = ROOT / (
    "rl_runs/multipl_e_goal/multipl_e_rl_ckpt_search_20260813_000356/launch.log"
)

# 2026-09-12：MultiPL-E 的正式最终 JCA eval 已经产出（mpe0912 job 120 / r08，
# 803/1352 = 59.39%），所以下面「最终产物链路对齐审计」里那一行不再写「尚无」。
# 但它并不能把该行的对齐结论变强，反而要写得更弱：r08 的三个 adapter 全部来自
# gpt5_paired_rwr 支线——
#   A1 = rl_runs/multipl_e_a1_8b_gpt5_paired_rwr_20260830（训练输入
#        rl_data/clean/multipl_e_8lang_best_sft_gpt5_a1_paired_core_20260825.jsonl）
#   A2 = rl_runs/multipl_e_a2_gpt5_paired_rwr_20260827（训练输入
#        rl_data/clean/multipl_e_8lang_best_sft_gpt5_a2_paired_core_20260827.jsonl）
#   A3 = sft_runs/multipl_e_8lang_v5_diverse_a1_5ep/A1/final（SFT，非 RL）
# 这两个 launch.log 里对 multipl_e_rl_ckpt_search_20260813_000356 的引用次数都是 0，
# 即本行审计的 14B-judged per-turn 评分池（alpha03_clean）不是 r08 adapter 的训练
# 入口。评分池对齐结论只适用于 0813 ckpt-search 那一支，不适用于已登记的 r08。
MULTIPL_E_REGISTERED_EVAL = ROOT / (
    "logs/baseline_queue/mpe0912/120_multipl_e_gpt5rwr_ckpt25_x10/"
    "mpe0912_120_multipl_e_gpt5rwr_ckpt25_x10_r08/mas_scores.json"
)
MULTIPL_E_REGISTERED_EVAL_SCORE = "803/1352 = 59.39%"
MULTIPL_E_REGISTERED_ADAPTER_LINEAGE = "gpt5_paired_rwr（A1 ckpt-25 / A2 ckpt-120）+ A3 SFT final"
DEFAULT_MATH_MAS_PATH = ROOT / (
    "experiments/math_rl_mas_thinking/artifacts/"
    "math_rl_mas_v13_nonthinking_global_turn_json_object_1p7b_4b_8b_corrected_v4d/"
    "02_score/train_judged_corrected.jsonl"
)
MATH_SELF_SMOKE_PATH = ROOT / (
    "logs/math_baselines/sas_self_judged_14b_rl/"
    "math_sas_parserfix_smoke3_20260824/02_score/train_judged.jsonl"
)
DEFAULT_MATH_REUSE_STATS_PATH = ROOT / (
    "experiments/math_specific_sft_rl_v1/artifacts/"
    "math_specific_sft_rl_reuse_legacy_v4d_real_rollout_sft_protocol_floor_v2_20260830/"
    "03_score/reuse_stats.json"
)
DEFAULT_MUSIQUE_FINAL_TRAIN_PATH = ROOT / "rl_data/rl_14b_qwen_0716.jsonl"
DEFAULT_MUSIQUE_FINAL_A3_TRAIN_PATH = ROOT / "rl_data/rl_14b_qwen_0716_a3confirm.jsonl"
MUSIQUE_FINAL_EVAL_PATH = ROOT / "logs/mas_eval_concurrent/rl_sft_mas_0717_1130/results.jsonl"
MUSIQUE_FINAL_ADAPTERS = (
    ROOT / "rl_runs/rl_14B_0716_v2/A1/final",
    ROOT / "rl_runs/rl_14B_0716_v2/A2/final",
    ROOT / "rl_runs/rl_14B_0716_v3/A3/final",
)
GSM_FINAL_EVAL_PATH = ROOT / (
    "outputs/gsm_eval/role_batched/gsm_judge_rl_v13_seed_sweep_infinite_20260809/"
    "gsm_judge_rl_v13_seed_sweep_infinite_20260809_seed43_dev132_"
    "thinking_hidden_self_handoff_ctx40960.jsonl"
)
GSM_FINAL_ADAPTER_ROOT = ROOT / (
    "rl_runs/gsm_judge_rl/gsm_judge_rl_v13_14b_role_c2c_1_05_1"
)
MATH_FINAL_RUN_ROOT = ROOT / (
    "experiments/math_specific_sft_rl_v1/artifacts/"
    "math_specific_sft_rl_reuse_legacy_v4d_real_rollout_sft_protocol_floor_v2_20260830"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "judge_score_bias_report.md"


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


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def binary_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    score = numeric(value)
    return None if score is None else score > 0.5


@dataclass(frozen=True)
class Observation:
    problem_id: str
    rollout_idx: str
    score: float
    correct: bool | None
    reasoning_score: float | None = None
    turn: int = 0
    outcome_correct: bool | None = None

    @property
    def trajectory_id(self) -> tuple[str, str]:
        return self.problem_id, self.rollout_idx


@dataclass
class RunData:
    dataset: str
    source: str
    path: Path
    total_records: int
    observations: list[Observation]
    score_audit: ScoreAudit | None = None
    turn_correctness_audit: MusiqueTurnCorrectnessAudit | None = None

    @property
    def rejected_records(self) -> int:
        return self.total_records - len(self.observations)


@dataclass(frozen=True)
class MetricSummary:
    n: int
    trajectories: int
    problems: int
    mean: float
    trajectory_mean: float
    positive_rate: float
    full_score_rate: float
    correct_n: int
    correct_mean: float
    correct_positive_rate: float
    correct_full_score_rate: float
    incorrect_n: int
    incorrect_mean: float
    incorrect_positive_rate: float
    incorrect_full_score_rate: float


@dataclass(frozen=True)
class PairedSummary:
    common_problems: int
    self_mean: float
    mas_mean: float
    difference: float
    ci_low: float
    ci_high: float
    self_greater_rate: float


@dataclass(frozen=True)
class OutcomePairedSummary:
    common_problems: int
    self_records: int
    mas_records: int
    self_mean: float
    mas_mean: float
    difference: float
    ci_low: float
    ci_high: float
    self_greater_rate: float


@dataclass(frozen=True)
class OutcomePoolAudit:
    rows: int
    trajectories: int
    correct_trajectories: int
    incorrect_trajectories: int
    unknown_trajectories: int


@dataclass(frozen=True)
class ScoreAudit:
    score_path: str
    judge_dimensions: str
    formula: str
    checked: int
    mismatches: int
    note: str = ""


@dataclass(frozen=True)
class MusiqueTurnCorrectnessAudit:
    source_rows: int
    valid_score_rows: int
    known_rows: int
    correct_rows: int
    incorrect_rows: int
    unknown_rows: int
    new_candidate_rows: int
    carried_candidate_rows: int
    confirmed_rows: int
    parse_failures: int


@dataclass(frozen=True)
class TrainingFileAudit:
    total: int
    explicit_qwen14b: int
    inherited: int
    formula_checked: int
    formula_mismatches: int


@dataclass(frozen=True)
class EvalSummary:
    total: int
    correct: int

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else math.nan


@dataclass(frozen=True)
class GsmTrainingAlignmentAudit:
    selected: int
    matched_to_source: int
    source_v3_v4_mismatches: int
    judge_score_mismatches: int
    reward_field_mismatches: int


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} input does not exist: {path}")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def replay_musique_turn_correctness(
    records: list[dict[str, Any]],
    problem_by_id: dict[str, MuSiQueProblem],
) -> tuple[list[bool | None], dict[str, int]]:
    indexed_groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        problem_id = str(record.get("problem_id") or "")
        rollout_idx = str(record.get("rollout_idx", 0))
        if problem_id not in problem_by_id:
            raise ValueError(f"MuSiQue turn replay has unknown problem_id: {problem_id!r}")
        indexed_groups[(problem_id, rollout_idx)].append((index, record))

    labels: list[bool | None] = [None] * len(records)
    counts: dict[str, int] = defaultdict(int)
    for (problem_id, _rollout_idx), indexed_records in indexed_groups.items():
        indexed_records.sort(
            key=lambda item: (int(item[1].get("turn", 0) or 0), item[0])
        )
        prior_tentative = False
        current_answer = ""
        for index, record in indexed_records:
            parsed = parse_musique_step(record, prior_tentative)
            if parsed.action is None:
                counts["parse_failures"] += 1
            if parsed.tentative_answer:
                current_answer = parsed.tentative_answer
                prior_tentative = True
                counts["new_candidate_rows"] += 1
            if parsed.final_answer:
                current_answer = parsed.final_answer
                counts["confirmed_rows"] += 1
            elif not parsed.tentative_answer and current_answer:
                counts["carried_candidate_rows"] += 1
            if current_answer:
                em, _f1 = compute_em_f1(current_answer, problem_by_id[problem_id])
                labels[index] = em > 0.5
            else:
                counts["no_candidate_rows"] += 1
    return labels, counts


def load_musique(
    path: Path,
    source: str,
    *,
    replay_turn_correctness: bool = False,
    data_dir: Path = MUSIQUE_DATA_DIR,
) -> RunData:
    require_file(path, f"MuSiQue {source}")
    records = list(iter_jsonl(path))
    replay_labels: list[bool | None] | None = None
    replay_counts: dict[str, int] = {}
    if replay_turn_correctness:
        problem_by_id = {
            problem.id: problem
            for problem in load_musique_problems("train", data_dir=data_dir)
        }
        replay_labels, replay_counts = replay_musique_turn_correctness(
            records, problem_by_id
        )
    observations: list[Observation] = []
    total_records = len(records)
    checked = 0
    mismatches = 0
    replay_correct = 0
    replay_incorrect = 0
    replay_unknown = 0
    secondary_field = "finalization_score" if source == "14B 自评" else "action_score"
    for index, record in enumerate(records):
        if record.get("judge_failed") is not False:
            continue
        if record.get("judge_model") != "qwen14b_judge":
            continue
        score = numeric(record.get("judge_score"))
        if score is None:
            continue
        turn_scores = record.get("turn_scores") or {}
        reasoning_score = numeric(turn_scores.get("reasoning_score"))
        secondary_score = numeric(turn_scores.get(secondary_field))
        nested_judge_score = numeric(turn_scores.get("judge_score"))
        if reasoning_score is not None and secondary_score is not None:
            checked += 1
            expected = 0.5 * reasoning_score + 0.5 * secondary_score
            if not math.isclose(score, expected, abs_tol=1e-6) or (
                nested_judge_score is not None
                and not math.isclose(score, nested_judge_score, abs_tol=1e-6)
            ):
                mismatches += 1
        if replay_labels is None:
            correct = binary_correct(record.get("em"))
        else:
            correct = replay_labels[index]
            replay_correct += int(correct is True)
            replay_incorrect += int(correct is False)
            replay_unknown += int(correct is None)
        observations.append(
            Observation(
                problem_id=str(record.get("problem_id") or ""),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                reasoning_score=reasoning_score,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=binary_correct(record.get("em")),
            )
        )
    turn_correctness_audit = None
    if replay_labels is not None:
        turn_correctness_audit = MusiqueTurnCorrectnessAudit(
            source_rows=total_records,
            valid_score_rows=len(observations),
            known_rows=replay_correct + replay_incorrect,
            correct_rows=replay_correct,
            incorrect_rows=replay_incorrect,
            unknown_rows=replay_unknown,
            new_candidate_rows=replay_counts.get("new_candidate_rows", 0),
            carried_candidate_rows=replay_counts.get("carried_candidate_rows", 0),
            confirmed_rows=replay_counts.get("confirmed_rows", 0),
            parse_failures=replay_counts.get("parse_failures", 0),
        )
    return RunData(
        "MuSiQue",
        source,
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="judge_score",
            judge_dimensions=f"reasoning_score + {secondary_field}",
            formula=f"0.5 × reasoning_score + 0.5 × {secondary_field}",
            checked=checked,
            mismatches=mismatches,
        ),
        turn_correctness_audit,
    )


def load_gsm_self(path: Path) -> RunData:
    require_file(path, "GSM-Hard self")
    observations: list[Observation] = []
    total_records = 0
    checked = 0
    mismatches = 0
    for record in iter_jsonl(path):
        total_records += 1
        if record.get("judge_failed") is not False:
            continue
        if record.get("judge_model") != "qwen14b_judge":
            continue
        score = numeric(record.get("judge_score"))
        process_score = numeric(record.get("process_score"))
        if score is None:
            score = process_score
        if score is None:
            continue
        if process_score is not None:
            checked += 1
            if not math.isclose(score, process_score, abs_tol=1e-6):
                mismatches += 1
        correct = binary_correct(record.get("em"))
        observations.append(
            Observation(
                problem_id=str(record.get("problem_id") or ""),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=correct,
            )
        )
    return RunData(
        "GSM-Hard",
        "14B 自评",
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="judge_score",
            judge_dimensions="process_score（单一 judge 维度）",
            formula="judge_score = process_score",
            checked=checked,
            mismatches=mismatches,
            note="outcome_score 由数值 EM 确定，不是 judge 子分；训练 reward = 0.6 × outcome_score + 0.4 × judge_score。",
        ),
    )


def gsm_all_failed_problem_ids(
    records: list[dict[str, Any]], expected_rollouts_per_problem: int = 8
) -> set[str]:
    outcomes: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        problem_id = str(record.get("problem_id") or "")
        rollout_idx = int(record.get("rollout_idx", -1))
        terminal_em = numeric(record.get("em"))
        if not problem_id or rollout_idx < 0 or terminal_em not in {0.0, 1.0}:
            raise ValueError("GSM source has an invalid problem, rollout, or terminal EM")
        previous = outcomes[problem_id].get(rollout_idx)
        if previous is not None and previous != terminal_em:
            raise ValueError(f"inconsistent terminal EM for {problem_id} rollout {rollout_idx}")
        outcomes[problem_id][rollout_idx] = terminal_em
    incomplete = {
        problem_id: len(rollouts)
        for problem_id, rollouts in outcomes.items()
        if len(rollouts) != expected_rollouts_per_problem
    }
    if incomplete:
        raise ValueError(f"unexpected GSM rollout counts: {sorted(incomplete.items())[:5]}")
    return {
        problem_id
        for problem_id, rollouts in outcomes.items()
        if not any(rollouts.values())
    }


def load_gsm_mas(
    path: Path,
    source: str,
    *,
    drop_all_failed_problems: bool = False,
) -> RunData:
    require_file(path, f"GSM-Hard {source}")
    observations: list[Observation] = []
    records = list(iter_jsonl(path))
    total_records = len(records)
    dropped_problem_ids = (
        gsm_all_failed_problem_ids(records) if drop_all_failed_problems else set()
    )
    checked = 0
    mismatches = 0
    for record in records:
        if str(record.get("problem_id") or "") in dropped_problem_ids:
            continue
        if record.get("collaboration_judge_status_v4") != "scored":
            continue
        if record.get("collaboration_judge_model_v4") != "qwen14b_gsm_judge":
            continue
        judge = record.get("collaboration_judge_v4") or {}
        score = numeric(judge.get("judge_score"))
        process_score = numeric(judge.get("process_score"))
        if score is None:
            score = process_score
        state = record.get("deterministic_state_v3") or {}
        correct = binary_correct(state.get("current_answer_correct"))
        if score is None:
            continue
        if process_score is not None:
            checked += 1
            if not math.isclose(score, process_score, abs_tol=1e-6):
                mismatches += 1
        observations.append(
            Observation(
                problem_id=str(record.get("problem_id") or ""),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=binary_correct(record.get("em")),
            )
        )
    return RunData(
        "GSM-Hard",
        source,
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="collaboration_judge_v4.judge_score",
            judge_dimensions="process_score（compact v4 单一 judge 维度）",
            formula="judge_score = process_score",
            checked=checked,
            mismatches=mismatches,
            note=(
                f"按最终训练规则排除 {len(dropped_problem_ids)} 个 8 次 rollout 全失败题；"
                if drop_all_failed_problems
                else ""
            )
            + "忽略顶层旧 judge_score/turn_scores；它们属于 14B v4 重打分之前的历史 judge。",
        ),
    )


def _multipl_problem_id(record: dict[str, Any]) -> str:
    problem_id = str(record.get("problem_id") or "")
    root_dataset = str(record.get("root_dataset") or record.get("dataset") or "")
    language = str(record.get("language") or "")
    parts = [part for part in (root_dataset, language, problem_id) if part]
    return ":".join(parts) if parts else ""


def inspect_outcome_pool(
    path: Path,
    label: str,
    *,
    multipl_e: bool = False,
) -> OutcomePoolAudit:
    require_file(path, label)
    rows = 0
    latest_outcomes: dict[tuple[str, str], tuple[int, bool | None]] = {}
    for record in iter_jsonl(path):
        rows += 1
        problem_id = (
            _multipl_problem_id(record)
            if multipl_e
            else str(record.get("problem_id") or "")
        )
        trajectory_id = problem_id, str(record.get("rollout_idx", 0))
        turn = int(record.get("turn", 0) or 0)
        passed = record.get("passed")
        outcome = passed if isinstance(passed, bool) else binary_correct(record.get("em"))
        previous = latest_outcomes.get(trajectory_id)
        if previous is None or turn > previous[0]:
            latest_outcomes[trajectory_id] = turn, outcome
        elif turn == previous[0] and outcome != previous[1]:
            raise ValueError(f"conflicting outcomes for {label} trajectory {trajectory_id}")
    outcomes = [value for _, value in latest_outcomes.values()]
    return OutcomePoolAudit(
        rows=rows,
        trajectories=len(outcomes),
        correct_trajectories=sum(value is True for value in outcomes),
        incorrect_trajectories=sum(value is False for value in outcomes),
        unknown_trajectories=sum(value is None for value in outcomes),
    )


def load_multipl_e_self(path: Path) -> RunData:
    require_file(path, "MultiPL-E-8Lang self")
    observations: list[Observation] = []
    total_records = 0
    checked = 0
    mismatches = 0
    for record in iter_jsonl(path):
        total_records += 1
        if record.get("judge_failed") is not False:
            continue
        if record.get("judge_model") != "qwen14b_judge":
            continue
        score = numeric(record.get("judge_score"))
        if score is None:
            continue
        turn_scores = record.get("turn_scores") or {}
        reasoning_score = numeric(turn_scores.get("reasoning_score"))
        finalization_score = numeric(turn_scores.get("code_finalization_score"))
        nested_judge_score = numeric(turn_scores.get("judge_score"))
        if reasoning_score is not None and finalization_score is not None:
            checked += 1
            expected = 0.5 * reasoning_score + 0.5 * finalization_score
            if not math.isclose(score, expected, abs_tol=1e-6) or (
                nested_judge_score is not None
                and not math.isclose(score, nested_judge_score, abs_tol=1e-6)
            ):
                mismatches += 1
        passed = record.get("passed")
        correct = passed if isinstance(passed, bool) else binary_correct(record.get("task_reward"))
        observations.append(
            Observation(
                problem_id=_multipl_problem_id(record),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                reasoning_score=reasoning_score,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=correct,
            )
        )
    return RunData(
        "MultiPL-E-8Lang",
        "14B 自评",
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="judge_score",
            judge_dimensions="reasoning_score + code_finalization_score",
            formula="0.5 × reasoning_score + 0.5 × code_finalization_score",
            checked=checked,
            mismatches=mismatches,
        ),
    )


def load_multipl_e_mas(path: Path, source: str = "14B 评价异构 MAS") -> RunData:
    require_file(path, f"MultiPL-E-8Lang {source}")
    observations: list[Observation] = []
    total_records = 0
    checked = 0
    mismatches = 0
    for record in iter_jsonl(path):
        total_records += 1
        if record.get("judge_failed") is not False:
            continue
        status = str(record.get("judge_status") or "")
        if "qwen14b_multipl_e_judge" not in status:
            continue
        score = numeric(record.get("judge_score"))
        if score is None:
            continue
        turn_scores = record.get("turn_scores") or {}
        reasoning_score = numeric(turn_scores.get("reasoning_score"))
        action_score = numeric(turn_scores.get("action_score"))
        nested_judge_score = numeric(turn_scores.get("judge_score"))
        if reasoning_score is not None and action_score is not None:
            checked += 1
            expected = 0.5 * reasoning_score + 0.5 * action_score
            if not math.isclose(score, expected, abs_tol=1e-6) or (
                nested_judge_score is not None
                and not math.isclose(score, nested_judge_score, abs_tol=1e-6)
            ):
                mismatches += 1
        passed = record.get("passed")
        correct = passed if isinstance(passed, bool) else binary_correct(record.get("task_reward"))
        observations.append(
            Observation(
                problem_id=_multipl_problem_id(record),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                reasoning_score=reasoning_score,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=correct,
            )
        )
    return RunData(
        "MultiPL-E-8Lang",
        source,
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="judge_score",
            judge_dimensions="reasoning_score + action_score",
            formula="0.5 × reasoning_score + 0.5 × action_score",
            checked=checked,
            mismatches=mismatches,
        ),
    )


def load_math_mas(path: Path, source: str = "14B 评价异构 MAS") -> RunData:
    require_file(path, f"MATH {source}")
    observations: list[Observation] = []
    total_records = 0
    checked = 0
    mismatches = 0
    for record in iter_jsonl(path):
        total_records += 1
        if record.get("collaboration_judge_status_v4") != "scored":
            continue
        if record.get("collaboration_judge_model_v4") != "qwen14b_gsm_judge":
            continue
        judge = record.get("collaboration_judge_v4") or {}
        score = numeric(judge.get("judge_score"))
        process_score = numeric(judge.get("process_score"))
        if score is None:
            score = process_score
        if score is None:
            continue
        if process_score is not None:
            checked += 1
            if not math.isclose(score, process_score, abs_tol=1e-6):
                mismatches += 1
        state = record.get("deterministic_state_v3") or {}
        correct = binary_correct(state.get("current_answer_correct"))
        terminal_correct = binary_correct(record.get("em"))
        observations.append(
            Observation(
                problem_id=str(record.get("problem_id") or ""),
                rollout_idx=str(record.get("rollout_idx", 0)),
                score=score,
                correct=correct,
                turn=int(record.get("turn", 0) or 0),
                outcome_correct=correct if terminal_correct is None else terminal_correct,
            )
        )
    return RunData(
        "MATH",
        source,
        path,
        total_records,
        observations,
        ScoreAudit(
            score_path="collaboration_judge_v4.judge_score",
            judge_dimensions="process_score（compact MATH 单一 judge 维度）",
            formula="judge_score = process_score",
            checked=checked,
            mismatches=mismatches,
            note="答案正确性来自 deterministic MATH evaluator，不是第二个 judge 子分。",
        ),
    )


def mean_or_nan(values: list[float]) -> float:
    return fmean(values) if values else math.nan


def rate(values: list[float], predicate: Callable[[float], bool]) -> float:
    return sum(predicate(value) for value in values) / len(values) if values else math.nan


def metric_summary(
    observations: list[Observation],
    score_getter: Callable[[Observation], float | None] = lambda item: item.score,
) -> MetricSummary:
    rows = [(item, score_getter(item)) for item in observations]
    rows = [(item, score) for item, score in rows if score is not None]
    scores = [score for _, score in rows]
    trajectory_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item, score in rows:
        trajectory_scores[item.trajectory_id].append(score)

    correct_scores = [score for item, score in rows if item.correct is True]
    incorrect_scores = [score for item, score in rows if item.correct is False]
    full_score = lambda value: math.isclose(value, 1.0, abs_tol=1e-12)
    return MetricSummary(
        n=len(scores),
        trajectories=len(trajectory_scores),
        problems=len({item.problem_id for item, _ in rows}),
        mean=mean_or_nan(scores),
        trajectory_mean=mean_or_nan([fmean(values) for values in trajectory_scores.values()]),
        positive_rate=rate(scores, lambda value: value > 0),
        full_score_rate=rate(scores, full_score),
        correct_n=len(correct_scores),
        correct_mean=mean_or_nan(correct_scores),
        correct_positive_rate=rate(correct_scores, lambda value: value > 0),
        correct_full_score_rate=rate(correct_scores, full_score),
        incorrect_n=len(incorrect_scores),
        incorrect_mean=mean_or_nan(incorrect_scores),
        incorrect_positive_rate=rate(incorrect_scores, lambda value: value > 0),
        incorrect_full_score_rate=rate(incorrect_scores, full_score),
    )


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(differences: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not differences or samples <= 0:
        return math.nan, math.nan
    generator = random.Random(seed)
    size = len(differences)
    bootstrap_means = [
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    bootstrap_means.sort()
    return quantile(bootstrap_means, 0.025), quantile(bootstrap_means, 0.975)


def paired_summary(
    self_observations: list[Observation],
    mas_observations: list[Observation],
    score_getter: Callable[[Observation], float | None],
    bootstrap_samples: int,
    seed: int,
) -> PairedSummary:
    def problem_means(observations: list[Observation]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for item in observations:
            score = score_getter(item)
            if score is not None:
                values[item.problem_id].append(score)
        return {problem_id: fmean(scores) for problem_id, scores in values.items()}

    self_means = problem_means(self_observations)
    mas_means = problem_means(mas_observations)
    common = sorted(self_means.keys() & mas_means.keys())
    differences = [self_means[key] - mas_means[key] for key in common]
    ci_low, ci_high = bootstrap_ci(differences, bootstrap_samples, seed)
    return PairedSummary(
        common_problems=len(common),
        self_mean=mean_or_nan([self_means[key] for key in common]),
        mas_mean=mean_or_nan([mas_means[key] for key in common]),
        difference=mean_or_nan(differences),
        ci_low=ci_low,
        ci_high=ci_high,
        self_greater_rate=rate(differences, lambda value: value > 0),
    )


def outcome_paired_summary(
    self_observations: list[Observation],
    mas_observations: list[Observation],
    score_getter: Callable[[Observation], float | None],
    outcome: bool,
    bootstrap_samples: int,
    seed: int,
) -> OutcomePairedSummary:
    def outcome_scores_by_problem(
        observations: list[Observation],
    ) -> dict[str, list[float]]:
        values: dict[str, list[float]] = defaultdict(list)
        for item in observations:
            score = score_getter(item)
            if item.correct is outcome and score is not None:
                values[item.problem_id].append(score)
        return values

    self_scores = outcome_scores_by_problem(self_observations)
    mas_scores = outcome_scores_by_problem(mas_observations)
    common = sorted(self_scores.keys() & mas_scores.keys())
    self_problem_means = [fmean(self_scores[key]) for key in common]
    mas_problem_means = [fmean(mas_scores[key]) for key in common]
    differences = [
        self_score - mas_score
        for self_score, mas_score in zip(self_problem_means, mas_problem_means)
    ]
    ci_low, ci_high = bootstrap_ci(differences, bootstrap_samples, seed)
    return OutcomePairedSummary(
        common_problems=len(common),
        self_records=sum(len(self_scores[key]) for key in common),
        mas_records=sum(len(mas_scores[key]) for key in common),
        self_mean=mean_or_nan(self_problem_means),
        mas_mean=mean_or_nan(mas_problem_means),
        difference=mean_or_nan(differences),
        ci_low=ci_low,
        ci_high=ci_high,
        self_greater_rate=rate(differences, lambda value: value > 0),
    )


def inspect_musique_final_training(path: Path) -> TrainingFileAudit:
    require_file(path, "MuSiQue final training")
    total = 0
    explicit_qwen = 0
    formula_checked = 0
    formula_mismatches = 0
    for record in iter_jsonl(path):
        total += 1
        if record.get("judge_model") == "qwen14b_judge":
            explicit_qwen += 1
        reward = numeric(record.get("reward"))
        task_reward = numeric(record.get("task_reward"))
        judge_score = numeric(record.get("judge_score"))
        if reward is not None and task_reward is not None and judge_score is not None:
            formula_checked += 1
            expected = 0.6 * task_reward + 0.4 * judge_score
            if not math.isclose(reward, expected, abs_tol=1e-4):
                formula_mismatches += 1
    return TrainingFileAudit(
        total=total,
        explicit_qwen14b=explicit_qwen,
        inherited=total - explicit_qwen,
        formula_checked=formula_checked,
        formula_mismatches=formula_mismatches,
    )


def inspect_eval(path: Path, label: str) -> EvalSummary:
    require_file(path, label)
    total = 0
    correct = 0
    for record in iter_jsonl(path):
        total += 1
        value = binary_correct(record.get("em"))
        if value is None and isinstance(record.get("correct"), bool):
            value = record["correct"]
        correct += int(value is True)
    return EvalSummary(total=total, correct=correct)


def audit_gsm_final_training(
    source_path: Path, selected_path: Path
) -> GsmTrainingAlignmentAudit:
    source_rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    source_v3_v4_mismatches = 0
    for record in iter_jsonl(source_path):
        key = (
            str(record.get("problem_id") or ""),
            int(record.get("rollout_idx", -1)),
            int(record.get("turn", -1)),
        )
        if key in source_rows:
            raise ValueError(f"duplicate GSM source identity: {key}")
        source_rows[key] = record
        score_v3 = numeric((record.get("collaboration_judge_v3") or {}).get("judge_score"))
        score_v4 = numeric((record.get("collaboration_judge_v4") or {}).get("judge_score"))
        if (
            score_v3 is None
            or score_v4 is None
            or not math.isclose(score_v3, score_v4, abs_tol=1e-12)
        ):
            source_v3_v4_mismatches += 1
    selected = 0
    matched = 0
    score_mismatches = 0
    reward_field_mismatches = 0
    for record in iter_jsonl(selected_path):
        selected += 1
        key = (
            str(record.get("problem_id") or ""),
            int(record.get("rollout_idx", -1)),
            int(record.get("turn", -1)),
        )
        source = source_rows.get(key)
        if source is not None:
            matched += 1
            source_score = numeric((source.get("collaboration_judge_v4") or {}).get("judge_score"))
            selected_score = numeric(
                (record.get("collaboration_judge_v4") or {}).get("judge_score")
            )
            if (
                source_score is None
                or selected_score is None
                or not math.isclose(source_score, selected_score, abs_tol=1e-12)
            ):
                score_mismatches += 1
        reward = numeric(record.get("reward"))
        train_weight = numeric(record.get("train_weight"))
        if (
            reward is None
            or train_weight is None
            or not math.isclose(reward, train_weight, abs_tol=1e-12)
        ):
            reward_field_mismatches += 1
    return GsmTrainingAlignmentAudit(
        selected=selected,
        matched_to_source=matched,
        source_v3_v4_mismatches=source_v3_v4_mismatches,
        judge_score_mismatches=score_mismatches,
        reward_field_mismatches=reward_field_mismatches,
    )


def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}%}"


def audit_row(run: RunData | None) -> str:
    if run is None or run.score_audit is None:
        return "| — | — | — | — | n/a | — |"
    audit = run.score_audit
    result = "通过" if audit.mismatches == 0 else f"失败（{audit.mismatches} 个不一致）"
    return (
        f"| {run.dataset} | {run.source} | `{audit.score_path}` | "
        f"{audit.formula} | {result}（{audit.checked} 条） | {audit.note or '—'} |"
    )


def outcome_paired_row(
    dataset: str,
    metric: str,
    outcome_label: str,
    summary: OutcomePairedSummary | None,
) -> str:
    if summary is None:
        return (
            f"| {dataset} | `{metric}` | {outcome_label} | n/a | n/a | n/a | "
            "n/a | n/a | n/a | n/a | n/a |"
        )
    if summary.common_problems == 0:
        return (
            f"| {dataset} | `{metric}` | {outcome_label} | 0 | 0 | 0 | "
            "n/a | n/a | n/a | n/a | n/a |"
        )
    return (
        f"| {dataset} | `{metric}` | {outcome_label} | {summary.common_problems} | "
        f"{summary.self_records} | {summary.mas_records} | "
        f"{summary.self_mean:.4f} | {summary.mas_mean:.4f} | "
        f"{summary.difference:+.4f} | [{summary.ci_low:.4f}, {summary.ci_high:.4f}] | "
        f"{pct(summary.self_greater_rate)} |"
    )


def render_report(
    musique_self: RunData,
    musique_mas: RunData,
    gsm_self: RunData,
    gsm_raw: RunData,
    gsm_mas: RunData,
    gsm_selected: RunData,
    musique_final_train_path: Path,
    musique_final_a3_train_path: Path,
    musique_final_audit: TrainingFileAudit,
    musique_final_a3_audit: TrainingFileAudit,
    musique_eval: EvalSummary,
    gsm_eval: EvalSummary,
    gsm_training_alignment: GsmTrainingAlignmentAudit,
    pool_audits: dict[str, OutcomePoolAudit],
    multipl_e_filter_stats: dict[str, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    output_path: Path,
    multipl_e_self: RunData | None = None,
    multipl_e_mas: RunData | None = None,
    math_mas: RunData | None = None,
    math_reuse_stats_path: Path | None = None,
    math_reuse_stats: dict[str, Any] | None = None,
) -> str:
    mu_self_main = metric_summary(musique_self.observations)
    mu_mas_main = metric_summary(musique_mas.observations)
    musique_turn_audit = musique_mas.turn_correctness_audit
    if musique_turn_audit is None:
        raise ValueError("MuSiQue MAS is missing per-turn correctness replay")
    if musique_turn_audit.parse_failures:
        raise ValueError("MuSiQue per-turn correctness replay has response parse failures")
    if musique_turn_audit.known_rows + musique_turn_audit.unknown_rows != mu_mas_main.n:
        raise ValueError("MuSiQue per-turn correctness coverage does not match valid scores")
    if (
        musique_turn_audit.correct_rows != mu_mas_main.correct_n
        or musique_turn_audit.incorrect_rows != mu_mas_main.incorrect_n
    ):
        raise ValueError("MuSiQue per-turn correctness audit and metric summary disagree")
    mu_self_reason = metric_summary(musique_self.observations, lambda item: item.reasoning_score)
    mu_mas_reason = metric_summary(musique_mas.observations, lambda item: item.reasoning_score)
    gsm_self_main = metric_summary(gsm_self.observations)
    gsm_raw_main = metric_summary(gsm_raw.observations)
    gsm_mas_main = metric_summary(gsm_mas.observations)
    gsm_selected_main = metric_summary(gsm_selected.observations)
    multipl_e_self_main = metric_summary(multipl_e_self.observations) if multipl_e_self else None
    multipl_e_mas_main = (
        metric_summary(multipl_e_mas.observations) if multipl_e_mas else None
    )
    multipl_e_self_reason = (
        metric_summary(multipl_e_self.observations, lambda item: item.reasoning_score)
        if multipl_e_self
        else None
    )
    multipl_e_mas_reason = (
        metric_summary(multipl_e_mas.observations, lambda item: item.reasoning_score)
        if multipl_e_mas
        else None
    )
    math_mas_main = metric_summary(math_mas.observations) if math_mas else None
    require_file(MATH_SELF_SMOKE_PATH, "MATH self judge smoke")
    math_self_smoke_rows = sum(1 for _ in iter_jsonl(MATH_SELF_SMOKE_PATH))

    paired_mu_main = paired_summary(
        musique_self.observations,
        musique_mas.observations,
        lambda item: item.score,
        bootstrap_samples,
        seed,
    )
    paired_mu_reason = paired_summary(
        musique_self.observations,
        musique_mas.observations,
        lambda item: item.reasoning_score,
        bootstrap_samples,
        seed + 1,
    )
    paired_gsm = paired_summary(
        gsm_self.observations,
        gsm_raw.observations,
        lambda item: item.score,
        bootstrap_samples,
        seed + 2,
    )
    paired_multipl_e = (
        paired_summary(
            multipl_e_self.observations,
            multipl_e_mas.observations,
            lambda item: item.score,
            bootstrap_samples,
            seed + 3,
        )
        if multipl_e_self and multipl_e_mas
        else None
    )
    paired_multipl_e_reason = (
        paired_summary(
            multipl_e_self.observations,
            multipl_e_mas.observations,
            lambda item: item.reasoning_score,
            bootstrap_samples,
            seed + 4,
        )
        if multipl_e_self and multipl_e_mas
        else None
    )

    outcome_mu_main = {
        outcome: outcome_paired_summary(
            musique_self.observations,
            musique_mas.observations,
            lambda item: item.score,
            outcome,
            bootstrap_samples,
            seed + 10 + int(not outcome),
        )
        for outcome in (True, False)
    }
    outcome_gsm_main = {
        outcome: outcome_paired_summary(
            gsm_self.observations,
            gsm_raw.observations,
            lambda item: item.score,
            outcome,
            bootstrap_samples,
            seed + 12 + int(not outcome),
        )
        for outcome in (True, False)
    }
    outcome_mu_reason = {
        outcome: outcome_paired_summary(
            musique_self.observations,
            musique_mas.observations,
            lambda item: item.reasoning_score,
            outcome,
            bootstrap_samples,
            seed + 20 + int(not outcome),
        )
        for outcome in (True, False)
    }
    outcome_multipl_e_main = (
        {
            outcome: outcome_paired_summary(
                multipl_e_self.observations,
                multipl_e_mas.observations,
                lambda item: item.score,
                outcome,
                bootstrap_samples,
                seed + 22 + int(not outcome),
            )
            for outcome in (True, False)
        }
        if multipl_e_self and multipl_e_mas
        else None
    )
    outcome_multipl_e_reason = (
        {
            outcome: outcome_paired_summary(
                multipl_e_self.observations,
                multipl_e_mas.observations,
                lambda item: item.reasoning_score,
                outcome,
                bootstrap_samples,
                seed + 24 + int(not outcome),
            )
            for outcome in (True, False)
        }
        if multipl_e_self and multipl_e_mas
        else None
    )
    musique_raw_audit = pool_audits["musique_raw"]
    gsm_raw_audit = pool_audits["gsm_raw"]
    multipl_e_raw_full_audit = pool_audits["multipl_e_raw_full"]
    multipl_e_raw_reroll_audit = pool_audits["multipl_e_raw_reroll"]
    multipl_e_training_audit = pool_audits["multipl_e_training"]
    math_raw_audit = pool_audits["math_raw"]
    if multipl_e_filter_stats["full"].get("require_final_pass") is not True:
        raise ValueError("MultiPL-E full cleaner did not require final pass")
    if multipl_e_filter_stats["reroll"].get("require_final_pass") is not True:
        raise ValueError("MultiPL-E reroll cleaner did not require final pass")

    lines = [
        "# Qwen3-14B 自评与 MAS 打分偏差分析",
        "",
        "本报告比较 Qwen3-14B 对自身 one-shot SAS rollout 的最终 `judge_score`，与 Qwen3-14B 对异构 MAS turn 的最终 `judge_score`。所有数字均直接读取已保存的 judge 输出，不重新调用模型。报告严格区分两层：judge 内部如何合成 `judge_score`，以及训练器实际消费的 `reward`/`train_weight`；二者不是同一个公式。",
        "",
        "## 结论先行",
        "",
        "**并非所有数据集都是 0.5 + 0.5。** 只有 MuSiQue 和 MultiPL-E 的 judge 内部最终分采用两个维度等权；GSM-Hard 与 MATH 的 compact judge 只有一个 `process_score`。而训练优化信号又是下一层：MuSiQue 为 0.6/0.4，MultiPL-E 为 0.3/0.7，GSM-Hard/MATH 使用带正确性符号、0.35/0.65 幅度和 transition/c2c 修正的 signed reward。",
        "",
        "## 评分池筛选审计",
        "",
        "本报告在每个选定评分文件内部不再按正确性筛记录：凡是已保存且可解析的 14B 分数均纳入。评分文件在上游是否经过 trajectory 选择仍单独审计；因此 MuSiQue、GSM-Hard、MATH 是完整 rollout 池，MultiPL-E 则回答“实际 RL 训练入口中的评分均值”，外推到全量 rollout 时必须保留选择偏差 caveat。",
        "",
        "| 数据集 | MAS 评分池审计 | 是否按正确性筛选 | 主分析处理 |",
        "|---|---|---|---|",
        f"| MuSiQue | 14B 重打分输入 {musique_raw_audit.rows} turn / {musique_raw_audit.trajectories} trajectory；最终答对 {musique_raw_audit.correct_trajectories}、答错 {musique_raw_audit.incorrect_trajectories}；其中 {len(musique_mas.observations)} turn 获得有效 14B 分数 | 否；仅排除 {musique_mas.rejected_records} 个 judge 解析失败 turn | 纳入主分析；正确性分层使用离线逐 turn 回放 |",
        f"| GSM-Hard | 完整 14B 评分池 {gsm_raw_main.n} turn / {gsm_raw_audit.trajectories} trajectory；最终答对 {gsm_raw_audit.correct_trajectories}、答错 {gsm_raw_audit.incorrect_trajectories} | 否；完整覆盖 1187 题 × 8 rollout | 纳入主分析；训练 eligible pool 另列 |",
        f"| MultiPL-E-8Lang | 实际 RL 入口 {multipl_e_mas_main.n if multipl_e_mas_main else 'n/a'} 个已评分 turn / {multipl_e_training_audit.trajectories} 条 trajectory；最终均成功，但 turn 当前答案含 {multipl_e_mas_main.correct_n if multipl_e_mas_main else 'n/a'} 对、{multipl_e_mas_main.incorrect_n if multipl_e_mas_main else 'n/a'} 错 | **上游按最终成功筛 trajectory**；分析未丢弃该文件中的任何有效 turn | 纳入实际训练评分池的均值与配对；明确标注选择条件 |",
        f"| MATH | 完整 14B 评分池 {math_mas_main.n if math_mas_main else 'n/a'} turn / {math_raw_audit.trajectories} trajectory；最终答对 {math_raw_audit.correct_trajectories}、答错 {math_raw_audit.incorrect_trajectories} | 否；完整覆盖 7500 题 × 8 rollout | MAS 均值可用；缺少完整 self judge，配对记为 n/a |",
        "",
        f"MuSiQue、GSM-Hard 和 MATH 的 MAS 池都覆盖最终正确与最终错误 trajectory。用于同对/同错的 MuSiQue、GSM-Hard、MultiPL-E 正确性现已统一为被评分 turn 当时的当前答案；MultiPL-E 的 trajectory 虽然最终均成功，仍保留 {multipl_e_mas_main.incorrect_n if multipl_e_mas_main else 'n/a'} 个错误中间 turn。MATH 只有 MAS 单侧正确性，因缺完整 self judge，双侧配对仍为 n/a。",
        "",
        "## MuSiQue 逐 turn 正确性回放审计",
        "",
        f"MuSiQue 原日志只直接保存 trajectory 最终 `em`。本报告从每个 response 中依次提取 `tentative_answer` / `confirmed_answer`，空 tentative 沿用链路中的上一非空候选，再用 `{MUSIQUE_DATA_DIR}` 的 gold answer 与项目官方 `src.grader.compute_em_f1` 重算当前候选 EM。首个候选出现前的 turn 不强行标错，而是记为不可判定。",
        "",
        "| 原始 turn | 有效 14B 评分 | 当前答案可判定 | 当前答案正确 | 当前答案错误 | 尚无候选 | response 解析失败 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {musique_turn_audit.source_rows} | {musique_turn_audit.valid_score_rows} | {musique_turn_audit.known_rows} | {musique_turn_audit.correct_rows} | {musique_turn_audit.incorrect_rows} | {musique_turn_audit.unknown_rows} | {musique_turn_audit.parse_failures} |",
        "",
        f"状态回放在全部源 turn 中识别到 {musique_turn_audit.new_candidate_rows} 次非空候选更新、{musique_turn_audit.carried_candidate_rows} 次沿用上一候选和 {musique_turn_audit.confirmed_rows} 次确认答案。{musique_turn_audit.unknown_rows} 个有效评分 turn 因当时尚无候选，只进入总体均值，不进入同对/同错分层。",
        "",
        "## 最终 judge 分定义与核验",
        "",
        "| 数据集 | 被评分对象 | 主读取字段 | 最终 judge 分公式 | 全量一致性核验 | 说明 |",
        "|---|---|---|---|---|---|",
        audit_row(musique_self),
        audit_row(musique_mas),
        audit_row(gsm_self),
        audit_row(gsm_raw),
        audit_row(multipl_e_self),
        audit_row(multipl_e_mas),
        audit_row(math_mas),
        "",
        "GSM-Hard self 的 `outcome_score` 以及 MATH/GSM MAS 的 deterministic correctness 都来自规则 evaluator，不是第二个 LLM judge 子分。",
        "",
        "## 训练实际消费的优化信号",
        "",
        "| 数据集 / 训练 | judge 内部最终分 | 训练 reward / advantage | trainer 实际字段 |",
        "|---|---|---|---|",
        "| MuSiQue JCA | `0.5 × reasoning + 0.5 × action` | `0.6 × task_reward + 0.4 × judge_score` | `reward` |",
        "| MuSiQue 14B self-RL | `0.5 × reasoning + 0.5 × finalization` | `0.6 × task_reward + 0.4 × judge_score` | `reward` |",
        "| GSM-Hard JCA（71.97% v13） | `judge_score = process_score` | `sign(correctness) × clip(0.35 + 0.65 × aligned_judge, 0.35, 1)`；错↔对 transition 乘 1.25 后再截断，A2 的 correct→correct 再乘 0.5 | `reward` |",
        "| GSM-Hard 14B self-RL | `judge_score = process_score` | `0.6 × outcome_score + 0.4 × process_score` | `reward` |",
        "| MultiPL-E JCA 候选池 / 14B self-RL | `0.5 × reasoning + 0.5 × action/finalization` | `0.3 × task_reward + 0.7 × judge_score` | `reward` |",
        "| MATH legacy v13 | `judge_score = process_score` | GSM-style signed reward：0.35/0.65、transition 1.25、A2 correct→correct 0.5 | `reward` |",
        "| MATH 新 pipeline | `judge_score = process_score` | prepare 后的 role-aware signed AWR 权重 | `train_weight` |",
        "",
        "所以用户问的“最终 judge 分”应读取上表第二列；如果问“最终训练到底优化了什么”，则必须看第三、四列，不能把 `0.5+0.5` 当作通用训练 reward。",
        "",
        "## 最终产物链路对齐审计",
        "",
        "| 数据集 | 正式 eval → adapter | adapter → 训练输入 | 实际字段 | 对齐结论 |",
        "|---|---|---|---|---|",
        f"| MuSiQue | `{MUSIQUE_FINAL_EVAL_PATH}`：{musique_eval.correct}/{musique_eval.total} = {musique_eval.score:.2%}；A1/A2=`{MUSIQUE_FINAL_ADAPTERS[0].parent.parent}`，A3=`{MUSIQUE_FINAL_ADAPTERS[2].parent.parent}` | stage 1=`{musique_final_train_path}`；A3 stage 2=`{musique_final_a3_train_path}` | 两阶段均为 `reward`，且 0.6/0.4 公式核验分别 {musique_final_audit.formula_checked - musique_final_audit.formula_mismatches}/{musique_final_audit.formula_checked}、{musique_final_a3_audit.formula_checked - musique_final_a3_audit.formula_mismatches}/{musique_final_a3_audit.formula_checked} 通过 | **adapter/eval 对齐；14B judge provenance 仅部分对齐**（每阶段仅 {musique_final_audit.explicit_qwen14b}/{musique_final_audit.total} 行显式 14B） |",
        f"| GSM-Hard | `{GSM_FINAL_EVAL_PATH}`：{gsm_eval.correct}/{gsm_eval.total} = {gsm_eval.score:.2%}；adapter=`{GSM_FINAL_ADAPTER_ROOT}` | `{gsm_selected.path}`，来自 `{gsm_mas.path}`；选择子集 {gsm_training_alignment.matched_to_source}/{gsm_training_alignment.selected} 行可回连，v3/v4 judge 字段不一致 {gsm_training_alignment.source_v3_v4_mismatches} 行，源→子集 judge 分不一致 {gsm_training_alignment.judge_score_mismatches} 行 | `reward`；与 `train_weight` 不一致 {gsm_training_alignment.reward_field_mismatches} 行 | **完全对齐 71.97% 最终产物** |",
        # 2026-09-12 前此行原文逐字为：
        #   f"| MultiPL-E-8Lang | 尚无正式最终 JCA eval | `{MULTIPL_E_FINAL_TRAIN_LOG}` 明确将 `{multipl_e_mas.path if multipl_e_mas else '—'}` 作为 A1/A2/A3 RL rollout 输入 | `reward`（0.3/0.7） | **评分池与实际训练入口对齐；正式最终 JCA eval 尚缺** |",
        f"| MultiPL-E-8Lang | 正式 eval 已补齐：`{MULTIPL_E_REGISTERED_EVAL}`：{MULTIPL_E_REGISTERED_EVAL_SCORE}（best-of-10 roll）；adapter 支线为 {MULTIPL_E_REGISTERED_ADAPTER_LINEAGE} | `{MULTIPL_E_FINAL_TRAIN_LOG}` 明确将 `{multipl_e_mas.path if multipl_e_mas else '—'}` 作为 A1/A2/A3 RL rollout 输入 | `reward`（0.3/0.7） | **评分池与 0813 ckpt-search 训练入口对齐；但已登记的 r08 eval 其 adapter 来自 gpt5_paired_rwr 支线，与本行评分池不同源，故该对齐结论不覆盖 r08** |",
        f"| MATH | 新 pipeline 的 A1/A2/A3 adapter 均已落盘于 `{MATH_FINAL_RUN_ROOT / '05_train/adapters'}`；正式 `06_eval` 为 `{MATH_FINAL_RUN_ROOT / '06_eval/variants/rl_all_final_a3_turn2_incumbent_v1/results.jsonl'}`，特殊三-turn 预算投影后 EM 为 `392/500 = 78.40%` | judge 分复用 `{math_mas.path if math_mas else '—'}`，prepare 后进入 `{MATH_FINAL_RUN_ROOT / '04_prepare/awr/train.jsonl'}` | `train_weight` | **训练、正式 eval 均已完成；超过三 turn 或第三 turn 后仍未成功 stop 时采用首个 `tentative_answer`** |",
        "",
        "## 四数据集 14B→MAS 评分池均值汇总",
        "",
        "下表按所选文件中全部有效 turn 级评分记录直接平均；分析阶段不按正确性删样本，judge 解析失败且没有数值分的记录无法纳入。MuSiQue、GSM-Hard、MATH 使用完整 rollout 评分池；MultiPL-E 使用最终训练实际读取的 per-turn 池，其上游只保留最终成功 trajectory。四个数据集的 judge 字段与 rubric 不完全相同，因此适合做各自数据集内分析，不宜把绝对值直接横向当作同一量尺。",
        "",
        "| 数据集 | 14B→MAS 最终指标 | 有效评分记录 | 平均分 |",
        "|---|---|---:|---:|",
        f"| MuSiQue | `judge_score` | {mu_mas_main.n} | {mu_mas_main.mean:.4f} |",
        f"| GSM-Hard | `collaboration_judge_v4.judge_score` | {gsm_raw_main.n} | {gsm_raw_main.mean:.4f} |",
        f"| MultiPL-E-8Lang | `judge_score` | {multipl_e_mas_main.n if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_main.mean) if multipl_e_mas_main else 'n/a'} |",
        f"| MATH | `collaboration_judge_v4.judge_score` | {math_mas_main.n if math_mas_main else 'n/a'} | {fmt(math_mas_main.mean) if math_mas_main else 'n/a'} |",
        "",
        "## 结论摘要",
        "",
        f"- MuSiQue 的总体平均最终 `judge_score`：14B 自评为 **{mu_self_main.mean:.4f}**，评价 MAS 为 **{mu_mas_main.mean:.4f}**。",
        f"- GSM-Hard 的总体平均最终 `judge_score`：14B 自评为 **{gsm_self_main.mean:.4f}**，评价完整 MAS 池为 **{gsm_raw_main.mean:.4f}**；两边日志中该字段都与各自 `process_score` 全量相等。",
        f"- MultiPL-E-8Lang 的完整 self 池均值为 **{multipl_e_self_main.mean:.4f}**，实际 RL 入口的 MAS turn 均值为 **{multipl_e_mas_main.mean:.4f}**，直接 pooled 差为 **{multipl_e_self_main.mean - multipl_e_mas_main.mean:+.4f}**；MAS 池上游只保留最终成功 trajectory，不能把该差值外推成全量 rollout 的无条件估计。" if multipl_e_self_main and multipl_e_mas_main else "- MultiPL-E-8Lang：缺少 self 或 MAS judge 统计。",
        f"- MATH 的完整 MAS 最终 `judge_score` 平均为 **{math_mas_main.mean:.4f}**（{math_mas_main.n} 条 turn）；compact MATH schema 中它与 `process_score` 全量相等。正式 14B self-RL judge 尚未找到，因此不填 self 均值。" if math_mas_main else "- MATH：缺少完整 MAS judge 统计。",
        f"- 更关键的是错误回答：MuSiQue、GSM-Hard、MultiPL-E 的 self 错误项正分率分别为 **{pct(mu_self_main.incorrect_positive_rate)}**、**{pct(gsm_self_main.incorrect_positive_rate)}**、**{pct(multipl_e_self_main.incorrect_positive_rate) if multipl_e_self_main else 'n/a'}**；对应 MAS 错误项分别为 **{pct(mu_mas_main.incorrect_positive_rate)}**、**{pct(gsm_raw_main.incorrect_positive_rate)}**、**{pct(multipl_e_mas_main.incorrect_positive_rate) if multipl_e_mas_main else 'n/a'}**。",
        f"- 控制题目与被评分答案正确性后，双方均错时的题目级配对差为：MuSiQue **{outcome_mu_main[False].difference:+.4f}**、GSM-Hard **{outcome_gsm_main[False].difference:+.4f}**、MultiPL-E **{outcome_multipl_e_main[False].difference:+.4f}**；MultiPL-E 的具体显著性以对应 bootstrap CI 为准。MuSiQue 与 MultiPL-E 的共同 `reasoning_score` 差分别为 **{outcome_mu_reason[False].difference:+.4f}**、**{outcome_multipl_e_reason[False].difference:+.4f}**。" if outcome_multipl_e_main and outcome_multipl_e_reason else "- MultiPL-E 配对统计不可用。",
        "- 因此，现有日志强烈支持“14B 自评更宽松、存在与 self-evaluation overconfidence 一致的校准偏差”。但 self 与 MAS 的回答结构、policy 和 judge rubric 并不完全相同，不能仅凭这组观察性比较断言已经证明了纯粹的模型身份偏置。",
        "",
        "## 总体平均分",
        "",
        "这里的 pooled mean 对所选评分文件中的全部有效 judge 分直接平均；MAS 的一条记录是一个 turn，self 的一条记录是一条 one-shot trajectory。分析阶段不按正确/错误删记录。MultiPL-E 的上游 trajectory 选择条件在前文单列。",
        "",
        "| 数据集 | 指标 | 14B 自评记录数 | 14B 自评均值 | 14B→MAS 记录数 | 14B→MAS 均值 | 均值差（self−MAS） |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| MuSiQue | `judge_score` | {mu_self_main.n} | {mu_self_main.mean:.4f} | {mu_mas_main.n} | {mu_mas_main.mean:.4f} | {mu_self_main.mean - mu_mas_main.mean:+.4f} |",
        f"| GSM-Hard | `judge_score` | {gsm_self_main.n} | {gsm_self_main.mean:.4f} | {gsm_raw_main.n} | {gsm_raw_main.mean:.4f} | {gsm_self_main.mean - gsm_raw_main.mean:+.4f} |",
        f"| MultiPL-E-8Lang | `judge_score` | {multipl_e_self_main.n if multipl_e_self_main else 'n/a'} | {fmt(multipl_e_self_main.mean) if multipl_e_self_main else 'n/a'} | {multipl_e_mas_main.n if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_main.mean) if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_self_main.mean - multipl_e_mas_main.mean) if multipl_e_self_main and multipl_e_mas_main else 'n/a'} |",
        f"| MATH | `judge_score` | n/a | n/a | {math_mas_main.n if math_mas_main else 'n/a'} | {fmt(math_mas_main.mean) if math_mas_main else 'n/a'} | n/a |",
        "",
        "MuSiQue self 的 `judge_score = 0.5 × reasoning_score + 0.5 × finalization_score`；MAS 的 `judge_score = 0.5 × reasoning_score + 0.5 × action_score`。MultiPL-E 也采用相同的 0.5/0.5 结构，但 self 的第二维叫 `code_finalization_score`。GSM-Hard 与 MATH 的 compact schema 只有一个 LLM judge 维度，因此 `judge_score = process_score`，不是漏算了第二个 judge 子分。",
        "",
        "## 按 trajectory 平均",
        "",
        "先在每条 `(problem_id, rollout_idx)` trajectory 内平均所有 turn，再对 trajectory 等权平均，避免较长 MAS 链路获得更高权重。",
        "",
        "| 数据集 | 指标 | 14B 自评 trajectory 数 | 14B 自评均值 | MAS trajectory 数 | 14B→MAS 均值 |",
        "|---|---|---:|---:|---:|---:|",
        f"| MuSiQue | `judge_score` | {mu_self_main.trajectories} | {mu_self_main.trajectory_mean:.4f} | {mu_mas_main.trajectories} | {mu_mas_main.trajectory_mean:.4f} |",
        f"| GSM-Hard | `judge_score` | {gsm_self_main.trajectories} | {gsm_self_main.trajectory_mean:.4f} | {gsm_raw_main.trajectories} | {gsm_raw_main.trajectory_mean:.4f} |",
        f"| MultiPL-E-8Lang | `judge_score` | {multipl_e_self_main.trajectories if multipl_e_self_main else 'n/a'} | {fmt(multipl_e_self_main.trajectory_mean) if multipl_e_self_main else 'n/a'} | {multipl_e_mas_main.trajectories if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_main.trajectory_mean) if multipl_e_mas_main else 'n/a'} |",
        f"| MATH | `judge_score` | n/a | n/a | {math_mas_main.trajectories if math_mas_main else 'n/a'} | {fmt(math_mas_main.trajectory_mean) if math_mas_main else 'n/a'} |",
        "",
        "## 共同题目配对比较",
        "",
        "对 self 和 MAS 共同覆盖的每道题，先分别计算该题的平均分，再计算题目级配对差值。95% 置信区间使用题目级 bootstrap。这个设计控制题目集合差异，但没有匹配同一条回答。",
        "",
        "| 数据集 | 指标 | 共同题数 | 14B 自评均值 | 14B→MAS 均值 | 配对差 | 95% bootstrap CI | self 更高的题目比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| MuSiQue | `judge_score` | {paired_mu_main.common_problems} | {paired_mu_main.self_mean:.4f} | {paired_mu_main.mas_mean:.4f} | {paired_mu_main.difference:+.4f} | [{paired_mu_main.ci_low:.4f}, {paired_mu_main.ci_high:.4f}] | {pct(paired_mu_main.self_greater_rate)} |",
        f"| GSM-Hard | `judge_score` | {paired_gsm.common_problems} | {paired_gsm.self_mean:.4f} | {paired_gsm.mas_mean:.4f} | {paired_gsm.difference:+.4f} | [{paired_gsm.ci_low:.4f}, {paired_gsm.ci_high:.4f}] | {pct(paired_gsm.self_greater_rate)} |",
        f"| MultiPL-E-8Lang | `judge_score` | {paired_multipl_e.common_problems} | {paired_multipl_e.self_mean:.4f} | {paired_multipl_e.mas_mean:.4f} | {paired_multipl_e.difference:+.4f} | [{paired_multipl_e.ci_low:.4f}, {paired_multipl_e.ci_high:.4f}] | {pct(paired_multipl_e.self_greater_rate)} |" if paired_multipl_e else "| MultiPL-E-8Lang | `judge_score` | n/a | n/a | n/a | n/a | n/a | n/a |",
        "| MATH | `judge_score` | n/a | n/a | n/a | n/a | n/a | n/a |",
        "",
        f"Bootstrap 参数：`samples={bootstrap_samples}`，`seed={seed}`（最终 judge 主表依次使用 MuSiQue seed、GSM-Hard seed+2、MultiPL-E seed+3）。MultiPL-E 结果以最终成功 trajectory 的实际训练池为条件。",
        "",
        "## 相同答案正确性下的配对比较",
        "",
        "这一节进一步控制被 14B 打分的那份答案是否正确。对每道共同题，分别汇总 self 与 MAS 中正确（或错误）的评分记录，再比较双方题目级均值；95% 置信区间对题目级差值做 bootstrap。这样不会把 MultiPL-E 链内先错后对的错误 turn 因最终成功而抹掉。",
        "",
        "正确性字段统一指被评分 turn 当时的当前答案：MuSiQue 从 response 回放最新非空候选并用官方 `compute_em_f1` 离线判定，GSM-Hard MAS 读取 `deterministic_state_v3.current_answer_correct`，MultiPL-E 读取逐 turn `passed`；self 一侧均为 one-shot 结果。因此本节的“同对/同错”不是要求 MAS 整条 trajectory 最终失败。",
        "",
        "| 数据集 | 指标 | 配对条件 | 共同题数 | self 评分记录 | MAS 评分 turn | 14B 自评均值 | 14B→MAS 均值 | 配对差 | 95% bootstrap CI | self 更高的题目比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        outcome_paired_row("MuSiQue", "judge_score", "双方被评分答案均正确", outcome_mu_main[True]),
        outcome_paired_row("MuSiQue", "judge_score", "双方被评分答案均错误", outcome_mu_main[False]),
        outcome_paired_row("GSM-Hard", "process_score", "双方被评分答案均正确", outcome_gsm_main[True]),
        outcome_paired_row("GSM-Hard", "process_score", "双方被评分答案均错误", outcome_gsm_main[False]),
        outcome_paired_row("MultiPL-E-8Lang", "judge_score", "双方被评分答案均正确", outcome_multipl_e_main[True] if outcome_multipl_e_main else None),
        outcome_paired_row("MultiPL-E-8Lang", "judge_score", "双方被评分答案均错误", outcome_multipl_e_main[False] if outcome_multipl_e_main else None),
        outcome_paired_row("MATH", "judge_score", "双方被评分答案均正确", None),
        outcome_paired_row("MATH", "judge_score", "双方被评分答案均错误", None),
        "",
        "MuSiQue、GSM-Hard 与 MultiPL-E 都能在双方被评分答案均错误时进行配对；MultiPL-E 的样本来自最终成功 trajectory 内的错误中间 turn，结论只适用于该训练池。MATH 只有完整 MAS judge，没有完整 self judge，因此双侧同对/同错仍为 n/a；其 MAS 单侧正确/错误均值见后文。",
        "",
        "### 同正确性下的共同 reasoning 子分",
        "",
        "MuSiQue 与 MultiPL-E 的 self/MAS 都显式保存同名 `reasoning_score`，因此再按同一答案正确性口径比较。GSM-Hard 与 MATH compact judge 没有独立 reasoning 子分。",
        "",
        "| 数据集 | 指标 | 配对条件 | 共同题数 | self 评分记录 | MAS 评分 turn | 14B 自评均值 | 14B→MAS 均值 | 配对差 | 95% bootstrap CI | self 更高的题目比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        outcome_paired_row("MuSiQue", "reasoning_score", "双方被评分答案均正确", outcome_mu_reason[True]),
        outcome_paired_row("MuSiQue", "reasoning_score", "双方被评分答案均错误", outcome_mu_reason[False]),
        outcome_paired_row("MultiPL-E-8Lang", "reasoning_score", "双方被评分答案均正确", outcome_multipl_e_reason[True] if outcome_multipl_e_reason else None),
        outcome_paired_row("MultiPL-E-8Lang", "reasoning_score", "双方被评分答案均错误", outcome_multipl_e_reason[False] if outcome_multipl_e_reason else None),
        "",
        "本节仍未匹配完全相同的回答文本，也没有控制错误答案之间的细粒度质量差异；它比总体均值更直接地排除了二值正确率差异，但仍属于观察性证据。最严格的身份偏置检验仍需匿名化后的同 prompt、同 rubric blind rejudge。",
        "",
        "## 错误项上的评分",
        "",
        "正分定义为 `score > 0`，满分定义为 `score = 1`。三套 MAS 正确性均对应被评分 turn 的当前答案：MuSiQue 为离线候选回放 EM，GSM-Hard 为 `deterministic_state_v3.current_answer_correct`，MultiPL-E 为逐 turn `passed`。",
        "",
        "| 数据集 | 指标 | 被评分对象 | 错误项数 | 错误项平均分 | 错误项正分率 | 错误项满分率 |",
        "|---|---|---|---:|---:|---:|---:|",
        f"| MuSiQue | `judge_score` | 14B 自己 | {mu_self_main.incorrect_n} | {mu_self_main.incorrect_mean:.4f} | {pct(mu_self_main.incorrect_positive_rate)} | {pct(mu_self_main.incorrect_full_score_rate)} |",
        f"| MuSiQue | `judge_score` | 异构 MAS | {mu_mas_main.incorrect_n} | {mu_mas_main.incorrect_mean:.4f} | {pct(mu_mas_main.incorrect_positive_rate)} | {pct(mu_mas_main.incorrect_full_score_rate)} |",
        f"| GSM-Hard | `judge_score` | 14B 自己 | {gsm_self_main.incorrect_n} | {gsm_self_main.incorrect_mean:.4f} | {pct(gsm_self_main.incorrect_positive_rate)} | {pct(gsm_self_main.incorrect_full_score_rate)} |",
        f"| GSM-Hard | `judge_score` | 异构 MAS | {gsm_raw_main.incorrect_n} | {gsm_raw_main.incorrect_mean:.4f} | {pct(gsm_raw_main.incorrect_positive_rate)} | {pct(gsm_raw_main.incorrect_full_score_rate)} |",
        f"| MultiPL-E-8Lang | `judge_score` | 14B 自己 | {multipl_e_self_main.incorrect_n if multipl_e_self_main else 'n/a'} | {fmt(multipl_e_self_main.incorrect_mean) if multipl_e_self_main else 'n/a'} | {pct(multipl_e_self_main.incorrect_positive_rate) if multipl_e_self_main else 'n/a'} | {pct(multipl_e_self_main.incorrect_full_score_rate) if multipl_e_self_main else 'n/a'} |",
        f"| MultiPL-E-8Lang | `judge_score` | 异构 MAS | {multipl_e_mas_main.incorrect_n if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_main.incorrect_mean) if multipl_e_mas_main else 'n/a'} | {pct(multipl_e_mas_main.incorrect_positive_rate) if multipl_e_mas_main else 'n/a'} | {pct(multipl_e_mas_main.incorrect_full_score_rate) if multipl_e_mas_main else 'n/a'} |",
        f"| MATH | `judge_score` | 异构 MAS | {math_mas_main.incorrect_n if math_mas_main else 'n/a'} | {fmt(math_mas_main.incorrect_mean) if math_mas_main else 'n/a'} | {pct(math_mas_main.incorrect_positive_rate) if math_mas_main else 'n/a'} | {pct(math_mas_main.incorrect_full_score_rate) if math_mas_main else 'n/a'} |",
        "",
        "错误回答仍被打高分，是 overconfidence 假设最直接的证据之一。比较时应优先看上一节的共同题、同正确性配对差，而不是把不同题目集合上的单侧均值直接相减。",
        "",
        "## 正确项与总体分布",
        "",
        "| 数据集 | 指标 | 被评分对象 | 正确项数 | 正确项均值 | 总体正分率 | 总体满分率 |",
        "|---|---|---|---:|---:|---:|---:|",
        f"| MuSiQue | `judge_score` | 14B 自己 | {mu_self_main.correct_n} | {mu_self_main.correct_mean:.4f} | {pct(mu_self_main.positive_rate)} | {pct(mu_self_main.full_score_rate)} |",
        f"| MuSiQue | `judge_score` | 异构 MAS | {mu_mas_main.correct_n} | {mu_mas_main.correct_mean:.4f} | {pct(mu_mas_main.positive_rate)} | {pct(mu_mas_main.full_score_rate)} |",
        f"| GSM-Hard | `judge_score` | 14B 自己 | {gsm_self_main.correct_n} | {gsm_self_main.correct_mean:.4f} | {pct(gsm_self_main.positive_rate)} | {pct(gsm_self_main.full_score_rate)} |",
        f"| GSM-Hard | `judge_score` | 异构 MAS | {gsm_raw_main.correct_n} | {gsm_raw_main.correct_mean:.4f} | {pct(gsm_raw_main.positive_rate)} | {pct(gsm_raw_main.full_score_rate)} |",
        f"| MultiPL-E-8Lang | `judge_score` | 14B 自己 | {multipl_e_self_main.correct_n if multipl_e_self_main else 'n/a'} | {fmt(multipl_e_self_main.correct_mean) if multipl_e_self_main else 'n/a'} | {pct(multipl_e_self_main.positive_rate) if multipl_e_self_main else 'n/a'} | {pct(multipl_e_self_main.full_score_rate) if multipl_e_self_main else 'n/a'} |",
        f"| MultiPL-E-8Lang | `judge_score` | 异构 MAS | {multipl_e_mas_main.correct_n if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_main.correct_mean) if multipl_e_mas_main else 'n/a'} | {pct(multipl_e_mas_main.positive_rate) if multipl_e_mas_main else 'n/a'} | {pct(multipl_e_mas_main.full_score_rate) if multipl_e_mas_main else 'n/a'} |",
        f"| MATH | `judge_score` | 异构 MAS | {math_mas_main.correct_n if math_mas_main else 'n/a'} | {fmt(math_mas_main.correct_mean) if math_mas_main else 'n/a'} | {pct(math_mas_main.positive_rate) if math_mas_main else 'n/a'} | {pct(math_mas_main.full_score_rate) if math_mas_main else 'n/a'} |",
        "",
        "## 共同 reasoning 子分（稳健性补充）",
        "",
        "这一节不是主结果，只检查 self 与 MAS 都显式提供 `reasoning_score` 时差异是否仍存在。MuSiQue 使用完整 MAS 池；MultiPL-E 使用实际训练 per-turn 池，并保留其上游最终成功筛选 caveat。judge prompt 仍不完全同构。",
        "",
        "| 数据集 | self reasoning 记录均值 | MAS reasoning 记录均值 | 共同题数 | 配对 self 均值 | 配对 MAS 均值 | 配对差（self−MAS） |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| MuSiQue | {mu_self_reason.mean:.4f} | {mu_mas_reason.mean:.4f} | {paired_mu_reason.common_problems} | {paired_mu_reason.self_mean:.4f} | {paired_mu_reason.mas_mean:.4f} | {paired_mu_reason.difference:+.4f} |",
        f"| MultiPL-E-8Lang | {fmt(multipl_e_self_reason.mean) if multipl_e_self_reason else 'n/a'} | {fmt(multipl_e_mas_reason.mean) if multipl_e_mas_reason else 'n/a'} | {paired_multipl_e_reason.common_problems if paired_multipl_e_reason else 'n/a'} | {fmt(paired_multipl_e_reason.self_mean) if paired_multipl_e_reason else 'n/a'} | {fmt(paired_multipl_e_reason.mas_mean) if paired_multipl_e_reason else 'n/a'} | {fmt(paired_multipl_e_reason.difference) if paired_multipl_e_reason else 'n/a'} |",
        "",
        f"Reasoning 配对 bootstrap 使用 MuSiQue `seed={seed + 1}`、MultiPL-E `seed={seed + 4}`。GSM-Hard/MATH compact judge 没有独立 reasoning 子分。",
        "",
        "## GSM-Hard 最终训练选择子集（附加）",
        "",
        "GSM-Hard v13 先对原始 14B 重打分池排除 269 个“8 次 rollout 全部失败”的问题，再做类别重采样得到最终训练文件。三种均值回答不同问题，不能互相替代。",
        "",
        "| 数据来源 | turn 数 | trajectory 数 | 平均最终 `judge_score` | 正分率 | 满分率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 原始 14B 重打分池（未过滤） | {gsm_raw_main.n} | {gsm_raw_main.trajectories} | {gsm_raw_main.mean:.4f} | {pct(gsm_raw_main.positive_rate)} | {pct(gsm_raw_main.full_score_rate)} |",
        f"| 最终训练 eligible pool（排除全失败题） | {gsm_mas_main.n} | {gsm_mas_main.trajectories} | {gsm_mas_main.mean:.4f} | {pct(gsm_mas_main.positive_rate)} | {pct(gsm_mas_main.full_score_rate)} |",
        f"| 最终训练选择子集 | {gsm_selected_main.n} | {gsm_selected_main.trajectories} | {gsm_selected_main.mean:.4f} | {pct(gsm_selected_main.positive_rate)} | {pct(gsm_selected_main.full_score_rate)} |",
        "",
        "## MultiPL-E 最终成功 trajectory 的逐 turn 评分池",
        "",
        f"上游两个 cleaner 都明确设置 `require_final_pass=true`：full pool 从 {multipl_e_filter_stats['full'].get('input_groups', 'n/a')} 条 trajectory 保留 {multipl_e_filter_stats['full'].get('kept_groups', 'n/a')} 条，reroll pool 从 {multipl_e_filter_stats['reroll'].get('input_groups', 'n/a')} 条保留 {multipl_e_filter_stats['reroll'].get('kept_groups', 'n/a')} 条。最终实际 RL 入口中的 {multipl_e_training_audit.trajectories} 条 trajectory 均成功，但训练文件保留了链内所有 {multipl_e_mas_main.n if multipl_e_mas_main else 'n/a'} 个有效评分 turn，其中 {multipl_e_mas_main.incorrect_n if multipl_e_mas_main else 'n/a'} 个 turn 当前答案错误。本报告没有人为删掉这些错误 turn。",
        "",
        "| 数据来源 | turn 数 | trajectory 数 | 最终成功 | 平均 `judge_score` | 平均 `reasoning_score` |",
        "|---|---:|---:|---:|---:|---:|",
        f"| MultiPL-E 最终训练 per-turn pool | {multipl_e_mas_main.n if multipl_e_mas_main else 'n/a'} | {multipl_e_mas_main.trajectories if multipl_e_mas_main else 'n/a'} | {multipl_e_training_audit.correct_trajectories}/{multipl_e_training_audit.trajectories} | {fmt(multipl_e_mas_main.mean) if multipl_e_mas_main else 'n/a'} | {fmt(multipl_e_mas_reason.mean) if multipl_e_mas_reason else 'n/a'} |",
        "",
        "## 日志来源与 provenance",
        "",
        "| 数据集 | 对象 | 输入文件 | 原始记录 | 有效 14B 评分 | 排除记录 |",
        "|---|---|---|---:|---:|---:|",
        f"| MuSiQue | 14B 自己 | `{musique_self.path}` | {musique_self.total_records} | {len(musique_self.observations)} | {musique_self.rejected_records} |",
        f"| MuSiQue | 异构 MAS | `{musique_mas.path}` | {musique_mas.total_records} | {len(musique_mas.observations)} | {musique_mas.rejected_records} |",
        f"| GSM-Hard | 14B 自己 | `{gsm_self.path}` | {gsm_self.total_records} | {len(gsm_self.observations)} | {gsm_self.rejected_records} |",
        f"| GSM-Hard | 异构 MAS 原始池 | `{gsm_raw.path}` | {gsm_raw.total_records} | {len(gsm_raw.observations)} | {gsm_raw.rejected_records} |",
        f"| GSM-Hard | 异构 MAS eligible pool | `{gsm_mas.path}` | {gsm_mas.total_records} | {len(gsm_mas.observations)} | {gsm_mas.rejected_records} |",
        f"| GSM-Hard | 异构 MAS 训练子集 | `{gsm_selected.path}` | {gsm_selected.total_records} | {len(gsm_selected.observations)} | {gsm_selected.rejected_records} |",
        f"| MultiPL-E-8Lang | 14B 自己 | `{multipl_e_self.path if multipl_e_self else '—'}` | {multipl_e_self.total_records if multipl_e_self else 'n/a'} | {len(multipl_e_self.observations) if multipl_e_self else 'n/a'} | {multipl_e_self.rejected_records if multipl_e_self else 'n/a'} |",
        f"| MultiPL-E-8Lang | 异构 MAS raw full（无 14B judge） | `{MULTIPL_E_RAW_FULL_PATH}` | {multipl_e_raw_full_audit.rows} | n/a | n/a |",
        f"| MultiPL-E-8Lang | 异构 MAS raw reroll（无 14B judge） | `{MULTIPL_E_RAW_REROLL_PATH}` | {multipl_e_raw_reroll_audit.rows} | n/a | n/a |",
        f"| MultiPL-E-8Lang | 异构 MAS 最终训练 per-turn pool | `{multipl_e_mas.path if multipl_e_mas else '—'}` | {multipl_e_mas.total_records if multipl_e_mas else 'n/a'} | {len(multipl_e_mas.observations) if multipl_e_mas else 'n/a'} | {multipl_e_mas.rejected_records if multipl_e_mas else 'n/a'} |",
        f"| MATH | 异构 MAS 完整池 | `{math_mas.path if math_mas else '—'}` | {math_mas.total_records if math_mas else 'n/a'} | {len(math_mas.observations) if math_mas else 'n/a'} | {math_mas.rejected_records if math_mas else 'n/a'} |",
        "",
        f"MuSiQue 的全量 14B MAS 重打分命令保存在 `{ROOT / 'logs/rl_rescore/rl_rescore_qwen14b_0715/command.txt'}`。逐 turn 正确性复用 `{MUSIQUE_DATA_DIR}`、`src.data.load_musique` 与 `src.grader.compute_em_f1`，不调用模型。但最终 adapter 的 stage-1 训练文件 `{musique_final_train_path}` 是混合 provenance：共 {musique_final_audit.total} 个 turn，只有 {musique_final_audit.explicit_qwen14b} 个显式标记 `judge_model=qwen14b_judge`，另 {musique_final_audit.inherited} 个继承旧评分；A3 stage-2 文件 `{musique_final_a3_train_path}` 也是 {musique_final_a3_audit.explicit_qwen14b}/{musique_final_a3_audit.total} 个显式 14B。因此本报告把全量 14B 重打分文件作为“14B 如何评价 MAS”的独立证据，不声称它就是 MuSiQue 最终训练文件的完整评分构成。",
        "",
        "GSM-Hard 源文件中的顶层旧字段与 `judge_model=gpt-5` 是历史遗留；本报告只读取 `collaboration_judge_v4.judge_score`，并要求 `collaboration_judge_model_v4=qwen14b_gsm_judge` 且状态为 `scored`。compact v4 内的 `judge_score` 与 `process_score` 全量一致；训练构造实际读取的兼容字段 `collaboration_judge_v3.judge_score` 与 v4 在 26,920 行上也全量一致。主 self-vs-MAS 表使用全部 26,920 个有效评分 turn；排除 269 个全失败题后的 20,645-turn eligible pool 只在训练附录出现。",
        f"MultiPL-E self 读取 `judge_model=qwen14b_judge` 的 one-shot SAS 记录。MAS 读取 `{multipl_e_mas.path if multipl_e_mas else '—'}` 的全部有效评分 turn；`{MULTIPL_E_FINAL_TRAIN_LOG}` 证明它是实际 A1/A2/A3 RL 输入。上游确实只保留最终成功 trajectory，但不能据此把链内 {multipl_e_mas_main.incorrect_n if multipl_e_mas_main else 'n/a'} 个错误 turn 当成不存在；总体、共同题和同正确性配对均照实报告，并附条件化 caveat。",
        f"MATH MAS 读取 `train_judged_corrected.jsonl` 中 `collaboration_judge_v4.judge_score` 的完整 169,350 条 turn，并要求 `collaboration_judge_model_v4=qwen14b_gsm_judge` 且状态为 `scored`；该字段与 `process_score` 全量一致。judge thinking 开启，而 policy rollout 为 no-thinking。该评分池用于训练信号和 judge 偏差分析，不是 `06_eval` 的 trajectory 结果，也不应用 MATH JCA 的三-turn 预算答案投影。仓库内只找到 `{MATH_SELF_SMOKE_PATH}` 的 {math_self_smoke_rows} 条 self judge smoke 记录以及 dry-run，不足以代表完整 7500 题训练池，所以 MATH self 与双侧配对明确记为 n/a；MAS 单侧对/错统计保留。",
        (
            f"新 MATH 实验的 `{math_reuse_stats_path}` 明确记录 "
            f"`judge_scores_reused={str(math_reuse_stats.get('judge_scores_reused')).lower()}`、"
            f"`rows={math_reuse_stats.get('rows', 'n/a')}`，并把 legacy judged path 指向 "
            f"`{math_reuse_stats.get('legacy_judged_path', 'n/a')}`。因此其 `03_score/train_judged.jsonl` "
            "只是经过一致性校验的评分复用副本，不作为第二批评分重复计数；本报告继续以旧 corrected 文件作为唯一主口径。"
            if math_reuse_stats_path and math_reuse_stats
            else "未提供新 MATH 实验的评分复用元数据；本报告仍只统计主 corrected 文件一次。"
        ),
        "",
        "## 论文表述建议",
        "",
        "可以写：**在 MuSiQue 与 GSM-Hard 的完整评分池中，Qwen3-14B 对自身 rollout 的评分高于其对异构 MAS rollout 的评分；这种差异在双方被评分答案均错误的共同题上仍然显著，与 self-evaluation overconfidence 或 self-favoring calibration 一致。MultiPL-E 的实际训练评分池呈现同方向差异，但该池条件于 trajectory 最终成功，应作为补充证据而非无条件总体估计。** MATH 因缺完整 self judge，暂不进入双侧比较结论。",
        "",
        "不建议直接写成“实验已经证明 14B 因为识别出自己的答案而偏袒自己”。要识别纯粹的身份偏置，下一步应做 blind matched rejudge：将同一批 self/MAS response 匿名化，用完全相同的 prompt、上下文和 rubric 交给同一个 14B judge 重打分。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/judge_score_bias_analysis.py",
        "```",
        "",
        f"输出文件：`{output_path}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musique-self-path", type=Path, default=DEFAULT_MUSIQUE_SELF_PATH)
    parser.add_argument("--musique-mas-path", type=Path, default=DEFAULT_MUSIQUE_MAS_PATH)
    parser.add_argument("--musique-mas-raw-path", type=Path, default=MUSIQUE_MAS_RAW_PATH)
    parser.add_argument("--gsm-self-path", type=Path, default=DEFAULT_GSM_SELF_PATH)
    parser.add_argument("--gsm-mas-path", type=Path, default=DEFAULT_GSM_MAS_PATH)
    parser.add_argument("--gsm-selected-path", type=Path, default=DEFAULT_GSM_SELECTED_PATH)
    parser.add_argument("--multipl-e-self-path", type=Path, default=DEFAULT_MULTIPL_E_SELF_PATH)
    parser.add_argument("--multipl-e-mas-path", type=Path, default=DEFAULT_MULTIPL_E_MAS_PATH)
    parser.add_argument("--multipl-e-raw-full-path", type=Path, default=MULTIPL_E_RAW_FULL_PATH)
    parser.add_argument(
        "--multipl-e-raw-reroll-path", type=Path, default=MULTIPL_E_RAW_REROLL_PATH
    )
    parser.add_argument(
        "--multipl-e-full-filter-stats-path",
        type=Path,
        default=MULTIPL_E_FULL_FILTER_STATS_PATH,
    )
    parser.add_argument(
        "--multipl-e-reroll-filter-stats-path",
        type=Path,
        default=MULTIPL_E_REROLL_FILTER_STATS_PATH,
    )
    parser.add_argument("--math-mas-path", type=Path, default=DEFAULT_MATH_MAS_PATH)
    parser.add_argument(
        "--math-reuse-stats-path", type=Path, default=DEFAULT_MATH_REUSE_STATS_PATH
    )
    parser.add_argument(
        "--musique-final-train-path", type=Path, default=DEFAULT_MUSIQUE_FINAL_TRAIN_PATH
    )
    parser.add_argument(
        "--musique-final-a3-train-path",
        type=Path,
        default=DEFAULT_MUSIQUE_FINAL_A3_TRAIN_PATH,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    musique_self = load_musique(args.musique_self_path, "14B 自评")
    musique_mas = load_musique(
        args.musique_mas_path,
        "14B 评价异构 MAS",
        replay_turn_correctness=True,
    )
    gsm_self = load_gsm_self(args.gsm_self_path)
    gsm_raw = load_gsm_mas(args.gsm_mas_path, "原始 14B 评价异构 MAS")
    gsm_mas = load_gsm_mas(
        args.gsm_mas_path,
        "14B 评价异构 MAS（最终训练 eligible pool）",
        drop_all_failed_problems=True,
    )
    gsm_selected = load_gsm_mas(args.gsm_selected_path, "最终训练选择子集")
    multipl_e_self = load_multipl_e_self(args.multipl_e_self_path)
    multipl_e_mas = load_multipl_e_mas(
        args.multipl_e_mas_path,
        "14B 评价异构 MAS（最终训练 per-turn pool）",
    )
    math_mas = load_math_mas(args.math_mas_path)
    math_reuse_stats = load_json_object(args.math_reuse_stats_path, "MATH score reuse metadata")
    pool_audits = {
        "musique_raw": inspect_outcome_pool(
            args.musique_mas_raw_path, "MuSiQue raw MAS pool"
        ),
        "gsm_raw": inspect_outcome_pool(args.gsm_mas_path, "GSM-Hard raw MAS pool"),
        "multipl_e_raw_full": inspect_outcome_pool(
            args.multipl_e_raw_full_path,
            "MultiPL-E raw full MAS pool",
            multipl_e=True,
        ),
        "multipl_e_raw_reroll": inspect_outcome_pool(
            args.multipl_e_raw_reroll_path,
            "MultiPL-E raw reroll MAS pool",
            multipl_e=True,
        ),
        "multipl_e_training": inspect_outcome_pool(
            args.multipl_e_mas_path,
            "MultiPL-E final-training judged MAS pool",
            multipl_e=True,
        ),
        "math_raw": inspect_outcome_pool(args.math_mas_path, "MATH raw MAS pool"),
    }
    multipl_e_filter_stats = {
        "full": load_json_object(
            args.multipl_e_full_filter_stats_path,
            "MultiPL-E full cleaning metadata",
        ),
        "reroll": load_json_object(
            args.multipl_e_reroll_filter_stats_path,
            "MultiPL-E reroll cleaning metadata",
        ),
    }
    if pool_audits["musique_raw"].rows != musique_mas.total_records:
        raise ValueError("MuSiQue raw and judged MAS pools have different row counts")
    if pool_audits["gsm_raw"].rows != gsm_raw.total_records:
        raise ValueError("GSM-Hard raw audit and judge pool have different row counts")
    if pool_audits["multipl_e_training"].rows != multipl_e_mas.total_records:
        raise ValueError("MultiPL-E training audit and judge pool have different row counts")
    if pool_audits["math_raw"].rows != math_mas.total_records:
        raise ValueError("MATH raw audit and judge pool have different row counts")
    for dataset in ("musique_raw", "gsm_raw", "math_raw"):
        audit = pool_audits[dataset]
        if not audit.correct_trajectories or not audit.incorrect_trajectories:
            raise ValueError(f"{dataset} does not contain both correct and incorrect outcomes")
    multipl_e_training_audit = pool_audits["multipl_e_training"]
    if (
        multipl_e_training_audit.incorrect_trajectories
        or multipl_e_training_audit.unknown_trajectories
    ):
        raise ValueError("MultiPL-E final-training judge pool contains a non-success outcome")
    musique_final_audit = inspect_musique_final_training(args.musique_final_train_path)
    musique_final_a3_audit = inspect_musique_final_training(
        args.musique_final_a3_train_path
    )
    musique_eval = inspect_eval(MUSIQUE_FINAL_EVAL_PATH, "MuSiQue final eval")
    gsm_eval = inspect_eval(GSM_FINAL_EVAL_PATH, "GSM-Hard final eval")
    gsm_training_alignment = audit_gsm_final_training(
        args.gsm_mas_path, args.gsm_selected_path
    )
    for adapter in (*MUSIQUE_FINAL_ADAPTERS, *(GSM_FINAL_ADAPTER_ROOT / agent / "final" for agent in ("A1", "A2", "A3"))):
        require_file(adapter / "adapter_model.safetensors", f"final adapter {adapter}")
    for agent in ("A1", "A2", "A3"):
        require_file(
            MATH_FINAL_RUN_ROOT / "05_train/adapters" / agent / "final/rl_train_summary.json",
            f"MATH final training summary {agent}",
        )

    report = render_report(
        musique_self=musique_self,
        musique_mas=musique_mas,
        gsm_self=gsm_self,
        gsm_raw=gsm_raw,
        gsm_mas=gsm_mas,
        gsm_selected=gsm_selected,
        musique_final_train_path=args.musique_final_train_path,
        musique_final_a3_train_path=args.musique_final_a3_train_path,
        musique_final_audit=musique_final_audit,
        musique_final_a3_audit=musique_final_a3_audit,
        musique_eval=musique_eval,
        gsm_eval=gsm_eval,
        gsm_training_alignment=gsm_training_alignment,
        pool_audits=pool_audits,
        multipl_e_filter_stats=multipl_e_filter_stats,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        output_path=args.output,
        multipl_e_self=multipl_e_self,
        multipl_e_mas=multipl_e_mas,
        math_mas=math_mas,
        math_reuse_stats_path=args.math_reuse_stats_path,
        math_reuse_stats=math_reuse_stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[report] {args.output}")


if __name__ == "__main__":
    main()
