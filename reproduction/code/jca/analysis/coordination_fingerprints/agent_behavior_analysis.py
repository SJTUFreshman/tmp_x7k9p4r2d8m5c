"""Quantify agent-level coordination behavior on MuSiQue/GSM-Hard runs.

The report is deliberately based on persisted trajectory JSONL files rather than
the human-readable logs.  For each agent and dataset it reports:

* handoff probability: handoff steps / all steps by that agent;
* confirm-stop probability: confirm-stop steps / all steps by that agent;
* answer-change probability: changed non-empty candidates / eligible candidates;
* wrong-to-right probability: wrong previous candidates corrected on the next
  non-empty candidate / opportunities following a wrong candidate;
* reasoning high-overlap probability: fraction of received handoffs whose
  source and immediately following receiver reasonings have ROUGE-L F1 >= 0.8;
* exact-repeat probability: fraction of received handoffs whose source and
  receiver reasonings are identical after trimming surrounding whitespace.

The default inputs include the designated JCA runs and the corresponding
training-before-evaluation baselines with bare 1.7B/4B/8B agents.

Conifer has no exact match.  Its candidate-level correctness is the deterministic
``all_explicit_passed`` flag from ``conifer_scoring.check_constraints``, which is
exactly what every persisted Conifer step already carries in ``hard_checks``
(verified step by step on the canonical run).  Conifer answers are free-form
paragraphs, so two candidates count as "the same answer" under the same
normalisation the Conifer harness uses for majority voting: casefold, then
collapse every non-word character run into a single space.  This paragraph used
to end with "Conifer has no surviving training-before-evaluation run, so only
the trained arm is reported" -- that was wrong: ``20260901_190940`` left 1003
trajectories on disk.  It was cut short by SIGTERM at 1003 of 1402 problems, so
the trained arm is also reported restricted to those same ids; see
``benchmark_paths.CONIFER_ZERO_SHOT`` for why the raw 1003-problem subset is not
comparable to the full 1402.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Conifer's scorer is not a package; the hub scripts all extend sys.path the same way.
CONIFER_JUDGE_DIR = ROOT / "conifer_training_hub/04_judge"
if str(CONIFER_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(CONIFER_JUDGE_DIR))

from gsm.src.data import extract_number  # noqa: E402
from gsm.src.grader import numeric_values_match  # noqa: E402
from src.grader import is_correct as musique_is_correct  # noqa: E402
from src.math_eval import compute_math_em  # noqa: E402
from conifer_scoring import check_constraints as conifer_check_constraints  # noqa: E402

try:
    from .benchmark_paths import (
        CONIFER_JCA,
        CONIFER_ZERO_SHOT,
        MATH_JCA,
        MATH_ZERO_SHOT,
    )
except ImportError:
    from benchmark_paths import (  # type: ignore[no-redef]
        CONIFER_JCA,
        CONIFER_ZERO_SHOT,
        MATH_JCA,
        MATH_ZERO_SHOT,
    )

from analysis.math.common import (  # noqa: E402
    MATH_JCA_TURN_BUDGET,
    is_math_jca_bootstrap_copy_step,
    math_jca_answer_projection,
)


AGENTS = ("A1", "A2", "A3")
DEFAULT_MUSIQUE_PATH = ROOT / "logs/mas_eval_concurrent/rl_sft_mas_0717_1130/results.jsonl"
DEFAULT_GSM_PATH = (
    ROOT
    / "outputs/gsm_eval/role_batched/"
    / "gsm_judge_rl_v13_seed_sweep_infinite_20260809/"
    / "gsm_judge_rl_v13_seed_sweep_infinite_20260809_seed43_dev132_"
    "thinking_hidden_self_handoff_ctx40960.jsonl"
)
DEFAULT_MUSIQUE_ZEROSHOT_PATH = (
    ROOT
    / "logs/sft_old_protocol/20260706_092608_base_old_sft_dev_start0_n2417/"
    "results.jsonl"
)
DEFAULT_GSM_ZEROSHOT_PATH = (
    ROOT
    / "outputs/gsm_eval/gsm_new_eval_thinking_20260808/"
    "gsm_zero_shot_mas_hetero_new_eval_dev132.jsonl"
)
DEFAULT_MATH_ZEROSHOT_PATH = MATH_ZERO_SHOT
DEFAULT_MATH_JCA_PATH = MATH_JCA
DEFAULT_CONIFER_JCA_PATH = CONIFER_JCA
DEFAULT_CONIFER_ZEROSHOT_PATH = CONIFER_ZERO_SHOT
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "agent_behavior_report.md"
DEFAULT_ROUGE_L_THRESHOLD = 0.8


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


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _candidate(step: dict[str, Any]) -> str:
    """Get the best answer candidate represented by one protocol step."""
    for field_name in ("tentative_answer", "confirmed_answer", "final_answer"):
        value = _clean_text(step.get(field_name))
        if value:
            return value
    return ""


def _action(step: dict[str, Any]) -> str:
    action = _clean_text(step.get("action")).lower()
    if action:
        return action
    if _clean_text(step.get("handoff_target")):
        return "handoff"
    if _clean_text(step.get("confirmed_answer")) or _clean_text(step.get("final_answer")):
        return "confirm_stop"
    return ""


def _is_handoff(step: dict[str, Any]) -> bool:
    """Return whether a step's resolved action is an actual hand-off."""
    return _action(step) in {"handoff", "hand-off"}


def _musique_problem(record: dict[str, Any]) -> SimpleNamespace:
    problem = record.get("problem") or {}
    return SimpleNamespace(
        answer=_clean_text(problem.get("answer")),
        answer_aliases=[_clean_text(alias) for alias in problem.get("answer_aliases", [])],
    )


def _conifer_vote_key(answer: str) -> str:
    """Normalise a Conifer answer the way the harness buckets votes."""
    return re.sub(r"[^\w]+", " ", answer.casefold(), flags=re.UNICODE).strip()


def _answer_correct(dataset: str, candidate: str, record: dict[str, Any]) -> bool:
    if not candidate:
        return False
    if dataset == "Conifer":
        # Deterministic and offline; reproduces each step's persisted
        # hard_checks.all_explicit_passed exactly.
        problem = record.get("problem") or {}
        return bool(conifer_check_constraints(problem, candidate)["all_explicit_passed"])

    if dataset == "MuSiQue":
        return bool(musique_is_correct(candidate, _musique_problem(record)))

    if dataset == "MATH":
        problem = record.get("problem") or {}
        target = _clean_text(record.get("gold_answer") or problem.get("gold_answer"))
        return bool(target and compute_math_em(candidate, target))

    problem = record.get("problem") or {}
    target = extract_number(_clean_text(problem.get("answer")))
    prediction = extract_number(candidate)
    if target is None or prediction is None:
        return False
    return bool(numeric_values_match(prediction, target))


def _answers_equal(dataset: str, left: str, right: str) -> bool:
    """Compare candidates using the dataset's answer semantics."""
    if not left or not right:
        return False
    if dataset == "Conifer":
        return _conifer_vote_key(left) == _conifer_vote_key(right)

    if dataset == "MuSiQue":
        from src.grader import extract_boxed_answer, normalize_answer

        left_answer = extract_boxed_answer(left) or left
        right_answer = extract_boxed_answer(right) or right
        return normalize_answer(left_answer) == normalize_answer(right_answer)

    if dataset == "MATH":
        return bool(compute_math_em(left, right) or compute_math_em(right, left))

    left_number = extract_number(left)
    right_number = extract_number(right)
    if left_number is None and right_number is None:
        return left.strip().lower() == right.strip().lower()
    if left_number is None or right_number is None:
        return False
    return bool(numeric_values_match(left_number, right_number))


class RougeLScorer:
    def __init__(self) -> None:
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise RuntimeError(
                "rouge-score is required for reasoning overlap; "
                "install it with `pip install rouge-score`"
            ) from exc
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, source: str, receiver: str) -> float:
        return float(self.scorer.score(source, receiver)["rougeL"].fmeasure)


@dataclass
class AgentStats:
    agent: str
    n_problems: int = 0
    n_steps: int = 0
    handoff_steps: int = 0
    confirm_stop_steps: int = 0
    answer_opportunities: int = 0
    changed_answers: int = 0
    wrong_previous_opportunities: int = 0
    wrong_to_right: int = 0
    wrong_changed_opportunities: int = 0
    wrong_to_right_changed: int = 0
    rouge_l_sum: float = 0.0
    overlap_pairs: int = 0
    overlap_above_threshold: int = 0
    exact_repeats: int = 0
    overlap_problems: int = 0

    @property
    def handoff_probability(self) -> float:
        return self.handoff_steps / self.n_steps if self.n_steps else 0.0

    @property
    def confirm_stop_probability(self) -> float:
        return self.confirm_stop_steps / self.n_steps if self.n_steps else 0.0

    @property
    def answer_change_probability(self) -> float:
        return self.changed_answers / self.answer_opportunities if self.answer_opportunities else 0.0

    @property
    def wrong_to_right_probability(self) -> float:
        return self.wrong_to_right / self.wrong_previous_opportunities if self.wrong_previous_opportunities else 0.0

    @property
    def wrong_to_right_given_change(self) -> float:
        return self.wrong_to_right_changed / self.wrong_changed_opportunities if self.wrong_changed_opportunities else 0.0

    @property
    def mean_rouge_l(self) -> float:
        return self.rouge_l_sum / self.overlap_pairs if self.overlap_pairs else 0.0

    @property
    def reasoning_high_overlap_probability(self) -> float:
        return self.overlap_above_threshold / self.overlap_pairs if self.overlap_pairs else 0.0

    @property
    def exact_repeat_probability(self) -> float:
        return self.exact_repeats / self.overlap_pairs if self.overlap_pairs else 0.0


def _steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = record.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    return [step for step in steps if isinstance(step, dict)]


def _initial_agent(record: dict[str, Any]) -> str:
    """Return the agent that received the problem first in a persisted run."""
    top_level = _clean_text(record.get("start_agent"))
    if top_level:
        return top_level
    trajectory = record.get("trajectory") or {}
    trajectory_start = _clean_text(trajectory.get("start_agent"))
    if trajectory_start:
        return trajectory_start
    steps = _steps(record)
    return _clean_text(steps[0].get("active_agent")) if steps else ""


def _is_synthetic_behavior_step(
    dataset: str,
    method: str,
    record: dict[str, Any],
    step_index: int,
    step: dict[str, Any],
) -> bool:
    return (
        dataset == "MATH"
        and method.lower() in {"jca", "zero-shot", "zeroshot"}
        and is_math_jca_bootstrap_copy_step(record, step_index, step)
    )


def _record_final_answer(record: dict[str, Any]) -> str:
    """Return the persisted final answer, including legacy trajectory records."""
    for value in (
        record.get("final_answer"),
        (record.get("trajectory") or {}).get("final_answer"),
    ):
        answer = _clean_text(value)
        if answer:
            return answer
    for step in reversed(_steps(record)):
        answer = _candidate(step)
        if answer:
            return answer
    return ""


def _record_em(dataset: str, record: dict[str, Any], method: str = "") -> float:
    """Return the result EM, applying the MATH JCA projection when needed."""
    if dataset == "Conifer":
        # Trust the harness verdict on the persisted final answer.  Falling back
        # to the last non-empty step candidate (as the generic path does) would
        # rescue 3 of the 8 records whose final_answer is empty and push the
        # score to 1287/1402, disagreeing with the canonical 1284/1402.
        return float(bool((record.get("final_checks") or {}).get("all_explicit_passed")))
    if dataset == "MATH" and method.lower() == "jca":
        projection = math_jca_answer_projection(record)
        problem = record.get("problem") or {}
        target = _clean_text(record.get("gold_answer") or problem.get("gold_answer"))
        return float(_answer_correct(dataset, projection["projected_answer"], record)) if target else 0.0
    if "em" in record:
        try:
            return float(record["em"])
        except (TypeError, ValueError):
            pass
    return float(_answer_correct(dataset, _record_final_answer(record), record))


def analyze_dataset(
    dataset: str,
    path: Path,
    rouge_l_scorer: RougeLScorer,
    overlap_threshold: float,
    method: str = "JCA",
    max_steps: int | None = None,
    initial_agent: str | None = None,
    restrict_ids: set[str] | None = None,
) -> tuple[list[AgentStats], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} input does not exist: {path}")

    stats = {agent: AgentStats(agent=agent) for agent in AGENTS}
    n_records = 0
    n_records_with_steps = 0
    n_correct = 0
    em_sum = 0.0
    n_analyzed_steps = 0
    n_raw_steps = 0
    n_persisted_steps = 0
    n_window_persisted_steps = 0
    n_synthetic_steps_removed = 0
    seen_ids: set[str] = set()
    problem_ids_by_agent: dict[str, set[str]] = defaultdict(set)
    overlap_problem_ids_by_agent: dict[str, set[str]] = defaultdict(set)
    selected_records: list[dict[str, Any]] = []

    for record_index, record in enumerate(iter_jsonl(path), start=1):
        if initial_agent is not None and _initial_agent(record) != initial_agent:
            continue
        problem = record.get("problem") or {}
        problem_id = (
            _clean_text(problem.get("id"))
            or _clean_text(problem.get("problem_id"))  # Conifer keeps the id under this name
            or _clean_text(record.get("problem_id"))
            or f"record_{record_index}"
        )
        if restrict_ids is not None and problem_id not in restrict_ids:
            continue
        n_records += 1
        selected_records.append(record)
        record_em = _record_em(dataset, record, method)
        em_sum += record_em
        if record_em >= 1.0:
            n_correct += 1
        if problem_id in seen_ids:
            raise ValueError(f"{path}: duplicate problem.id {problem_id!r}")
        seen_ids.add(problem_id)

        persisted_steps = _steps(record)
        window_steps = (
            persisted_steps
            if max_steps is None
            else persisted_steps[:max_steps]
        )
        raw_behavior_steps = [
            step
            for step_index, step in enumerate(persisted_steps)
            if not _is_synthetic_behavior_step(
                dataset, method, record, step_index, step
            )
        ]
        behavior_steps = [
            (step_index, step)
            for step_index, step in enumerate(window_steps)
            if not _is_synthetic_behavior_step(
                dataset, method, record, step_index, step
            )
        ]
        synthetic_steps_removed = len(window_steps) - len(behavior_steps)
        n_persisted_steps += len(persisted_steps)
        n_window_persisted_steps += len(window_steps)
        n_synthetic_steps_removed += synthetic_steps_removed
        if behavior_steps:
            n_records_with_steps += 1
        n_raw_steps += len(raw_behavior_steps)
        n_analyzed_steps += len(behavior_steps)

        previous_candidate = ""
        previous_correct = False

        for step_index, step in enumerate(window_steps):
            current_candidate = _candidate(step)
            current_correct = (
                _answer_correct(dataset, current_candidate, record)
                if current_candidate
                else False
            )
            if _is_synthetic_behavior_step(
                dataset, method, record, step_index, step
            ):
                if current_candidate:
                    previous_candidate = current_candidate
                    previous_correct = current_correct
                continue

            agent = _clean_text(step.get("active_agent"))
            if agent not in stats:
                continue
            problem_ids_by_agent[agent].add(problem_id)
            agent_stats = stats[agent]
            agent_stats.n_steps += 1

            action = _action(step)
            if _is_handoff(step):
                agent_stats.handoff_steps += 1
            if action in {"confirm_stop", "confirm-stop"}:
                agent_stats.confirm_stop_steps += 1

            if current_candidate and previous_candidate:
                agent_stats.answer_opportunities += 1
                changed = not _answers_equal(dataset, previous_candidate, current_candidate)
                if changed:
                    agent_stats.changed_answers += 1
                if not previous_correct:
                    agent_stats.wrong_previous_opportunities += 1
                    if changed:
                        agent_stats.wrong_changed_opportunities += 1
                    if current_correct:
                        agent_stats.wrong_to_right += 1
                        if changed:
                            agent_stats.wrong_to_right_changed += 1
            if current_candidate:
                previous_candidate = current_candidate
                previous_correct = current_correct

        for step_index, source_step in enumerate(window_steps[:-1]):
            receiver_step = window_steps[step_index + 1]
            if _is_synthetic_behavior_step(
                dataset, method, record, step_index, source_step
            ) or _is_synthetic_behavior_step(
                dataset, method, record, step_index + 1, receiver_step
            ):
                continue
            source_agent = _clean_text(source_step.get("active_agent"))
            receiver_agent = _clean_text(source_step.get("handoff_target"))
            if (
                not _is_handoff(source_step)
                or source_agent not in stats
                or receiver_agent not in stats
                or source_agent == receiver_agent
            ):
                continue

            if _clean_text(receiver_step.get("active_agent")) != receiver_agent:
                continue

            source_reasoning = _clean_text(source_step.get("reasoning"))
            receiver_reasoning = _clean_text(receiver_step.get("reasoning"))
            if not source_reasoning or not receiver_reasoning:
                continue

            rouge_l = rouge_l_scorer.score(source_reasoning, receiver_reasoning)
            receiver_stats = stats[receiver_agent]
            receiver_stats.rouge_l_sum += rouge_l
            receiver_stats.overlap_pairs += 1
            if rouge_l >= overlap_threshold:
                receiver_stats.overlap_above_threshold += 1
            if source_reasoning == receiver_reasoning:
                receiver_stats.exact_repeats += 1
            overlap_problem_ids_by_agent[receiver_agent].add(problem_id)

    for agent, agent_stats in stats.items():
        agent_stats.n_problems = len(problem_ids_by_agent[agent])
        agent_stats.overlap_problems = len(overlap_problem_ids_by_agent[agent])

    metadata = {
        "dataset": dataset,
        "path": str(path),
        "n_records": n_records,
        "n_records_with_steps": n_records_with_steps,
        "n_analyzed_steps": n_analyzed_steps,
        "n_raw_steps": n_raw_steps,
        "n_persisted_steps": n_persisted_steps,
        "n_window_persisted_steps": n_window_persisted_steps,
        "n_synthetic_steps_removed": n_synthetic_steps_removed,
        "step_limit": max_steps,
        "n_problem_ids": len(seen_ids),
        "n_correct": n_correct,
        "em": em_sum / n_records if n_records else 0.0,
        "initial_agent": initial_agent,
        "restrict_ids": len(restrict_ids) if restrict_ids is not None else None,
    }
    if dataset == "MATH" and method.lower() == "jca":
        math_rows = selected_records
        projections = [math_jca_answer_projection(record) for record in math_rows]
        metadata["raw_n_correct"] = sum(
            bool(_answer_correct(dataset, item["raw_final_answer"], record))
            for item, record in zip(projections, math_rows)
        )
        metadata["answer_projection"] = {
            "policy": projections[0]["policy"] if projections else "",
            "overlong_records": sum(item["overlong"] for item in projections),
            "fallback_correct": sum(
                _answer_correct(dataset, item["projected_answer"], record)
                for item, record in zip(projections, math_rows)
                if item["projection_applied"]
            ),
        }
    return list(stats.values()), metadata


def _fmt_rate(value: float, numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{value:.1%} ({numerator}/{denominator})"


def render_report(
    datasets: list[tuple[str, str, Path, list[AgentStats], dict[str, Any]]],
    output_path: Path,
    overlap_threshold: float,
) -> str:
    lines = [
        "# 协作指纹：Agent 级行为量化分析",
        "",
        "本报告汇总 MuSiQue、GSM-Hard 的 JCA/训练前基线，MATH 的 JCA 与训练前裸异构基线，以及 Conifer 的 JCA 训练后运行。MATH 两种运行的行为概率都排除强制 A3 格式转换；JCA 先截取前三个落盘 turn 再排除，zero-shot 使用完整落盘窗口。JCA 完整轨迹仅作审计。",
        "Conifer 没有 EM，正确性一律用确定性的 `all_explicit_passed`。它的训练前 zero-shot 运行**没跑完**（1003/1402 题，SIGTERM 中断）且这批题偏难，所以额外给了一行限制到同一批 1003 个 `problem_id` 的训练后结果；跨训练前后比较请只看后两行。",
        "所有概率均同时报告 `分子/分母`；分母为零时记为 `n/a`。",
        "",
        "## 运行分数",
        "",
        "EM 通常读取各 eval JSONL 中由对应 runner 持久化的字段；MATH JCA 按特殊三-turn 预算回退规则重算，其他缺失字段才使用项目 grader 回退计算。",
        "",
        "| 数据集 | 运行 | 正确数/题目数 | EM |",
        "|---|---|---:|---:|",
    ]

    for dataset, label, _, _, metadata in datasets:
        lines.append(
            f"| {dataset} | {label} | {metadata['n_correct']}/{metadata['n_records']} "
            f"| {metadata['em']:.2%} |"
        )

    lines.extend([
        "",
        "MuSiQue 的 JCA 路径是 `rl_sft_mas_0717_1130`，在 20 个完整 `rl_sft_mas_*` 运行中最高（1024/2417）。",
        "MuSiQue 训练前基线使用 `Qwen3-1.7B + Qwen3-4B + Qwen3-8B` 裸模型、无 LoRA 的 `base_old_sft` 全量运行（514/2417）。",
        "GSM-Hard 的 JCA 路径是同一最终 adapter 的正式批量 seed sweep 中 seed=43 的论文主结果（95/132 = 71.97%）。",
        "GSM-Hard 训练前基线使用 `Qwen3-1.7B + Qwen3-4B + Qwen3-8B` 裸模型、无 LoRA 的异构 MAS 运行（91/132）。",
        "MATH JCA 超过三 turn，或第三 turn 后仍未成功 `stop` 时，使用首个 `tentative_answer` 作为最终 scorer 输入，最新 eval 因此为 392/500（78.40%）；主行为统计只看前三个落盘 turn中的自由策略动作。",
        "MATH 训练前异构 zero-shot 使用 `Qwen3-1.7B + Qwen3-4B + Qwen3-8B` 裸模型，并通过零初始化 adapter 加载（无有效训练 LoRA）；no-thinking，500 题（315/500）。",
        "Conifer 的 JCA 路径是 `rl_mas_eval_forever_20260906/round_4`，即 `benchmark_paths.py` 登记的 canonical 单点（coverage 88.90 / explicit 95.90，1284/1402 = 91.58%）；该表「EM」一列对 Conifer 读作 `all_explicit_passed` 通过率，不是 EM。",
        "Conifer 训练前 zero-shot 是 `10_outputs/20260901_190940_conifer_zero_shot_mas`：裸 `Qwen3-1.7B + Qwen3-4B + Qwen3-8B`，`ADAPTER_A1/A2/A3` 全空（无 LoRA），协议常数与训练后逐项一致（`T_MAX=6`、`ROUTING=dynamic`、`MIN_AGENTS_BEFORE_STOP=3`、`MIN_HANDOFFS_BEFORE_STOP=2`、`START_AGENT=balanced/42`、top-p 0.95、max_new_tokens 1024）。两处不可控差异：**该运行被 SIGTERM 掐掉，只落盘 1003/1402 题**（覆盖 test.jsonl 前 1100 条里的 1003 条，且这批题在 `all_explicit_passed` 口径上偏易——平均每题 0.50 个显式检查项，补集 0.93；同一个训练后 round_4 在子集上 962/1003 = 95.91%，在补集上只有 322/399 = 80.70%），以及**解码温度 0.0 而训练后是 0.3**。因此训练前后只应比较最后两行。",
        "",
        "## 指标定义",
        "",
        "- **Hand-off 概率** = resolved `action=handoff` 的 step 数 / 该 Agent 的全部 step 数；若缺少 action，则由非空 `handoff_target` 推断为 hand-off。显式 `confirm_stop` 优先，二者互斥。",
        "- **confirm_stop 概率** = confirm-stop step 数 / 该 Agent 的全部 step 数。",
        "- **答案更改概率** = 更改后的非空候选数 / 同时存在当前和此前候选的 step 数。",
        "- **错→对概率** = 此前候选错误且当前候选正确的次数 / 此前候选错误且当前有非空候选的次数。",
        "- MATH JCA 与 zero-shot 的首个落盘 step 是 evaluator 强制的 A3 格式转换：它不进入 Agent 的 step、Hand-off 或 confirm-stop 分母，但其中的 `tentative_answer` 仍作为下一自由 step 判断答案是否改变、是否错→对的前态。JCA 先截取前三个落盘 turn，再做这一步排除，因此绝不会把第 4 个落盘 turn 补进来；zero-shot 不做三-turn截断。",
        f"- **Reasoning 高重合概率**：仅统计真实相邻 Hand-off；若 source step 的 `handoff_target` 等于下一 step 的 `active_agent`，则比较两步 reasoning，并将该次比较归入接收方 Agent。指标为 ROUGE-L F1 ≥ {overlap_threshold:.1f} 的次数 / 该 Agent 的有效接手比较次数。ROUGE-L 使用英文词干归一化。",
        "- **平均 ROUGE-L**：上述有效接手对比的 ROUGE-L F1 平均值；越接近 1，按相同顺序复用的文本越多。",
        "- **完全复述概率**：去除 reasoning 首尾空白后，source 与 receiver 文本完全相同的次数 / 有效接手比较次数。",
        "- Conifer 的两处口径替换：**正确**指 `conifer_scoring.check_constraints(problem, candidate)` 的 `all_explicit_passed`（确定性、可离线重算，逐 step 与落盘 `hard_checks` 完全一致，4676/4676）；**两个候选相同**指 Conifer harness 投票用的归一化（casefold 后把所有非词字符压成单空格）后逐字符相等。Conifer 答案是长段落，这个判据偏严，答案更改概率因此接近上界。",
        "",
    ])

    for dataset, label, path, agent_stats, metadata in datasets:
        lines.extend([
            f"## {dataset} / {label}",
            "",
            f"输入文件：`{path}`  ",
            f"题目数：{metadata['n_records']}；EM：{metadata['n_correct']}/{metadata['n_records']} = {metadata['em']:.2%}；有轨迹记录：{metadata['n_records_with_steps']}；纳入行为统计的 step：{metadata['n_analyzed_steps']}",
            "",
        ])
        if metadata.get("n_synthetic_steps_removed"):
            lines.extend([
                (
                    "行为统计已排除 "
                    f"{metadata['n_synthetic_steps_removed']} 条强制复制 bootstrap 答案的 synthetic step；"
                    f"主窗口原有 {metadata['n_window_persisted_steps']} 条持久化 step，"
                    f"原文件共 {metadata['n_persisted_steps']} 条。被排除 step 的 tentative 仍用于答案变化前态。"
                ),
                "",
            ])
        projection = metadata.get("answer_projection")
        if projection:
            lines.extend([
                f"MATH JCA 答案投影：{projection['policy']}；原始正确 {metadata.get('raw_n_correct', 'n/a')}，触发预算回退 {projection['overlong_records']} 题，其中回退答案正确 {projection['fallback_correct']} 题。",
                "",
            ])
        lines.extend([
            "| Agent | 参与题数 | Hand-off 概率 | confirm_stop 概率 | 答案更改概率 | 错→对概率 | Reasoning 高重合概率 | 平均 ROUGE-L | 完全复述概率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in agent_stats:
            lines.append(
                f"| {item.agent} | {item.n_problems} "
                f"| {_fmt_rate(item.handoff_probability, item.handoff_steps, item.n_steps)} "
                f"| {_fmt_rate(item.confirm_stop_probability, item.confirm_stop_steps, item.n_steps)} "
                f"| {_fmt_rate(item.answer_change_probability, item.changed_answers, item.answer_opportunities)} "
                f"| {_fmt_rate(item.wrong_to_right_probability, item.wrong_to_right, item.wrong_previous_opportunities)} "
                f"| {_fmt_rate(item.reasoning_high_overlap_probability, item.overlap_above_threshold, item.overlap_pairs)} "
                f"| {item.mean_rouge_l:.3f}（{item.overlap_pairs} 次接手对比）"
                f"| {_fmt_rate(item.exact_repeat_probability, item.exact_repeats, item.overlap_pairs)} |"
            )
        lines.extend([
            "",
            "### 分母与计数明细",
            "",
            "| Agent | Step 数 | Hand-off 数 | confirm_stop 数 | 答案比较机会 | 更改答案数 | 错误后续机会 | 错→对数 | 错后更改数 | 错→对/错后更改 | 有效接手对比 | 高重合数 | 完全复述数 | 有效接手题数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in agent_stats:
            lines.append(
                f"| {item.agent} | {item.n_steps} | {item.handoff_steps} | {item.confirm_stop_steps} "
                f"| {item.answer_opportunities} | {item.changed_answers} | {item.wrong_previous_opportunities} "
                f"| {item.wrong_to_right} | {item.wrong_to_right_changed} "
                f"| {_fmt_rate(item.wrong_to_right_given_change, item.wrong_to_right_changed, item.wrong_changed_opportunities)} "
                f"| {item.overlap_pairs} | {item.overlap_above_threshold} | {item.exact_repeats} | {item.overlap_problems} |"
            )
        lines.append("")

        full_audit = metadata.get("full_trajectory_audit")
        if full_audit:
            lines.extend([
                "### 完整 raw 轨迹行为审计",
                "",
                (
                    f"主表先截取前 {metadata['step_limit']} 个落盘 turn，再排除 synthetic step，"
                    f"共纳入 {metadata['n_analyzed_steps']} 个自由策略 step；完整落盘轨迹共有 "
                    f"{full_audit['n_persisted_steps']} 个 step，排除 "
                    f"{full_audit['n_synthetic_steps_removed']} 个 synthetic step 后，完整行为轨迹为 "
                    f"{full_audit['n_analyzed_steps']} 个 step。以下仅用于观察预算后的行为，不进入论文主指标。"
                ),
                "",
                "| Agent | Step 数 | Hand-off 概率 | confirm_stop 概率 | 答案更改概率 | 错→对概率 | Reasoning 高重合概率 | 平均 ROUGE-L | 完全复述概率 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for raw_stats in full_audit["agent_stats"]:
                lines.append(
                    f"| {raw_stats.agent} | {raw_stats.n_steps} | "
                    f"{_fmt_rate(raw_stats.handoff_probability, raw_stats.handoff_steps, raw_stats.n_steps)} | "
                    f"{_fmt_rate(raw_stats.confirm_stop_probability, raw_stats.confirm_stop_steps, raw_stats.n_steps)} | "
                    f"{_fmt_rate(raw_stats.answer_change_probability, raw_stats.changed_answers, raw_stats.answer_opportunities)} | "
                    f"{_fmt_rate(raw_stats.wrong_to_right_probability, raw_stats.wrong_to_right, raw_stats.wrong_previous_opportunities)} | "
                    f"{_fmt_rate(raw_stats.reasoning_high_overlap_probability, raw_stats.overlap_above_threshold, raw_stats.overlap_pairs)} | "
                    f"{raw_stats.mean_rouge_l:.3f} | "
                    f"{_fmt_rate(raw_stats.exact_repeat_probability, raw_stats.exact_repeats, raw_stats.overlap_pairs)} |"
                )
            lines.append("")

    lines.extend([
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/agent_behavior_analysis.py",
        "```",
        "",
        f"输出文件：`{output_path}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musique-path", type=Path, default=DEFAULT_MUSIQUE_PATH)
    parser.add_argument("--gsm-path", type=Path, default=DEFAULT_GSM_PATH)
    parser.add_argument("--musique-zeroshot-path", type=Path, default=DEFAULT_MUSIQUE_ZEROSHOT_PATH)
    parser.add_argument("--gsm-zeroshot-path", type=Path, default=DEFAULT_GSM_ZEROSHOT_PATH)
    parser.add_argument("--math-jca-path", type=Path, default=DEFAULT_MATH_JCA_PATH)
    parser.add_argument("--math-zeroshot-path", type=Path, default=DEFAULT_MATH_ZEROSHOT_PATH)
    parser.add_argument("--conifer-path", type=Path, default=DEFAULT_CONIFER_JCA_PATH)
    parser.add_argument(
        "--conifer-zeroshot-path", type=Path, default=DEFAULT_CONIFER_ZEROSHOT_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rouge-l-threshold", type=float, default=DEFAULT_ROUGE_L_THRESHOLD)
    args = parser.parse_args()

    if not 0.0 <= args.rouge_l_threshold <= 1.0:
        parser.error("--rouge-l-threshold must be between 0 and 1")
    rouge_l_scorer = RougeLScorer()

    datasets: list[tuple[str, str, Path, list[AgentStats], dict[str, Any]]] = []
    runs = (
        ("MuSiQue", "JCA（正式运行）", args.musique_path, "JCA"),
        ("MuSiQue", "训练前基线（1.7B+4B+8B，无 LoRA）", args.musique_zeroshot_path, "zero-shot"),
        ("GSM-Hard", "JCA（正式运行）", args.gsm_path, "JCA"),
        ("GSM-Hard", "训练前基线（1.7B+4B+8B，无 LoRA）", args.gsm_zeroshot_path, "zero-shot"),
        ("MATH", "JCA 正式 eval（三-turn 预算回退）", args.math_jca_path, "JCA"),
        ("MATH", "训练前异构 zero-shot（1.7B+4B+8B；零初始化 adapter；no-thinking）", args.math_zeroshot_path, "zero-shot"),
        ("Conifer", "JCA 训练后 eval（round_4）", args.conifer_path, "JCA"),
        (
            "Conifer",
            "训练前异构 zero-shot MAS（1.7B+4B+8B，无 LoRA；仅 1003/1402 题）",
            args.conifer_zeroshot_path,
            "zero-shot",
        ),
        (
            "Conifer",
            "JCA 训练后（限 zero-shot 同批 1003 题，供对照）",
            args.conifer_path,
            "JCA",
        ),
    )
    # Conifer 的 zero-shot 运行被 SIGTERM 掐掉，只覆盖 1402 题里的 1003 题，且这批题
    # 偏难；直接拿它和训练后的 1402 题行比会把题目难度差记到训练头上，所以额外跑一行
    # 限制到同一批 problem_id 的训练后结果。
    conifer_common_ids = {
        _clean_text((record.get("problem") or {}).get("problem_id"))
        or _clean_text(record.get("problem_id"))
        for record in iter_jsonl(args.conifer_zeroshot_path)
    }
    conifer_common_ids.discard("")
    for dataset, label, path, method in runs:
        step_limit = (
            MATH_JCA_TURN_BUDGET
            if dataset == "MATH" and method.lower() == "jca"
            else None
        )
        restrict_ids = conifer_common_ids if "限 zero-shot 同批" in label else None
        stats, metadata = analyze_dataset(
            dataset,
            path,
            rouge_l_scorer,
            args.rouge_l_threshold,
            method=method,
            max_steps=step_limit,
            restrict_ids=restrict_ids,
        )
        if step_limit is not None:
            raw_stats, raw_metadata = analyze_dataset(
                dataset,
                path,
                rouge_l_scorer,
                args.rouge_l_threshold,
                method=method,
            )
            metadata["full_trajectory_audit"] = {
                "n_analyzed_steps": raw_metadata["n_analyzed_steps"],
                "n_persisted_steps": raw_metadata["n_persisted_steps"],
                "n_synthetic_steps_removed": raw_metadata[
                    "n_synthetic_steps_removed"
                ],
                "agent_stats": raw_stats,
            }
        datasets.append((dataset, label, path, stats, metadata))

    report = render_report(
        datasets,
        args.output,
        args.rouge_l_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] {args.output}")


if __name__ == "__main__":
    main()
