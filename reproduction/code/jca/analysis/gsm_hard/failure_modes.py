"""Auditable GSM-Hard failure and collaboration analysis.

The primary axes measure answer-regime robustness, execution health, and the
fate of correct intermediate candidates.  A legacy numerical-signature axis is
retained as an appendix; it describes prediction/gold relationships rather than
claiming to identify reasoning causes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
SCALE_FACTORS = (10.0, 100.0, 1000.0, 60.0, 3600.0, 12.0, 24.0, 7.0)
OUTCOME_ORDER = (
    "missing_answer",
    "non_numeric",
    "sign_error",
    "rounding_or_precision",
    "off_by_one",
    "scale_or_unit",
    "other_numeric",
)
PROCESS_ORDER = (
    "execution_or_protocol_failure",
    "harmful_correction",
    "premature_stop",
    "unrepaired_wrong_answer",
    "other_collaboration_failure",
)
GOLD_STRATUM_ORDER = (
    "negative",
    "nonnegative_below_one",
    "nonnegative_integer",
    "nonnegative_fraction",
)
GOLD_STRATUM_LABELS = {
    "negative": "负数",
    "nonnegative_below_one": "非负且小于 1",
    "nonnegative_integer": "非负整数（>=1）",
    "nonnegative_fraction": "非负非整数（>=1）",
}
DISPLAY_NAMES = {
    "mad": "MAD",
    "agentverse": "AgentVerse",
    "aflow": "AFlow",
    "gptswarm": "GPTSwarm",
    "jca": "JCA",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def extract_number(value: object) -> Optional[float]:
    """Mirror the GSM-Hard grader's last-number behavior."""
    text = str(value or "").strip()
    if not text:
        return None
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    candidates = [boxed.group(1)] if boxed else NUMBER_PATTERN.findall(text)
    if not candidates:
        return None
    try:
        number = float(candidates[-1].replace(",", "").strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def numeric_match(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return False
    if abs(left - right) <= 1e-6:
        return True
    return abs(right) > 1e-9 and abs(left - right) / abs(right) < 1e-6


def _is_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record.get("correct"))
    return float(record.get("em", 0.0) or 0.0) >= 1.0


def classify_gold_stratum(record: dict[str, Any]) -> str:
    gold = extract_number((record.get("problem") or {}).get("answer"))
    if gold is None:
        raise ValueError("GSM-Hard record has no numeric gold answer")
    if gold < 0:
        return "negative"
    if gold < 1:
        return "nonnegative_below_one"
    if gold.is_integer():
        return "nonnegative_integer"
    return "nonnegative_fraction"


def _load_json_object(raw_output: object) -> Optional[Any]:
    """Parse the visible JSON payload used by the baseline adapters."""
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start():])
                return payload
            except json.JSONDecodeError:
                continue
    return None


def _aflow_solve_answers(record: dict[str, Any]) -> list[object]:
    answers: list[object] = []
    for operation in record.get("op_records", []) or []:
        if operation.get("op") != "solve" or not operation.get("ok"):
            continue
        raw_outputs = operation.get("raw_outputs") or []
        raw_output = raw_outputs[-1] if raw_outputs else operation.get("raw_output")
        payload = _load_json_object(raw_output)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if isinstance(payload, dict) and payload.get("answer") not in (None, ""):
            answers.append(payload["answer"])
    return answers


def intermediate_answers(method: str, record: dict[str, Any]) -> list[object]:
    """Return parsed candidates available before final answer selection."""
    method = method.lower()
    if method == "agentverse":
        return [
            answer.get("answer")
            for iteration in (record.get("agentverse") or {}).get("iterations", []) or []
            for answer in iteration.get("answers", []) or []
            if answer.get("answer") not in (None, "")
        ]
    if method == "aflow":
        return _aflow_solve_answers(record)
    if method == "gptswarm":
        return [
            node.get("answer")
            for node in (record.get("swarm") or {}).get("node_outputs", []) or []
            if node.get("node_type") != "Aggregator"
            and node.get("answer") not in (None, "")
        ]
    if method == "mad":
        return [
            turn.get("answer")
            for turn in (record.get("mad") or {}).get("turns", []) or []
            if turn.get("answer") not in (None, "")
        ]
    if method == "jca":
        return [
            step.get("tentative_answer")
            for step in (record.get("trajectory") or {}).get("steps", []) or []
            if step.get("tentative_answer") not in (None, "")
        ]
    raise ValueError(f"unsupported method: {method}")


def _has_explicit_error(method: str, record: dict[str, Any]) -> bool:
    method = method.lower()
    if method == "aflow":
        return bool(record.get("error"))
    container_name = {
        "agentverse": "agentverse",
        "gptswarm": "swarm",
        "mad": "mad",
        "jca": "trajectory",
    }.get(method)
    container = record.get(container_name) or {} if container_name else {}
    if container.get("error"):
        return True
    if method == "jca":
        terminated_by = record.get("terminated_by", container.get("terminated_by"))
        return terminated_by not in (None, "stop")
    return False


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-12)


def classify_outcome(record: dict[str, Any]) -> str:
    final_answer = record.get("final_answer")
    if final_answer is None or not str(final_answer).strip():
        return "missing_answer"
    prediction = extract_number(final_answer)
    if prediction is None:
        return "non_numeric"
    gold = extract_number((record.get("problem") or {}).get("answer"))
    if gold is None:
        return "non_numeric"
    if prediction * gold < 0 and _relative_error(abs(prediction), abs(gold)) <= 1e-3:
        return "sign_error"
    if prediction == gold:
        return "other_numeric"
    if _relative_error(prediction, gold) <= 1e-3:
        return "rounding_or_precision"
    if abs(abs(prediction - gold) - 1.0) <= 1e-6:
        return "off_by_one"
    if prediction != 0 and gold != 0 and prediction * gold > 0:
        ratio = abs(prediction / gold)
        for factor in SCALE_FACTORS:
            if abs(ratio - factor) / factor <= 1e-3:
                return "scale_or_unit"
            inverse = 1.0 / factor
            if abs(ratio - inverse) / inverse <= 1e-3:
                return "scale_or_unit"
    return "other_numeric"


def classify_jca_process(record: dict[str, Any]) -> str:
    trajectory = record.get("trajectory") or {}
    if (
        record.get("terminated_by") not in (None, "stop")
        or trajectory.get("terminated_by") not in (None, "stop")
        or trajectory.get("error")
    ):
        return "execution_or_protocol_failure"
    steps = trajectory.get("steps", []) or []
    gold = extract_number((record.get("problem") or {}).get("answer"))
    final = extract_number(record.get("final_answer"))
    tentative_values = [
        extract_number(step.get("tentative_answer"))
        for step in steps
        if step.get("tentative_answer") not in (None, "")
    ]
    if any(numeric_match(value, gold) for value in tentative_values) and not numeric_match(final, gold):
        return "harmful_correction"
    active_agents = {
        str(step.get("active_agent")) for step in steps if step.get("active_agent")
    }
    handoffs = sum(bool(step.get("handoff_target")) for step in steps)
    if len(active_agents) < 2 or handoffs == 0:
        return "premature_stop"
    if any(value is not None and not numeric_match(value, gold) for value in tentative_values):
        return "unrepaired_wrong_answer"
    return "other_collaboration_failure"


def analyze(method: str, path: Path, example_limit: int = 5) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    process_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    gold_strata: dict[str, Counter[str]] = {
        key: Counter() for key in GOLD_STRATUM_ORDER
    }
    funnel_counts: Counter[str] = Counter()
    outcome_examples: dict[str, list[str]] = defaultdict(list)
    process_examples: dict[str, list[str]] = defaultdict(list)
    records = list(iter_jsonl(path))
    n_correct = 0
    for record in records:
        correct = _is_correct(record)
        problem_id = str((record.get("problem") or {}).get("id") or "")
        stratum = classify_gold_stratum(record)
        gold_strata[stratum]["n_problems"] += 1
        gold_strata[stratum]["n_correct"] += int(correct)
        final_value = extract_number(record.get("final_answer"))
        if correct:
            execution_counts["correct"] += 1
        elif _has_explicit_error(method, record):
            execution_counts["execution_or_protocol_failure"] += 1
        elif final_value is None:
            execution_counts["missing_or_non_numeric"] += 1
        else:
            execution_counts["wrong_numeric_answer"] += 1

        gold = extract_number((record.get("problem") or {}).get("answer"))
        candidate_values = [
            extract_number(value) for value in intermediate_answers(method, record)
        ]
        valid_candidates = [value for value in candidate_values if value is not None]
        has_correct_candidate = any(
            numeric_match(value, gold) for value in valid_candidates
        )
        funnel_counts["with_parsed_candidate"] += int(bool(valid_candidates))
        funnel_counts["with_correct_candidate"] += int(has_correct_candidate)
        if has_correct_candidate and correct:
            funnel_counts["correct_candidate_retained"] += 1
        elif has_correct_candidate:
            funnel_counts["correct_candidate_lost"] += 1
        elif correct:
            funnel_counts["rescued_without_correct_candidate"] += 1
        else:
            funnel_counts["no_correct_candidate_final_wrong"] += 1

        if correct:
            n_correct += 1
            continue
        outcome = classify_outcome(record)
        outcome_counts[outcome] += 1
        if len(outcome_examples[outcome]) < example_limit:
            outcome_examples[outcome].append(problem_id)
        if method.lower() == "jca":
            process = classify_jca_process(record)
            process_counts[process] += 1
            if len(process_examples[process]) < example_limit:
                process_examples[process].append(problem_id)
    n_wrong = len(records) - n_correct
    pct = lambda count: 100.0 * count / n_wrong if n_wrong else 0.0
    strata_payload = {}
    for key in GOLD_STRATUM_ORDER:
        n_stratum = gold_strata[key]["n_problems"]
        n_stratum_correct = gold_strata[key]["n_correct"]
        strata_payload[key] = {
            "n_problems": n_stratum,
            "n_correct": n_stratum_correct,
            "accuracy": n_stratum_correct / n_stratum if n_stratum else None,
        }
    candidate_oracle = funnel_counts["with_correct_candidate"]
    retained = funnel_counts["correct_candidate_retained"]
    result: dict[str, Any] = {
        "method": method.lower(),
        "path": str(path),
        "n_problems": len(records),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "correct_by_problem": {
            str((record.get("problem") or {}).get("id") or ""): _is_correct(record)
            for record in records
        },
        "execution_counts": {
            "correct": execution_counts["correct"],
            "execution_or_protocol_failure": execution_counts[
                "execution_or_protocol_failure"
            ],
            "missing_or_non_numeric": execution_counts["missing_or_non_numeric"],
            "wrong_numeric_answer": execution_counts["wrong_numeric_answer"],
        },
        "gold_strata": strata_payload,
        "collaboration_funnel": {
            "with_parsed_candidate": funnel_counts["with_parsed_candidate"],
            "with_correct_candidate": candidate_oracle,
            "correct_candidate_retained": retained,
            "correct_candidate_lost": funnel_counts["correct_candidate_lost"],
            "rescued_without_correct_candidate": funnel_counts[
                "rescued_without_correct_candidate"
            ],
            "no_correct_candidate_final_wrong": funnel_counts[
                "no_correct_candidate_final_wrong"
            ],
            "candidate_oracle_accuracy": candidate_oracle / len(records) if records else 0.0,
            "correct_candidate_retention_rate": (
                retained / candidate_oracle if candidate_oracle else None
            ),
        },
        "outcome_counts": {key: outcome_counts[key] for key in OUTCOME_ORDER},
        "outcome_percentages": {key: pct(outcome_counts[key]) for key in OUTCOME_ORDER},
        "outcome_examples": dict(outcome_examples),
    }
    if method.lower() == "jca":
        result["process_counts"] = {key: process_counts[key] for key in PROCESS_ORDER}
        result["process_percentages"] = {key: pct(process_counts[key]) for key in PROCESS_ORDER}
        result["process_examples"] = dict(process_examples)
    return result


def analyze_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure shared easy/hard examples across methods."""
    if not results:
        return {}
    common_ids = set(results[0]["correct_by_problem"])
    if any(set(result["correct_by_problem"]) != common_ids for result in results[1:]):
        raise ValueError("failure comparison requires identical problem IDs")
    distribution: Counter[int] = Counter()
    unique_correct: Counter[str] = Counter()
    for problem_id in common_ids:
        correct_methods = [
            result["method"] for result in results
            if result["correct_by_problem"][problem_id]
        ]
        distribution[len(correct_methods)] += 1
        if len(correct_methods) == 1:
            unique_correct[correct_methods[0]] += 1
    pairwise = []
    for left_index, left in enumerate(results):
        left_wrong = {
            problem_id for problem_id in common_ids
            if not left["correct_by_problem"][problem_id]
        }
        for right in results[left_index + 1:]:
            right_wrong = {
                problem_id for problem_id in common_ids
                if not right["correct_by_problem"][problem_id]
            }
            union = left_wrong | right_wrong
            intersection = left_wrong & right_wrong
            pairwise.append({
                "left": left["method"],
                "right": right["method"],
                "n_shared_wrong": len(intersection),
                "n_union_wrong": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
            })
    n_methods = len(results)
    return {
        "n_problems": len(common_ids),
        "n_methods": n_methods,
        "correct_method_count_distribution": {
            str(count): distribution[count] for count in range(n_methods + 1)
        },
        "all_wrong": distribution[0],
        "all_correct": distribution[n_methods],
        "oracle_correct": len(common_ids) - distribution[0],
        "unique_correct": {
            result["method"]: unique_correct[result["method"]]
            for result in results
        },
        "pairwise_error_overlap": pairwise,
    }


def _display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def render(
    results: list[dict[str, Any]],
    comparison: Optional[dict[str, Any]] = None,
) -> str:
    lines = [
        "### 7.1 按标准答案数值形态拆分的准确率",
        "",
        "GSM-Hard 替换原题数值后会产生负数、极小数和非整数答案；本表直接衡量各方法对这些答案形态的鲁棒性。",
        "",
        "| 标准答案类型 | 题数 | " + " | ".join(_display_name(result["method"]) for result in results) + " |",
        "|---|---:|" + "---:|" * len(results),
    ]
    for key in GOLD_STRATUM_ORDER:
        n_problems = results[0]["gold_strata"][key]["n_problems"]
        values = []
        for result in results:
            row = result["gold_strata"][key]
            values.append(
                f"{row['n_correct']}/{row['n_problems']} ({100 * row['accuracy']:.1f}%)"
                if row["accuracy"] is not None else "—"
            )
        lines.append(f"| {GOLD_STRATUM_LABELS[key]} | {n_problems} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "### 7.2 正确候选到最终答案的协作漏斗",
        "",
        "> 正确候选覆盖率表示任一协议实际接受的中间答案曾命中标准答案；保留率表示出现正确候选后，最终答案仍然正确。",
        "",
        "| 方法 | 有可解析候选 | 正确候选覆盖率 | 最终正确 | 丢失正确候选 | 正确候选保留率 | 无正确候选时救回 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        funnel = result["collaboration_funnel"]
        retention = funnel["correct_candidate_retention_rate"]
        lines.append(
            f"| {_display_name(result['method'])} | {funnel['with_parsed_candidate']}/{result['n_problems']} "
            f"| {funnel['with_correct_candidate']}/{result['n_problems']} ({100 * funnel['candidate_oracle_accuracy']:.1f}%) "
            f"| {result['n_correct']}/{result['n_problems']} | {funnel['correct_candidate_lost']} "
            f"| {'—' if retention is None else f'{100 * retention:.1f}%'} "
            f"| {funnel['rescued_without_correct_candidate']} |"
        )
    lines.extend([
        "",
        "### 7.3 运行、解析与数值错误分离",
        "",
        "| 方法 | 正确 | 明确执行/协议失败 | 无可解析最终答案 | 有数值答案但错误 |",
        "|---|---:|---:|---:|---:|",
    ])
    for result in results:
        counts = result["execution_counts"]
        lines.append(
            f"| {_display_name(result['method'])} | {counts['correct']} "
            f"| {counts['execution_or_protocol_failure']} | {counts['missing_or_non_numeric']} "
            f"| {counts['wrong_numeric_answer']} |"
        )
    if comparison:
        lines.extend([
            "",
            "### 7.4 跨方法共同难度",
            "",
            "| 做对该题的方法数 | 题数 |",
            "|---:|---:|",
        ])
        distribution = comparison["correct_method_count_distribution"]
        for count in range(comparison["n_methods"] + 1):
            lines.append(f"| {count} | {distribution[str(count)]} |")
        lines.extend([
            "",
            f"五种方法的跨方法理论上限为 {comparison['oracle_correct']}/{comparison['n_problems']}；"
            f"其中 {comparison['all_wrong']} 题五种方法全错，{comparison['all_correct']} 题五种方法全对。",
        ])
    lines.extend([
        "",
        "### 7.5 数值关系签名（附录）",
        "",
        "> 这些规则只描述错误预测与标准答案的数值关系，不应解释为建模或算术错误原因。",
        "",
        "| 方法 | 错误题数/总题数 | 缺失答案 | 非数值答案 | 符号错误 | 精度错误 | 差一错误 | 比例或单位错误 | 其他数值错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        counts = result["outcome_counts"]
        lines.append(
            f"| {_display_name(result['method'])} | {result['n_wrong']}/{result['n_problems']} "
            f"| {counts['missing_answer']} | {counts['non_numeric']} "
            f"| {counts['sign_error']} | {counts['rounding_or_precision']} "
            f"| {counts['off_by_one']} | {counts['scale_or_unit']} "
            f"| {counts['other_numeric']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--path", action="append", required=True, type=Path)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if len(args.method) != len(args.path):
        parser.error("--method and --path must be supplied the same number of times")
    results = [
        analyze(method, path, example_limit=args.example_limit)
        for method, path in zip(args.method, args.path)
    ]
    print(render(results, analyze_comparison(results)))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] saved -> {args.json}")


if __name__ == "__main__":
    main()
