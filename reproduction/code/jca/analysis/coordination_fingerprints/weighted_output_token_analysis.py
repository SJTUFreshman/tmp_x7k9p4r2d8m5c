"""统计各 benchmark 正式 eval 的参数加权 output token 成本。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "analysis"
for import_root in (ROOT, ANALYSIS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from analysis.common import count_tokens as musique_count_tokens
from analysis.gsm_hard.common import (
    count_tokens as gsm_count_tokens,
    load_records as load_gsm_records,
)
from analysis.math.common import (
    COMPONENT_LOADERS as MATH_COMPONENT_LOADERS,
    attempt_raw as math_attempt_raw,
    iter_jsonl as iter_math_jsonl,
    math_jca_bootstrap_component,
    model_size as math_model_size,
    token_count as math_count_tokens,
)
from analysis.metric_tokens_by_model_size import (
    _iter_calls as musique_iter_calls,
    _iter_jsonl as iter_jsonl,
)
from analysis.coordination_fingerprints.trajectory_adapters import (
    ProtocolUnit,
    protocol_units,
    raw_attempts,
)
from analysis.coordination_fingerprints.benchmark_paths import (
    CONIFER_AFLOW,
    CONIFER_AFLOW_HARD_SCORE,
    CONIFER_AFLOW_SCORE,
    CONIFER_AGENTVERSE,
    CONIFER_AGENTVERSE_HARD_SCORE,
    CONIFER_AGENTVERSE_SCORE,
    CONIFER_GPTSWARM,
    CONIFER_GPTSWARM_HARD_SCORE,
    CONIFER_GPTSWARM_SCORE,
    CONIFER_JCA,
    CONIFER_JCA_HARD_SCORE,
    CONIFER_JCA_SCORE,
    CONIFER_MAD,
    CONIFER_MAD_HARD_SCORE,
    CONIFER_MAD_SCORE,
    CONIFER_SELF_RL,
    CONIFER_ZERO_SHOT,
    GSM_AFLOW,
    GSM_AGENTVERSE,
    GSM_GPTSWARM,
    GSM_JCA,
    GSM_MAD,
    GSM_ZEROSHOT,
    MATH_AFLOW,
    MATH_AFLOW_SCORE,
    MATH_AGENTVERSE,
    MATH_AGENTVERSE_SCORE,
    MATH_GPTSWARM,
    MATH_GPTSWARM_SCORE,
    MATH_JCA,
    MATH_JCA_SCORE,
    MATH_MAD,
    MATH_MAD_SCORE,
    MATH_ZERO_SHOT,
    MULTIPL_E_AFLOW,
    MULTIPL_E_AGENTVERSE,
    MULTIPL_E_GPTSWARM,
    MULTIPL_E_JCA,
    MULTIPL_E_JCA_SCORE,
    MULTIPL_E_SELF_RL,
    MULTIPL_E_MAD,
    MULTIPL_E_ZERO_SHOT,
    MUSIQUE_AFLOW,
    MUSIQUE_AFLOW_SCORE,
    MUSIQUE_AGENTVERSE,
    MUSIQUE_AGENTVERSE_SCORE,
    MUSIQUE_GPTSWARM,
    MUSIQUE_GPTSWARM_SCORE,
    MUSIQUE_JCA,
    MUSIQUE_MAD,
    MUSIQUE_MAD_SCORE,
    MUSIQUE_ZEROSHOT,
)


MODEL_SIZES = ("1.7B", "4B", "8B", "14B", "unknown")
MODEL_PARAMETERS = {"1.7B": 1.7, "4B": 4.0, "8B": 8.0, "14B": 14.0}
THREE_MODEL_DENOMINATOR = 13.7
FOUR_MODEL_DENOMINATOR = 27.7
TOKEN_COUNTING_VERSION = 3
GPTSWARM_TYPE_TO_SIZE = {
    "IO": "1.7B",
    "CoT": "4B",
    "Debate": "8B",
    "Aggregator": "8B",
}


@dataclass(frozen=True)
class RunSpec:
    benchmark: str
    method: str
    label: str
    path: Path | None
    score: str | None
    note: str
    loader: str
    max_turns: int | None = None
    # 2026-09-13 新增。默认 None = 沿用 _agent_size 的 A1=1.7B/A2=4B/A3=8B 约定，
    # 所有既有 run 行为不变。只有 capacity 被反转的 run（A1 才是 8B）需要显式给出
    # agent→模型规模映射，否则 W3 会把 8B 的 token 当成 1.7B 计权。映射来源是该
    # run 自己的 config.env 里的 MODEL_A1/A2/A3。
    agent_size_map: tuple[tuple[str, str], ...] | None = None


@dataclass
class RunStats:
    benchmark: str
    method: str
    label: str
    path: str
    n_questions: int
    tokens_1_7b: int
    tokens_4b: int
    tokens_8b: int
    tokens_14b: int
    tokens_unknown: int
    raw_output_tokens: int
    weighted_total_w3: float | None
    weighted_mean_w3: float | None
    weighted_total_w4: float | None
    weighted_mean_w4: float | None
    score: str | None
    note: str
    max_turns: int | None = None


def _empty_counts() -> Counter[str]:
    return Counter({size: 0 for size in MODEL_SIZES})


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _agent_size(value: Any, agent_size_map: dict[str, str] | None = None) -> str:
    text = str(value or "")
    lowered = text.lower()
    # 2026-09-13：agent_size_map 优先。槽位名不再必然对应固定模型规模（reversed
    # capacity 的 run 里 A1 是 8B），所以先查显式映射，查不到才退回旧约定。
    if agent_size_map:
        slot = text.split("_", 1)[0]
        if slot in agent_size_map:
            return agent_size_map[slot]
    if text == "A1" or text.startswith("A1_") or "1.7b" in lowered:
        return "1.7B"
    if text == "A2" or text.startswith("A2_") or "4b" in lowered:
        return "4B"
    if text == "A3" or text.startswith("A3_") or "8b" in lowered:
        return "8B"
    if text == "SAS/14B" or "14b" in lowered or "sas" in lowered:
        return "14B"
    return "unknown"


def _multipl_e_size(
    method: str, unit: ProtocolUnit, agent_size_map: dict[str, str] | None = None
) -> str:
    size = _agent_size(unit.agent, agent_size_map)
    if method == "agentverse" and unit.kind in {"recruit", "evaluation"} and size == "unknown":
        return "8B"
    if method == "gptswarm" and size == "unknown":
        size = GPTSWARM_TYPE_TO_SIZE.get(str(unit.payload.get("node_type") or ""), "unknown")
    return size


def _count_musique(path: Path, method: str) -> tuple[int, Counter[str]]:
    counts = _empty_counts()
    n_questions = 0
    loader_method = "jca" if method == "zero-shot" else method.lower()
    for raw in iter_jsonl(path):
        n_questions += 1
        for size, text in musique_iter_calls(loader_method, raw):
            counts[size] += musique_count_tokens(text)
    return n_questions, counts


def _count_gsm(path: Path, method: str) -> tuple[int, Counter[str]]:
    loader_method = "jca" if method == "zero-shot" else method.lower()
    records = load_gsm_records(loader_method, path)
    counts = _empty_counts()
    for record in records:
        for output in record.outputs:
            counts[output.model_size] += gsm_count_tokens(output.text)
    return len(records), counts


def _multipl_e_raw_attempts(record: dict[str, Any]) -> Iterable[Any]:
    raw_responses = record.get("raw_responses")
    if isinstance(raw_responses, list):
        return raw_responses
    for key in ("raw_response", "response"):
        if record.get(key) not in (None, ""):
            return [record[key]]
    return ()


def _count_multipl_e(
    path: Path, method: str, agent_size_map: dict[str, str] | None = None
) -> tuple[int, Counter[str]]:
    counts = _empty_counts()
    n_questions = 0
    normalized = method.lower()
    if normalized == "self-rl":
        for raw in _iter_records(path):
            n_questions += 1
            for output in _multipl_e_raw_attempts(raw):
                counts["14B"] += musique_count_tokens(str(output))
        return n_questions, counts

    protocol_method = "jca" if normalized == "zero-shot" else normalized
    for raw in _iter_records(path):
        n_questions += 1
        for unit in protocol_units(protocol_method, raw):
            size = _multipl_e_size(protocol_method, unit, agent_size_map)
            attempts = raw_attempts(unit.payload)
            if attempts is None:
                continue
            for output in attempts:
                counts[size] += musique_count_tokens(str(output))
    return n_questions, counts


def _conifer_agent_size_map(record: dict[str, Any]) -> dict[str, str] | None:
    """Resolve A1/A2/A3 -> model size from the run's own `sampling.model_paths`.

    Conifer 的每条记录都自带 `sampling.model_paths`（A1/A2/A3 的真实 checkpoint 路径），
    所以不必像 MultiPL-E 那样靠外部 config.env 传 agent_size_map，也就不存在
    reversed-capacity 漏传导致 8B token 记进 1.7B 的那类风险。
    """
    sampling = record.get("sampling")
    if not isinstance(sampling, dict):
        return None
    paths = sampling.get("model_paths")
    if not isinstance(paths, dict):
        return None
    mapping: dict[str, str] = {}
    for slot, model_path in paths.items():
        size = _agent_size(model_path)
        if size != "unknown":
            mapping[str(slot)] = size
    return mapping or None


def _count_conifer(path: Path, method: str) -> tuple[int, Counter[str]]:
    """Count Conifer output tokens.

    Conifer 的六条 arm 全部由 conifer_training_hub/03_rollout/run_conifer_mas.py 落成
    同一套 MAS trajectory schema，所以统一走 protocol_units 的 "jca" 分支，不按
    baseline 分派 loader。两个 Conifer 专属细节：

    - MAD 每题固定 10 个 step，其中第 10 个是合成的 `final_step`，`raw_output` 为空，
      不对应任何模型生成。`raw_attempts` 对空串返回 None，这里直接跳过，因此
      MAD 的 token 只来自前 9 个真实 step。
    - 只有 AFlow（走 baseline_queue 链路）的 step 带 `raw_outputs` 列表，重试输出会被
      完整计入；其余四条只有单个 `raw_output`，落盘时就没保留重试，token 是下界。
    """
    counts = _empty_counts()
    n_questions = 0
    for raw in _iter_records(path):
        n_questions += 1
        size_map = _conifer_agent_size_map(raw)
        for unit in protocol_units("jca", raw):
            size = _agent_size(unit.agent, size_map)
            attempts = raw_attempts(unit.payload)
            if attempts is None:
                continue
            for output in attempts:
                counts[size] += musique_count_tokens(str(output))
    return n_questions, counts


def _count_math(
    path: Path, method: str, max_turns: int | None = None
) -> tuple[int, Counter[str]]:
    if max_turns is not None and max_turns < 0:
        raise ValueError("max_turns must be non-negative")
    counts = _empty_counts()
    n_questions = 0
    normalized_method = method.lower()
    loader_method = "jca" if normalized_method in {"jca", "zero-shot", "zeroshot"} else normalized_method
    loader = MATH_COMPONENT_LOADERS[loader_method]
    for raw in iter_math_jsonl(path):
        n_questions += 1
        if normalized_method in {"jca", "zero-shot", "zeroshot"}:
            bootstrap = math_jca_bootstrap_component(raw)
            if bootstrap is not None:
                model, attempts = bootstrap
                size = math_model_size(model)
                for attempt in attempts:
                    counts[size] += math_count_tokens(math_attempt_raw(attempt))
        components, _, _, _ = loader(raw)
        if max_turns is not None:
            components = components[:max_turns]
        for model, attempts in components:
            size = math_model_size(model)
            for attempt in attempts:
                counts[size] += math_count_tokens(math_attempt_raw(attempt))
    return n_questions, counts


def _weighted(counts: Counter[str], denominator: float) -> float | None:
    if counts["unknown"]:
        return None
    if denominator == THREE_MODEL_DENOMINATOR and counts["14B"]:
        return None
    sizes = (
        ("1.7B", "4B", "8B")
        if denominator == THREE_MODEL_DENOMINATOR
        else tuple(MODEL_PARAMETERS)
    )
    return sum(
        counts[size] * MODEL_PARAMETERS[size] / denominator for size in sizes
    )


def analyze_run(spec: RunSpec) -> RunStats:
    if spec.path is None:
        raise ValueError(f"pending run cannot be analyzed: {spec.benchmark}/{spec.method}")
    if not spec.path.is_file():
        raise FileNotFoundError(f"{spec.benchmark}/{spec.method}: {spec.path}")

    if spec.loader == "musique":
        n_questions, counts = _count_musique(spec.path, spec.method)
    elif spec.loader == "gsm":
        n_questions, counts = _count_gsm(spec.path, spec.method)
    elif spec.loader == "multipl-e":
        n_questions, counts = _count_multipl_e(
            spec.path, spec.method, dict(spec.agent_size_map) if spec.agent_size_map else None
        )
    elif spec.loader == "math":
        n_questions, counts = _count_math(spec.path, spec.method, spec.max_turns)
    elif spec.loader == "conifer":
        n_questions, counts = _count_conifer(spec.path, spec.method)
    else:
        raise ValueError(f"unknown loader: {spec.loader}")

    raw_total = sum(counts.values())
    weighted_w3 = _weighted(counts, THREE_MODEL_DENOMINATOR)
    weighted_w4 = _weighted(counts, FOUR_MODEL_DENOMINATOR)
    return RunStats(
        benchmark=spec.benchmark,
        method=spec.method,
        label=spec.label,
        path=str(spec.path),
        n_questions=n_questions,
        tokens_1_7b=counts["1.7B"],
        tokens_4b=counts["4B"],
        tokens_8b=counts["8B"],
        tokens_14b=counts["14B"],
        tokens_unknown=counts["unknown"],
        raw_output_tokens=raw_total,
        weighted_total_w3=weighted_w3,
        weighted_mean_w3=weighted_w3 / n_questions if weighted_w3 is not None and n_questions else None,
        weighted_total_w4=weighted_w4,
        weighted_mean_w4=weighted_w4 / n_questions if weighted_w4 is not None and n_questions else None,
        score=spec.score,
        note=spec.note,
        max_turns=spec.max_turns,
    )


def _fmt_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "不适用"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _fmt_path(path: str) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return path


def render_report(stats: list[RunStats], pending: list[RunSpec], output_path: Path) -> str:
    lines = [
        "# 各 Benchmark 参数加权 Output Token 分析",
        "",
        "本报告统计已保存正式 eval 中每个模型产生的完整 assistant output token，并给出整个 eval 的参数加权总量与平均每题加权量。统计不重新运行模型。",
        "",
        "## 统计口径",
        "",
        "- 三模型主口径为 `W3 = 1.7/13.7 × T1.7B + 4/13.7 × T4B + 8/13.7 × T8B`，与此前 MuSiQue、GSM-Hard、MultiPL-E 和 MATH 报告一致。",
        "- `T` 统计日志中所有持久化的完整 raw assistant output；同一逻辑调用的 parser/transport 重试也计入。prompt token、离线 evaluator replay 和 AFlow workflow search 不计入。",
        "- “参数加权总量”是整个 eval 的总量；“平均每题”除以该结果文件中的题目数，失败题仍在分母中。",
        "- MultiPL-E SAS self-RL 是 14B-only，主表的 W3 不适用；其 14B token 和独立的 W4 补充值单列，不能直接与 W3 数值横向比较。",
        "- MATH zero-shot 与 JCA 都先计每题一次独立 8B bootstrap solver（包括 bootstrap retry）；zero-shot 再计完整 trajectory，JCA 再计前 3 个 trajectory turn。turn 内 retry 仍计入，题目数分母保持 500。强制 A3 格式转换是首个 trajectory turn，不与 bootstrap solver 重复。",
        "",
        "## 汇总结果",
        "",
        "| 数据集 | 方法 | 题目数 | 得分 | 1.7B token | 4B token | 8B token | 14B token | raw output 总量 | 参数加权总量 W3 | 平均每题 W3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item.benchmark} | {item.method} | {item.n_questions:,} | {item.score or '—'} "
            f"| {_fmt_number(item.tokens_1_7b)} | {_fmt_number(item.tokens_4b)} "
            f"| {_fmt_number(item.tokens_8b)} | {_fmt_number(item.tokens_14b)} "
            f"| {_fmt_number(item.raw_output_tokens)} "
            f"| {_fmt_number(item.weighted_total_w3)} | {_fmt_number(item.weighted_mean_w3)} |"
        )

    self_rl = [item for item in stats if item.method == "self-RL" and item.tokens_14b]
    if self_rl:
        lines.extend([
            "",
            "## 14B-only 补充",
            "",
            "为完整呈现 SAS self-RL，下面给出 `W4 = 14/27.7 × T14B` 的独立补充值。W4 与主表 W3 使用不同归一化分母，仅用于说明 14B-only eval 的规模。",
            "",
            "| 数据集 | 方法 | 14B raw token | W4 参数加权总量 | W4 平均每题 |",
            "|---|---|---:|---:|---:|",
        ])
        for item in self_rl:
            lines.append(
                f"| {item.benchmark} | {item.method} | {_fmt_number(item.tokens_14b)} "
                f"| {_fmt_number(item.weighted_total_w4)} | {_fmt_number(item.weighted_mean_w4)} |"
            )

    lines.extend([
        "",
        "## 结果来源与注意事项",
        "",
        "| 数据集 | 方法 | 采用的结果文件 | 备注 |",
        "|---|---|---|---|",
    ])
    for item in stats:
        note = item.note or "—"
        lines.append(f"| {item.benchmark} | {item.method} | `{_fmt_path(item.path)}` | {note} |")

    lines.extend([
        "",
        "- MuSiQue AFlow 自 2026-09-14 起改用 `070_musique_aflow_repeat5` 官方 5-roll 中 EM 最高的 `run_01_20260911_170029`（1011/2417，41.83%）：该 run 先 `MODE=search` 搜出 `round_07_proposed.py` 再用它部署，search 与 deployment 同源。此前为 `aflow_hetero_search20_dev20_full2417.jsonl`（1019/2417，42.16%），其 search 响应数只能从 vLLM access log 反推；更早的 44.19% 重搜版本一直未采用。新 workflow 每题 18.911 个 op，旧的为 7.000。",
        "- GSM-Hard JCA 和 zero-shot 采用 `thinking_hidden` eval 日志；日志中的 raw output 只保存可见响应，hidden thinking 已不可恢复，因此这两行是可观测 output token 的下界，不能与包含完整 thinking 的方法作严格成本比较。",
        "- MATH 与 MultiPL-E baseline 的模型归属从各自 trajectory 中的 `agent/model/caller_id` 读取；AgentVerse 的 recruiter/evaluator 按该运行的 Meta/A3 路由计入 8B。",
        "- MATH 的 zero-shot 与四个 baseline 使用 no-thinking 运行；baseline 使用同一 `shard_04` 的 500 题，本报告的 deployment token 不再读取旧的 thinking-enabled 5000 题日志。其中 GPTSwarm 与 AFlow 于 2026-09-11 重跑：GPTSwarm 的 `TEMPERATURE` 由 0.0 对齐到 0.7，AFlow 改为 `MODE=both`、自带 no-thinking search，deployment 评的就是该 search 的产物。",
    ])
    if pending:
        lines.extend(["", "## 尚未可统计的正式 eval", ""])
        for spec in pending:
            reason = spec.note or "结果文件尚未形成"
            lines.append(f"- {spec.benchmark} / {spec.method}：{reason}")

    lines.extend([
        "",
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/weighted_output_token_analysis.py",
        "```",
        "",
        f"JSON 明细：`{output_path.with_suffix('.json')}`",
        "",
    ])
    return "\n".join(lines)


def _default_specs() -> list[RunSpec]:
    return [
        RunSpec("MuSiQue", "JCA", "JCA（正式运行）", MUSIQUE_JCA, "1024/2417", "—", "musique"),
        RunSpec("MuSiQue", "zero-shot", "训练前异构 MAS", MUSIQUE_ZEROSHOT, "514/2417", "—", "musique"),
        RunSpec("MuSiQue", "MAD", "MAD training-free（2026-09-11 重跑）", MUSIQUE_MAD, MUSIQUE_MAD_SCORE, "带完整 raw_outputs ledger", "musique"),
        RunSpec("MuSiQue", "AgentVerse", "AgentVerse training-free（2026-09-11 重跑）", MUSIQUE_AGENTVERSE, MUSIQUE_AGENTVERSE_SCORE, "带完整 raw_outputs ledger", "musique"),
        RunSpec("MuSiQue", "AFlow", "AFlow training-free（2026-09-11 重跑）", MUSIQUE_AFLOW, MUSIQUE_AFLOW_SCORE, "MODE=search 自带 no-thinking search，与 deployment 同源；两端都有 raw_outputs ledger", "musique"),
        RunSpec("MuSiQue", "GPTSwarm", "GPTSwarm training-free（2026-09-11 重跑）", MUSIQUE_GPTSWARM, MUSIQUE_GPTSWARM_SCORE, "带完整 raw_outputs ledger", "musique"),
        RunSpec("GSM-Hard", "JCA", "JCA（正式运行，seed=43）", GSM_JCA, "95/132", "仅保存可见 raw output", "gsm"),
        RunSpec("GSM-Hard", "zero-shot", "训练前异构 MAS", GSM_ZEROSHOT, "91/132", "仅保存可见 raw output", "gsm"),
        RunSpec("GSM-Hard", "MAD", "MAD training-free", GSM_MAD, "88/132", "—", "gsm"),
        RunSpec("GSM-Hard", "AgentVerse", "AgentVerse training-free", GSM_AGENTVERSE, "95/132", "—", "gsm"),
        RunSpec("GSM-Hard", "AFlow", "AFlow training-free", GSM_AFLOW, "90/132", "—", "gsm"),
        RunSpec("GSM-Hard", "GPTSwarm", "GPTSwarm training-free", GSM_GPTSWARM, "97/132", "—", "gsm"),
        RunSpec("MultiPL-E-8Lang", "zero-shot", "异构 zero-shot MAS", MULTIPL_E_ZERO_SHOT, "569/1352", "—", "multipl-e"),
        RunSpec("MultiPL-E-8Lang", "MAD", "MAD training-free", MULTIPL_E_MAD, "744/1352", "—", "multipl-e"),
        RunSpec("MultiPL-E-8Lang", "AgentVerse", "AgentVerse training-free（repeat-5 run_03）", MULTIPL_E_AGENTVERSE, "801/1352", "—", "multipl-e"),
        RunSpec("MultiPL-E-8Lang", "GPTSwarm", "GPTSwarm training-free", MULTIPL_E_GPTSWARM, "698/1352", "—", "multipl-e"),
        RunSpec("MultiPL-E-8Lang", "AFlow", "AFlow training-free", MULTIPL_E_AFLOW, "579/1352", "—", "multipl-e"),
        RunSpec("MultiPL-E-8Lang", "self-RL", "SAS self-RL normalized v2", MULTIPL_E_SELF_RL, "817/1352", "14B-only one-shot", "multipl-e"),
        RunSpec("MATH", "MAD", "MAD training-free", MATH_MAD, MATH_MAD_SCORE, "no-thinking；共同 shard_04（500 题）", "math"),
        RunSpec("MATH", "AgentVerse", "AgentVerse training-free", MATH_AGENTVERSE, MATH_AGENTVERSE_SCORE, "no-thinking；共同 shard_04（500 题）", "math"),
        RunSpec("MATH", "GPTSwarm", "GPTSwarm training-free（2026-09-11 重跑）", MATH_GPTSWARM, MATH_GPTSWARM_SCORE, "no-thinking；共同 shard_04（500 题）；TEMPERATURE=0.7，与其余数据集对齐", "math"),
        RunSpec("MATH", "AFlow", "AFlow training-free（2026-09-11 重跑）", MATH_AFLOW, MATH_AFLOW_SCORE, "no-thinking；共同 shard_04（500 题）；MODE=both，评的是本 run 自带的 no-thinking search 产物", "math"),
        RunSpec("MATH", "zero-shot", "训练前异构 MAS", MATH_ZERO_SHOT, "315/500", "no-thinking；裸模型 + 零初始化 adapter；1.7B+4B+8B；成本包含独立 8B bootstrap solver + 完整 trajectory；500 题", "math"),
        # 2026-09-12 前此条无路径，原值逐字为：
        #   RunSpec("MultiPL-E-8Lang", "JCA", "最终 JCA 训练产物", None, None, "最终训练/eval 尚未完成", "multipl-e"),
        # agent_size_map 必须给：该 run 的 config.env 是 MODEL_A1=Qwen3-8B /
        # MODEL_A2=Qwen3-4B / MODEL_A3=Qwen3-1.7B。不给的话 _agent_size 会按默认
        # 约定把 A1 当 1.7B、A3 当 8B，W3 会严重偏低（107,415 而非 203,493）。
        RunSpec("MultiPL-E-8Lang", "JCA", "最终 JCA 训练产物（mpe0912 job 120 / r08）", MULTIPL_E_JCA, MULTIPL_E_JCA_SCORE, "mpe0912 job 120 / r08；reversed capacity（A1=8B/A2=4B/A3=1.7B）；best-of-10 roll，同 config 10 roll weighted 0.5821-0.5939", "multipl-e", agent_size_map=(("A1", "8B"), ("A2", "4B"), ("A3", "1.7B"))),
        RunSpec("MATH", "JCA", "JCA 正式 eval（三-turn 预算回退）", MATH_JCA, MATH_JCA_SCORE, "no-thinking；共同 shard_04；超过三 turn或第三 turn 后仍未成功 stop 时用首个 tentative_answer；成本包含独立 8B bootstrap solver + 前 3 个 trajectory turn", "math", 3),
        RunSpec("MATH", "self-RL", "SAS self-RL", None, None, "尚无登记为同口径可比的正式结果", "math"),
        # 2026-09-14 新增 Conifer。六条 arm 共用同一套 MAS trajectory schema，所以
        # loader 只有一个 "conifer"，不按 baseline 分派。得分列是二值口径
        # all_explicit_passed/1402（Conifer 没有 EM），连续 hard_score 写在 note 里。
        # 模型归属由每条记录自带的 sampling.model_paths 解析，不需要 agent_size_map。
        RunSpec("Conifer", "JCA", "JCA 正式 eval（forever-loop round_4）", CONIFER_JCA, CONIFER_JCA_SCORE, f"hard_score {CONIFER_JCA_HARD_SCORE}；用户指定的 coverage 88.90 / explicit 95.90 一轮，也是 111 轮里 hard 最高的；**step 只有单个 raw_output，重试输出未落盘，token 是下界**", "conifer"),
        RunSpec("Conifer", "MAD", "MAD training-free（four-arm round_27）", CONIFER_MAD, CONIFER_MAD_SCORE, f"hard_score {CONIFER_MAD_HARD_SCORE}；固定 10 step，第 10 步是 raw_output 为空的合成 final_step、不计 token；重试输出未落盘，token 是下界", "conifer"),
        RunSpec("Conifer", "AgentVerse", "AgentVerse training-free（four-arm round_27）", CONIFER_AGENTVERSE, CONIFER_AGENTVERSE_SCORE, f"hard_score {CONIFER_AGENTVERSE_HARD_SCORE}；step 数 4/8/12 三档（Conifer 的 AgentVerse 没有独立 recruiter 调用，不是 5/9/13）；末尾合成 confirm_stop step 的 raw_output 为空、不计 token；重试输出未落盘，token 是下界", "conifer"),
        RunSpec("Conifer", "GPTSwarm", "GPTSwarm training-free（four-arm round_27）", CONIFER_GPTSWARM, CONIFER_GPTSWARM_SCORE, f"hard_score {CONIFER_GPTSWARM_HARD_SCORE}；固定 7 node；重试输出未落盘，token 是下界", "conifer"),
        RunSpec("Conifer", "AFlow", "AFlow training-free（130 批次 r04）", CONIFER_AFLOW, CONIFER_AFLOW_SCORE, f"hard_score {CONIFER_AFLOW_HARD_SCORE}（5 roll 0.94440 ± 0.00070，取最高）；固定 6 op；**Conifer 唯一带完整 raw_outputs ledger 的一条**，重试 token 已计入", "conifer"),
        RunSpec("Conifer", "self-RL", "SAS 14B self-RL", CONIFER_SELF_RL, None, "27 轮分数俱全（hard 0.93931 ± 0.00085）但没有任何一轮的 trajectories.jsonl 留在盘上，无法统计 token", "conifer"),
        RunSpec("Conifer", "zero-shot", "异构 zero-shot MAS", CONIFER_ZERO_SHOT, None, "10_outputs 下 5 个 *_conifer_zero_shot_mas 目录全部为空，无可用产物", "conifer"),
    ]


def _apply_overrides(specs: list[RunSpec], args: argparse.Namespace) -> list[RunSpec]:
    override_names = {
        ("MuSiQue", "JCA"): "musique_jca",
        ("MuSiQue", "zero-shot"): "musique_zeroshot",
        ("MuSiQue", "MAD"): "musique_mad",
        ("MuSiQue", "AgentVerse"): "musique_agentverse",
        ("MuSiQue", "AFlow"): "musique_aflow",
        ("MuSiQue", "GPTSwarm"): "musique_gptswarm",
        ("GSM-Hard", "JCA"): "gsm_jca",
        ("GSM-Hard", "zero-shot"): "gsm_zeroshot",
        ("GSM-Hard", "MAD"): "gsm_mad",
        ("GSM-Hard", "AgentVerse"): "gsm_agentverse",
        ("GSM-Hard", "AFlow"): "gsm_aflow",
        ("GSM-Hard", "GPTSwarm"): "gsm_gptswarm",
        ("MultiPL-E-8Lang", "zero-shot"): "multipl_e_zero_shot",
        ("MultiPL-E-8Lang", "MAD"): "multipl_e_mad",
        ("MultiPL-E-8Lang", "AgentVerse"): "multipl_e_agentverse",
        ("MultiPL-E-8Lang", "GPTSwarm"): "multipl_e_gptswarm",
        ("MultiPL-E-8Lang", "AFlow"): "multipl_e_aflow",
        ("MultiPL-E-8Lang", "self-RL"): "multipl_e_self_rl",
        ("MultiPL-E-8Lang", "JCA"): "multipl_e_jca",
        ("MATH", "MAD"): "math_mad",
        ("MATH", "AgentVerse"): "math_agentverse",
        ("MATH", "GPTSwarm"): "math_gptswarm",
        ("MATH", "AFlow"): "math_aflow",
        ("MATH", "zero-shot"): "math_zero_shot",
        ("MATH", "JCA"): "math_jca",
    }
    updated = []
    for spec in specs:
        name = override_names.get((spec.benchmark, spec.method))
        override = getattr(args, name) if name else None
        updated.append(
            RunSpec(
                spec.benchmark,
                spec.method,
                spec.label,
                override or spec.path,
                spec.score,
                spec.note,
                spec.loader,
                spec.max_turns,
                # 2026-09-13：这里原本到 max_turns 为止。RunSpec 是逐字段重建的，
                # 漏掉新字段不会报错，只会被静默重置为 None——reversed capacity 的
                # agent→规模映射就是这样丢掉的。新增字段务必同步加到这里。
                spec.agent_size_map,
            )
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = _default_specs()
    override_names = sorted({
        "musique_jca", "musique_zeroshot", "musique_mad", "musique_agentverse", "musique_aflow", "musique_gptswarm",
        "gsm_jca", "gsm_zeroshot", "gsm_mad", "gsm_agentverse", "gsm_aflow", "gsm_gptswarm",
        "multipl_e_zero_shot", "multipl_e_mad", "multipl_e_agentverse", "multipl_e_gptswarm", "multipl_e_aflow", "multipl_e_self_rl", "multipl_e_jca",
        "math_mad", "math_agentverse", "math_gptswarm", "math_aflow", "math_zero_shot", "math_jca",
    })
    default_by_name = {
        "musique_jca": MUSIQUE_JCA, "musique_zeroshot": MUSIQUE_ZEROSHOT, "musique_mad": MUSIQUE_MAD,
        "musique_agentverse": MUSIQUE_AGENTVERSE, "musique_aflow": MUSIQUE_AFLOW, "musique_gptswarm": MUSIQUE_GPTSWARM,
        "gsm_jca": GSM_JCA, "gsm_zeroshot": GSM_ZEROSHOT, "gsm_mad": GSM_MAD, "gsm_agentverse": GSM_AGENTVERSE,
        "gsm_aflow": GSM_AFLOW, "gsm_gptswarm": GSM_GPTSWARM,
        "multipl_e_zero_shot": MULTIPL_E_ZERO_SHOT, "multipl_e_mad": MULTIPL_E_MAD,
        "multipl_e_agentverse": MULTIPL_E_AGENTVERSE, "multipl_e_gptswarm": MULTIPL_E_GPTSWARM,
        "multipl_e_aflow": MULTIPL_E_AFLOW, "multipl_e_self_rl": MULTIPL_E_SELF_RL,
        "multipl_e_jca": MULTIPL_E_JCA,
        "math_mad": MATH_MAD, "math_agentverse": MATH_AGENTVERSE, "math_gptswarm": MATH_GPTSWARM, "math_aflow": MATH_AFLOW,
        "math_zero_shot": MATH_ZERO_SHOT, "math_jca": MATH_JCA,
    }
    for name in override_names:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default_by_name[name])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "weighted_output_token_report.md",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    specs = _apply_overrides(defaults, args)
    stats: list[RunStats] = []
    pending: list[RunSpec] = []
    for spec in specs:
        if spec.path is None:
            pending.append(spec)
            continue
        print(f"[count] {spec.benchmark} / {spec.method}", flush=True)
        stats.append(analyze_run(spec))
    stats.sort(key=lambda item: (item.benchmark, item.method))
    output_path = args.output
    json_path = args.json or output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(stats, pending, output_path), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "token_counting_version": TOKEN_COUNTING_VERSION,
                "formula_w3": "1.7/13.7*T1.7B + 4/13.7*T4B + 8/13.7*T8B",
                "formula_w4": "1.7/27.7*T1.7B + 4/27.7*T4B + 8/27.7*T8B + 14/27.7*T14B",
                "runs": [asdict(item) for item in stats],
                "pending": [asdict(item) for item in pending],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[report] {output_path}")
    print(f"[json] {json_path}")


if __name__ == "__main__":
    main()
