"""Summarize trajectory turn counts across all available benchmark runs.

One turn is one persisted protocol unit, so a chain such as
``A1 -> A2 -> A3`` has three turns. The primary mean includes every eval
record; records without a protocol unit therefore contribute zero turns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.math.common import MATH_JCA_TURN_BUDGET

try:
    from .benchmark_paths import (
        CONIFER_AFLOW,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_JCA,
        CONIFER_MAD,
        GSM_JCA,
        GSM_ZEROSHOT,
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
        MUSIQUE_JCA,
        MUSIQUE_ZEROSHOT,
    )
    from .trajectory_adapters import protocol_units, raw_attempts, unit_chain
except ImportError:
    from benchmark_paths import (
        CONIFER_AFLOW,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_JCA,
        CONIFER_MAD,
        GSM_JCA,
        GSM_ZEROSHOT,
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
        MUSIQUE_JCA,
        MUSIQUE_ZEROSHOT,
    )
    from trajectory_adapters import protocol_units, raw_attempts, unit_chain


DEFAULT_MUSIQUE_PATH = MUSIQUE_JCA
DEFAULT_MUSIQUE_ZEROSHOT_PATH = MUSIQUE_ZEROSHOT
DEFAULT_GSM_PATH = GSM_JCA
DEFAULT_GSM_ZEROSHOT_PATH = GSM_ZEROSHOT
DEFAULT_MULTIPL_E_ZERO_SHOT_PATH = MULTIPL_E_ZERO_SHOT
DEFAULT_MULTIPL_E_JCA_PATH = MULTIPL_E_JCA
DEFAULT_MULTIPL_E_MAD_PATH = MULTIPL_E_MAD
DEFAULT_MULTIPL_E_AGENTVERSE_PATH = MULTIPL_E_AGENTVERSE
DEFAULT_MULTIPL_E_GPTSWARM_PATH = MULTIPL_E_GPTSWARM
DEFAULT_MULTIPL_E_AFLOW_PATH = MULTIPL_E_AFLOW
DEFAULT_MULTIPL_E_SELF_RL_PATH = MULTIPL_E_SELF_RL
DEFAULT_MATH_MAD_PATH = MATH_MAD
DEFAULT_MATH_AGENTVERSE_PATH = MATH_AGENTVERSE
DEFAULT_MATH_GPTSWARM_PATH = MATH_GPTSWARM
DEFAULT_MATH_AFLOW_PATH = MATH_AFLOW
DEFAULT_MATH_ZERO_SHOT_PATH = MATH_ZERO_SHOT
DEFAULT_MATH_JCA_PATH = MATH_JCA
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "turn_count_report.md"


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


def trajectory_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = record.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    return [step for step in steps if isinstance(step, dict)]


@dataclass
class TurnStats:
    dataset: str
    label: str
    path: Path
    n_records: int = 0
    n_with_steps: int = 0
    total_turns: int = 0
    turn_counts: Counter[int] = field(default_factory=Counter)
    chain_counts: Counter[str] = field(default_factory=Counter)
    unit_limit: int | None = None
    raw_n_with_steps: int = 0
    raw_total_turns: int = 0
    raw_turn_counts: Counter[int] = field(default_factory=Counter)
    raw_chain_counts: Counter[str] = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        return self.n_with_steps / self.n_records if self.n_records else 0.0

    @property
    def average_turns(self) -> float:
        return self.total_turns / self.n_records if self.n_records else 0.0

    @property
    def average_turns_with_steps(self) -> float:
        return self.total_turns / self.n_with_steps if self.n_with_steps else 0.0

    @property
    def raw_average_turns(self) -> float:
        return self.raw_total_turns / self.n_records if self.n_records else 0.0

    @property
    def raw_average_turns_with_steps(self) -> float:
        return (
            self.raw_total_turns / self.raw_n_with_steps
            if self.raw_n_with_steps
            else 0.0
        )

    @property
    def min_turns(self) -> int:
        return min(self.turn_counts) if self.turn_counts else 0

    @property
    def max_turns(self) -> int:
        return max(self.turn_counts) if self.turn_counts else 0

    @property
    def raw_min_turns(self) -> int:
        return min(self.raw_turn_counts) if self.raw_turn_counts else 0

    @property
    def raw_max_turns(self) -> int:
        return max(self.raw_turn_counts) if self.raw_turn_counts else 0


def analyze(
    dataset: str,
    label: str,
    path: Path,
    method: str = "JCA",
    max_units: int | None = None,
    # 2026-09-14：只保留真正产生过模型生成的 protocol unit。Conifer 的 MAD 与
    # AgentVerse 每题末尾都有一个 harness 自己写的 confirm_stop step（raw_output 为
    # 空串，不对应任何请求）；它不是一个 turn，留着会把 MAD 记成 10 turn 而不是 9。
    require_generation: bool = False,
) -> TurnStats:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} input does not exist: {path}")
    if max_units is not None and max_units < 0:
        raise ValueError("max_units must be non-negative")

    stats = TurnStats(dataset=dataset, label=label, path=path, unit_limit=max_units)
    for record in iter_jsonl(path):
        stats.n_records += 1
        raw_units = protocol_units(method, record)
        if require_generation:
            raw_units = [
                unit for unit in raw_units if raw_attempts(unit.payload) is not None
            ]
        raw_chain = [unit.agent for unit in raw_units]
        raw_turns = len(raw_units)
        if raw_turns:
            stats.raw_n_with_steps += 1
            stats.raw_total_turns += raw_turns
            stats.raw_turn_counts[raw_turns] += 1
            stats.raw_chain_counts[" -> ".join(raw_chain)] += 1

        units = raw_units if max_units is None else raw_units[:max_units]
        chain_values = [unit.agent for unit in units]
        n_turns = len(units)
        if not n_turns:
            continue

        stats.n_with_steps += 1
        stats.total_turns += n_turns
        stats.turn_counts[n_turns] += 1
        chain = " -> ".join(chain_values)
        stats.chain_counts[chain] += 1
    return stats


def fmt_percent(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.1%}"


def render_report(runs: list[TurnStats], output_path: Path) -> str:
    lines = [
        "# 各 Benchmark 调用链路 Turn 数分析",
        "",
        "本报告统计所有已有正式结果的调用链路长度。一个已执行的协议模型单元计为一个 turn；例如 `A1 -> A2 -> A3` 记为 3 turns。MATH JCA 主指标采用与 78.40% EM 一致的前三个落盘 turn，预算后的完整 raw 轨迹只作审计。",
        "主平均值以正式 eval 的全部题目为分母；没有协议单元的记录贡献 0 turn。另列非空链路均值与覆盖率，避免隐藏失败记录。",
        "2026-09-14 加入 Conifer（5 行）。它的五个 arm 都由 `run_conifer_mas.py` 产出、落同一套 MAS trajectory，因此 turn 的定义与其余数据集一致；但 MAD 与 AgentVerse 每题末尾各有一个 harness 自己写的 `confirm_stop` step（`raw_output` 为空串，不对应任何模型请求），已不计为 turn——否则 MAD 会记成 10 turn 而不是协议规定的 9。另外 Conifer 的 AgentVerse 没有单独的 recruiter 调用，每轮是 3 个 expert 加 1 次 evaluator，所以 turn 数落在 4 / 8 / 12 而不是另外四个数据集的 5 / 9 / 13。",
        "",
        "## 汇总",
        "",
        "| 数据集 | 运行 | 记录数 | 非空链路 | 覆盖率 | 总有效 turn | 平均 turn/全部题 | 平均 turn/非空链路 | 最短 | 最长 | 主口径 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run.dataset} | {run.label} | {run.n_records} | {run.n_with_steps} "
            f"| {fmt_percent(run.n_with_steps, run.n_records)} | {run.total_turns} "
            f"| {run.average_turns:.3f} | {run.average_turns_with_steps:.3f} "
            f"| {run.min_turns} | {run.max_turns} "
            f"| {'前 ' + str(run.unit_limit) + ' 个 turn' if run.unit_limit is not None else '完整轨迹'} |"
        )

    lines.extend([
        "",
        "论文主口径的平均 turn 数为 `总有效 turn / 正式 eval 全部题目数`；非空链路均值仅用于审计。MATH zero-shot 与 JCA 的独立 bootstrap solver 都不是协议 turn；其后的强制 A3 格式转换确实执行了一次模型生成并落为协议 step，因此计作第 1 个 turn，但在 Agent 行为概率报告中不视为自由策略动作。JCA 额外应用前三-turn窗口，zero-shot 使用完整轨迹。",
    ])

    for run in runs:
        lines.extend([
            "",
            f"## {run.dataset} / {run.label}",
            "",
            f"输入文件：`{run.path}`  ",
            f"记录数：{run.n_records}；非空链路：{run.n_with_steps}；覆盖率：{run.coverage:.2%}；平均 turn/全部题：{run.average_turns:.3f}；平均 turn/非空链路：{run.average_turns_with_steps:.3f}",
            "",
            "### Turn 数分布",
            "",
            "| Turn 数 | 记录数 | 占有轨迹记录比例 |",
            "|---:|---:|---:|",
        ])
        for turn_count, frequency in sorted(run.turn_counts.items()):
            lines.append(
                f"| {turn_count} | {frequency} | {fmt_percent(frequency, run.n_with_steps)} |"
            )

        lines.extend([
            "",
            "### 最常见调用链路（前 10）",
            "",
            "| 调用链路 | 记录数 | 占有轨迹记录比例 |",
            "|---|---:|---:|",
        ])
        for chain, frequency in run.chain_counts.most_common(10):
            lines.append(
                f"| `{chain}` | {frequency} | {fmt_percent(frequency, run.n_with_steps)} |"
            )

        if run.unit_limit is not None:
            lines.extend([
                "",
                "### 完整 raw 轨迹审计",
                "",
                (
                    f"原始文件共有 {run.raw_total_turns} 个 turn；按全部 {run.n_records} 题平均为 "
                    f"{run.raw_average_turns:.3f}，按 {run.raw_n_with_steps} 条非空链路平均为 "
                    f"{run.raw_average_turns_with_steps:.3f}。这些预算后 turn 不进入论文主指标。"
                ),
                "",
                "| Raw turn 数 | 记录数 | 占 raw 非空链路比例 |",
                "|---:|---:|---:|",
            ])
            for turn_count, frequency in sorted(run.raw_turn_counts.items()):
                lines.append(
                    f"| {turn_count} | {frequency} | {fmt_percent(frequency, run.raw_n_with_steps)} |"
                )

    lines.extend([
        "",
        "## 尚未可用的正式运行",
        "",
        # 2026-09-12 前此处还有一行占位，原文逐字为：
        #   "- MultiPL-E-8Lang 最终 JCA 训练产物尚未完成。",
        # mpe0912 job 120 / r08 已登记为正式 eval，该行随之删除。
        "- MATH self-RL 尚无登记为同一 no-thinking shard 口径的可比正式结果；MATH JCA 已使用最新正式 eval。",
        "- Conifer self-RL（sas14b_rl）27 轮分数俱全，但没有任何一轮的 `trajectories.jsonl` 留在盘上，无法统计 turn；Conifer zero-shot 的 5 个输出目录全部为空。两者都不以别的 round 代替。",
        "",
        "以上运行不读取、不估算，也不以旧 SFT 或 smoke 结果代替。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/turn_count_analysis.py",
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
    parser.add_argument("--multipl-e-zero-shot-path", type=Path, default=DEFAULT_MULTIPL_E_ZERO_SHOT_PATH)
    parser.add_argument("--multipl-e-jca-path", type=Path, default=DEFAULT_MULTIPL_E_JCA_PATH)
    parser.add_argument("--multipl-e-mad-path", type=Path, default=DEFAULT_MULTIPL_E_MAD_PATH)
    parser.add_argument("--multipl-e-agentverse-path", type=Path, default=DEFAULT_MULTIPL_E_AGENTVERSE_PATH)
    parser.add_argument("--multipl-e-gptswarm-path", type=Path, default=DEFAULT_MULTIPL_E_GPTSWARM_PATH)
    parser.add_argument("--multipl-e-aflow-path", type=Path, default=DEFAULT_MULTIPL_E_AFLOW_PATH)
    parser.add_argument("--multipl-e-self-rl-path", type=Path, default=DEFAULT_MULTIPL_E_SELF_RL_PATH)
    parser.add_argument("--math-mad-path", type=Path, default=DEFAULT_MATH_MAD_PATH)
    parser.add_argument("--math-agentverse-path", type=Path, default=DEFAULT_MATH_AGENTVERSE_PATH)
    parser.add_argument("--math-gptswarm-path", type=Path, default=DEFAULT_MATH_GPTSWARM_PATH)
    parser.add_argument("--math-aflow-path", type=Path, default=DEFAULT_MATH_AFLOW_PATH)
    parser.add_argument("--math-jca-path", type=Path, default=DEFAULT_MATH_JCA_PATH)
    parser.add_argument("--math-zero-shot-path", type=Path, default=DEFAULT_MATH_ZERO_SHOT_PATH)
    parser.add_argument(
        "--math-jca-max-turns",
        type=int,
        default=MATH_JCA_TURN_BUDGET,
    )
    parser.add_argument("--conifer-jca-path", type=Path, default=CONIFER_JCA)
    parser.add_argument("--conifer-mad-path", type=Path, default=CONIFER_MAD)
    parser.add_argument("--conifer-agentverse-path", type=Path, default=CONIFER_AGENTVERSE)
    parser.add_argument("--conifer-gptswarm-path", type=Path, default=CONIFER_GPTSWARM)
    parser.add_argument("--conifer-aflow-path", type=Path, default=CONIFER_AFLOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    if args.math_jca_max_turns < 0:
        parser.error("--math-jca-max-turns must be non-negative")

    runs = [
        analyze("MuSiQue", "JCA（正式运行）", args.musique_path, "JCA"),
        analyze("MuSiQue", "训练前基线（1.7B+4B+8B，无 LoRA）", args.musique_zeroshot_path, "JCA"),
        analyze("GSM-Hard", "JCA（正式运行）", args.gsm_path, "JCA"),
        analyze("GSM-Hard", "训练前基线（1.7B+4B+8B，无 LoRA）", args.gsm_zeroshot_path, "JCA"),
        analyze("MultiPL-E-8Lang", "异构 zero-shot MAS", args.multipl_e_zero_shot_path, "JCA"),
        analyze("MultiPL-E-8Lang", "最终 JCA 训练产物（mpe0912 job 120 / r08）", args.multipl_e_jca_path, "JCA"),
        analyze("MultiPL-E-8Lang", "MAD training-free", args.multipl_e_mad_path, "MAD"),
        analyze("MultiPL-E-8Lang", "AgentVerse training-free", args.multipl_e_agentverse_path, "AgentVerse"),
        analyze("MultiPL-E-8Lang", "GPTSwarm training-free", args.multipl_e_gptswarm_path, "GPTSwarm"),
        analyze("MultiPL-E-8Lang", "AFlow training-free", args.multipl_e_aflow_path, "AFlow"),
        analyze("MultiPL-E-8Lang", "SAS self-RL normalized v2", args.multipl_e_self_rl_path, "self-RL"),
        analyze("MATH", "训练前异构 zero-shot（1.7B+4B+8B；零初始化 adapter；no-thinking）", args.math_zero_shot_path, "JCA"),
        analyze(
            "MATH",
            "JCA 正式 eval（三-turn 有效执行视图）",
            args.math_jca_path,
            "JCA",
            max_units=args.math_jca_max_turns,
        ),
        analyze("MATH", "MAD training-free（no-thinking shard_04）", args.math_mad_path, "MAD"),
        analyze("MATH", "AgentVerse training-free（no-thinking shard_04）", args.math_agentverse_path, "AgentVerse"),
        analyze("MATH", "GPTSwarm training-free（no-thinking shard_04）", args.math_gptswarm_path, "GPTSwarm"),
        analyze("MATH", "AFlow training-free（no-thinking shard_04）", args.math_aflow_path, "AFlow"),
        # Conifer（2026-09-14 加入）。五个 arm 都是 run_conifer_mas.py 的统一 MAS
        # trajectory schema，所以 method 一律传 "JCA"；方法之间的差异体现在 step 序列
        # 本身，而不是落盘格式。MAD/AgentVerse 末尾的 harness confirm_stop step 由
        # require_generation 剔除。
        analyze("Conifer", "JCA 正式 eval（round_4）", args.conifer_jca_path, "JCA", require_generation=True),
        analyze("Conifer", "MAD training-free（round_27）", args.conifer_mad_path, "JCA", require_generation=True),
        analyze("Conifer", "AgentVerse training-free（round_27）", args.conifer_agentverse_path, "JCA", require_generation=True),
        analyze("Conifer", "GPTSwarm training-free（round_27）", args.conifer_gptswarm_path, "JCA", require_generation=True),
        analyze("Conifer", "AFlow training-free（130 批次 r04）", args.conifer_aflow_path, "JCA", require_generation=True),
    ]
    report = render_report(runs, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] {args.output}")


if __name__ == "__main__":
    main()
