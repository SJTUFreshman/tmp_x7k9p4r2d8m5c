"""Build reproducible four-bucket turn tables for JCA and zero-shot MAS.

The four mutually exclusive buckets are:

1. trajectories with at most two turns, including empty trajectories;
2. linear trajectories with at least three turns;
3. cyclic trajectories with at most four turns;
4. cyclic trajectories with at least five turns.

MATH's evaluator-inserted bootstrap-copy step is excluded from both views.
Conifer zero-shot uses one fixed rollout from its complete four-rollout pool so
that every benchmark contributes exactly one trajectory per problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.math.common import is_math_jca_bootstrap_copy_step

try:
    from .benchmark_paths import (
        CONIFER_JCA,
        CONIFER_N_PROBLEMS,
        CONIFER_ZERO_SHOT_POOL,
        CONIFER_ZERO_SHOT_POOL_ROLLOUT_INDEX,
        GSM_JCA,
        GSM_ZEROSHOT,
        MATH_JCA,
        MATH_ZERO_SHOT,
        MULTIPL_E_JCA,
        MULTIPL_E_ZERO_SHOT,
        MUSIQUE_JCA,
        MUSIQUE_ZEROSHOT,
    )
    from .trajectory_adapters import raw_attempts
except ImportError:
    from benchmark_paths import (
        CONIFER_JCA,
        CONIFER_N_PROBLEMS,
        CONIFER_ZERO_SHOT_POOL,
        CONIFER_ZERO_SHOT_POOL_ROLLOUT_INDEX,
        GSM_JCA,
        GSM_ZEROSHOT,
        MATH_JCA,
        MATH_ZERO_SHOT,
        MULTIPL_E_JCA,
        MULTIPL_E_ZERO_SHOT,
        MUSIQUE_JCA,
        MUSIQUE_ZEROSHOT,
    )
    from trajectory_adapters import raw_attempts


BUCKET_ORDER = ("short", "linear", "short_cycle", "long_cycle")
BUCKET_LABELS = {
    "short": "第一种（≤2 turn）",
    "linear": "第二种（linear ≥3 turn）",
    "short_cycle": "第三种（cyclic ≤4 turn）",
    "long_cycle": "第四种（cyclic ≥5 turn）",
}
AGENTS = frozenset({"A1", "A2", "A3"})
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "linear_cyclic_turn_bucket_report.md"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    phase: str
    path: Path
    expected_records: int
    remove_math_bootstrap: bool = False
    require_generation: bool = False
    rollout_index: int | None = None
    require_unique_problem_ids: bool = False


@dataclass
class BucketStats:
    count: int = 0
    total_turns: int = 0
    lengths: Counter[int] = field(default_factory=Counter)

    @property
    def mean_turns(self) -> float | None:
        return self.total_turns / self.count if self.count else None

    def add(self, turns: int) -> None:
        self.count += 1
        self.total_turns += turns
        self.lengths[turns] += 1


@dataclass
class RunStats:
    spec: RunSpec
    buckets: dict[str, BucketStats] = field(
        default_factory=lambda: {name: BucketStats() for name in BUCKET_ORDER}
    )
    records_seen: int = 0
    records_selected: int = 0
    removed_bootstrap_steps: int = 0
    problem_ids: set[str] = field(default_factory=set)

    @property
    def total_records(self) -> int:
        return sum(bucket.count for bucket in self.buckets.values())


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


def classify_chain(chain: tuple[str, ...]) -> str:
    turns = len(chain)
    if turns <= 2:
        return "short"
    if turns == len(set(chain)):
        return "linear"
    if turns <= 4:
        return "short_cycle"
    return "long_cycle"


def extract_chain(record: dict[str, Any], spec: RunSpec) -> tuple[tuple[str, ...], int]:
    trajectory = record.get("trajectory") or {}
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError(f"{spec.dataset}/{spec.phase}: trajectory.steps must be a list")

    chain: list[str] = []
    removed_bootstrap_steps = 0
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if spec.require_generation and raw_attempts(step) is None:
            continue
        if spec.remove_math_bootstrap and is_math_jca_bootstrap_copy_step(
            record, index, step
        ):
            removed_bootstrap_steps += 1
            continue
        agent = str(step.get("active_agent") or "").strip()
        if agent not in AGENTS:
            raise ValueError(
                f"{spec.dataset}/{spec.phase}: unsupported active_agent={agent!r}"
            )
        chain.append(agent)
    return tuple(chain), removed_bootstrap_steps


def analyze_run(spec: RunSpec) -> RunStats:
    if not spec.path.is_file():
        raise FileNotFoundError(f"{spec.dataset}/{spec.phase} input missing: {spec.path}")

    stats = RunStats(spec=spec)
    for record in iter_jsonl(spec.path):
        stats.records_seen += 1
        if spec.rollout_index is not None and int(record.get("rollout_idx", 0)) != spec.rollout_index:
            continue
        stats.records_selected += 1
        if spec.require_unique_problem_ids:
            problem_id = str(record.get("problem_id") or "").strip()
            if not problem_id:
                raise ValueError(f"{spec.dataset}/{spec.phase}: missing problem_id")
            if problem_id in stats.problem_ids:
                raise ValueError(
                    f"{spec.dataset}/{spec.phase}: duplicate problem_id={problem_id}"
                )
            stats.problem_ids.add(problem_id)
        chain, removed = extract_chain(record, spec)
        stats.removed_bootstrap_steps += removed
        stats.buckets[classify_chain(chain)].add(len(chain))

    if stats.records_selected != spec.expected_records:
        raise ValueError(
            f"{spec.dataset}/{spec.phase}: expected {spec.expected_records} selected "
            f"records, found {stats.records_selected}"
        )
    if stats.total_records != stats.records_selected:
        raise AssertionError(f"{spec.dataset}/{spec.phase}: bucket partition is incomplete")
    return stats


def validate_conifer_alignment(
    trained: list[RunStats], zero_shot: list[RunStats]
) -> None:
    trained_ids = next(run.problem_ids for run in trained if run.spec.dataset == "Conifer")
    zero_shot_ids = next(
        run.problem_ids for run in zero_shot if run.spec.dataset == "Conifer"
    )
    if trained_ids != zero_shot_ids:
        raise ValueError(
            "Conifer trained and zero-shot problem IDs differ: "
            f"trained_only={len(trained_ids - zero_shot_ids)}, "
            f"zero_shot_only={len(zero_shot_ids - trained_ids)}"
        )


def aggregate(runs: Iterable[RunStats]) -> dict[str, BucketStats]:
    totals = {name: BucketStats() for name in BUCKET_ORDER}
    for run in runs:
        for name in BUCKET_ORDER:
            source = run.buckets[name]
            target = totals[name]
            target.count += source.count
            target.total_turns += source.total_turns
            target.lengths.update(source.lengths)
    return totals


def fmt_count(bucket: BucketStats, denominator: int) -> str:
    return f"{bucket.count} ({bucket.count / denominator:.2%})"


def fmt_mean(bucket: BucketStats) -> str:
    return "—" if bucket.mean_turns is None else f"{bucket.mean_turns:.2f}"


def render_table(lines: list[str], runs: list[RunStats]) -> None:
    headers = ["数据集"]
    alignments = ["---"]
    for name in BUCKET_ORDER:
        headers.extend([BUCKET_LABELS[name], f"{BUCKET_LABELS[name]}平均 turn"])
        alignments.extend(["---:", "---:"])
    lines.extend([f"| {' | '.join(headers)} |", f"| {' | '.join(alignments)} |"])

    for run in runs:
        values = [run.spec.dataset]
        for name in BUCKET_ORDER:
            bucket = run.buckets[name]
            values.extend([fmt_count(bucket, run.total_records), fmt_mean(bucket)])
        lines.append(f"| {' | '.join(values)} |")

    totals = aggregate(runs)
    denominator = sum(bucket.count for bucket in totals.values())
    values = ["合计"]
    for name in BUCKET_ORDER:
        values.extend([fmt_count(totals[name], denominator), fmt_mean(totals[name])])
    lines.append(f"| {' | '.join(values)} |")


def render_audit_table(lines: list[str], phase: str, runs: list[RunStats]) -> None:
    lines.extend(
        [
            f"### {phase} 输入审计",
            "",
            "| 数据集 | 输入文件 | 文件记录数 | 入表记录数 | rollout 选择 | 剔除 bootstrap-copy step |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for run in runs:
        rollout = "全部" if run.spec.rollout_index is None else str(run.spec.rollout_index)
        lines.append(
            f"| {run.spec.dataset} | `{run.spec.path}` | {run.records_seen} "
            f"| {run.records_selected} | {rollout} | {run.removed_bootstrap_steps} |"
        )
    lines.append("")


def render_bucket_audit(lines: list[str], phases: list[tuple[str, list[RunStats]]]) -> None:
    lines.extend(
        [
            "## 合计审计",
            "",
            "平均 turn 直接由本表的整数 `总 turn / 轨迹数` 计算，不从已四舍五入的行均值反推。",
            "",
            "| 阶段 | 类别 | 轨迹数 | 总 turn | 平均 turn | turn 长度分布 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for phase, runs in phases:
        totals = aggregate(runs)
        for name in BUCKET_ORDER:
            bucket = totals[name]
            distribution = ", ".join(
                f"{turns}:{count}" for turns, count in sorted(bucket.lengths.items())
            )
            lines.append(
                f"| {phase} | {BUCKET_LABELS[name]} | {bucket.count} "
                f"| {bucket.total_turns} | {fmt_mean(bucket)} | `{distribution}` |"
            )
    lines.append("")


def build_specs(conifer_zero_shot_rollout_index: int) -> tuple[list[RunSpec], list[RunSpec]]:
    trained = [
        RunSpec("MuSiQue", "训练后", MUSIQUE_JCA, 2417),
        RunSpec("GSM-Hard", "训练后", GSM_JCA, 132),
        RunSpec("MATH", "训练后", MATH_JCA, 500, remove_math_bootstrap=True),
        RunSpec("MPL-E-8", "训练后", MULTIPL_E_JCA, 1352),
        RunSpec(
            "Conifer",
            "训练后",
            CONIFER_JCA,
            CONIFER_N_PROBLEMS,
            require_generation=True,
            require_unique_problem_ids=True,
        ),
    ]
    zero_shot = [
        RunSpec("MuSiQue", "zero-shot", MUSIQUE_ZEROSHOT, 2417),
        RunSpec("GSM-Hard", "zero-shot", GSM_ZEROSHOT, 132),
        RunSpec("MATH", "zero-shot", MATH_ZERO_SHOT, 500, remove_math_bootstrap=True),
        RunSpec("MPL-E-8", "zero-shot", MULTIPL_E_ZERO_SHOT, 1352),
        RunSpec(
            "Conifer",
            "zero-shot",
            CONIFER_ZERO_SHOT_POOL,
            CONIFER_N_PROBLEMS,
            require_generation=True,
            rollout_index=conifer_zero_shot_rollout_index,
            require_unique_problem_ids=True,
        ),
    ]
    return trained, zero_shot


def render_report(
    trained: list[RunStats],
    zero_shot: list[RunStats],
    output_path: Path,
) -> str:
    lines = [
        "# 五数据集 JCA 轨迹四分类：训练前 zero-shot 与训练后",
        "",
        "本报告比较五个数据集的训练前异构 MAS zero-shot 与训练后 JCA canonical run。每个数据集一题一条轨迹，两张表的总分母均为 5803。",
        "",
        "## 分类定义",
        "",
        "四类互斥且覆盖全部记录：",
        "",
        "1. **第一种**：turn 数不超过 2；零 turn 的空轨迹也归入该类。",
        "2. **第二种**：turn 数至少为 3，且 active-agent 链路中没有重复 agent。",
        "3. **第三种**：链路中存在重复 agent，且 turn 数不超过 4。",
        "4. **第四种**：链路中存在重复 agent，且 turn 数至少为 5。",
        "",
        "一个 `trajectory.steps` 中真正执行的协议模型单元计为一个 turn。MATH 两个阶段都剔除 evaluator 强制插入的 bootstrap-copy step；Conifer 剔除不对应模型生成的 harness step。",
        "",
        "## 训练前异构 MAS zero-shot",
        "",
    ]
    render_table(lines, zero_shot)
    lines.extend(
        [
            "",
            "Conifer 的完整训练前池包含 1402 题 × 4 rollout；主表固定取 `rollout_idx=0`，与官方导出器默认值一致。该切片的 1402 个 problem ID 与训练后 Conifer canonical run 完全同集。正式 eval `20260901_190940` 被 SIGTERM 截断在 1003/1402 题，因此不用于全量合计。",
            "",
            "GSM-Hard 的 canonical zero-shot 开启 thinking；MuSiQue、MATH、MPL-E-8 与 Conifer 关闭。MATH 加载 fresh zero-init adapter，等价于未改变裸模型；其余数据集不加载训练 adapter。",
            "",
            "## 训练后 JCA",
            "",
        ]
    )
    render_table(lines, trained)
    lines.extend(["", "## 输入与复现审计", ""])
    render_audit_table(lines, "zero-shot", zero_shot)
    render_audit_table(lines, "训练后", trained)
    render_bucket_audit(lines, [("zero-shot", zero_shot), ("训练后", trained)])
    lines.extend(
        [
            "## 复现命令",
            "",
            "```bash",
            "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/linear_cyclic_turn_bucket_analysis.py",
            "```",
            "",
            f"输出文件：`{output_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--conifer-zero-shot-rollout-index",
        type=int,
        default=CONIFER_ZERO_SHOT_POOL_ROLLOUT_INDEX,
    )
    args = parser.parse_args()
    if args.conifer_zero_shot_rollout_index < 0:
        parser.error("--conifer-zero-shot-rollout-index must be non-negative")

    trained_specs, zero_shot_specs = build_specs(args.conifer_zero_shot_rollout_index)
    trained = [analyze_run(spec) for spec in trained_specs]
    zero_shot = [analyze_run(spec) for spec in zero_shot_specs]
    validate_conifer_alignment(trained, zero_shot)
    report = render_report(trained, zero_shot, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] {args.output}")


if __name__ == "__main__":
    main()
