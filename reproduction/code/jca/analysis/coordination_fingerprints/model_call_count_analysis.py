"""Count per-question model calls across all available benchmark runs.

The primary metric is the number of completed model generations, including
retries. Depending on the run, generations are recovered from either complete
``raw_outputs`` ledgers or the run's vLLM access logs. For AFlow, the report
adds the one-time workflow-search calls to the deployment calls. It also
includes protocol-level logical calls, which exclude retries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .benchmark_paths import (
        CONIFER_AFLOW,
        CONIFER_AFLOW_SEARCH_STATE,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_JCA,
        CONIFER_MAD,
        GSM_JCA,
        GSM_ZEROSHOT,
        MATH_AFLOW,
        MATH_AFLOW_SEARCH_STATE,
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
        MUSIQUE_AFLOW,
        MUSIQUE_AFLOW_SEARCH_STATE,
        MUSIQUE_AGENTVERSE,
        MUSIQUE_GPTSWARM,
        MUSIQUE_JCA,
        MUSIQUE_MAD,
        MUSIQUE_ZEROSHOT,
    )
    from .trajectory_adapters import protocol_units, raw_attempts
except ImportError:
    from benchmark_paths import (
        CONIFER_AFLOW,
        CONIFER_AFLOW_SEARCH_STATE,
        CONIFER_AGENTVERSE,
        CONIFER_GPTSWARM,
        CONIFER_JCA,
        CONIFER_MAD,
        GSM_JCA,
        GSM_ZEROSHOT,
        MATH_AFLOW,
        MATH_AFLOW_SEARCH_STATE,
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
        MUSIQUE_AFLOW,
        MUSIQUE_AFLOW_SEARCH_STATE,
        MUSIQUE_AGENTVERSE,
        MUSIQUE_GPTSWARM,
        MUSIQUE_JCA,
        MUSIQUE_MAD,
        MUSIQUE_ZEROSHOT,
    )
    from trajectory_adapters import protocol_units, raw_attempts

from analysis.math.common import (
    math_jca_answer_projection,
    math_jca_bootstrap_component,
)
try:
    from src.math_eval import compute_math_em
except ImportError:
    from jca.src.math_eval import compute_math_em


OUTPUT_DIR = Path(__file__).resolve().parent

# 2026-09-11 起这四条与其余数据集一样从 benchmark_paths 导入。此前它们在本文件里另抄
# 了一份字面路径，导致 benchmark_paths 换 run 时这里不会跟着走。
DEFAULT_MUSIQUE_JCA = MUSIQUE_JCA
DEFAULT_MUSIQUE_MAD = MUSIQUE_MAD
DEFAULT_MUSIQUE_AGENTVERSE = MUSIQUE_AGENTVERSE
DEFAULT_MUSIQUE_AFLOW = MUSIQUE_AFLOW
DEFAULT_MUSIQUE_GPTSWARM = MUSIQUE_GPTSWARM

DEFAULT_GSM_ROOT = ROOT / "logs/gsm_hard_baselines/gsmhard_latest_roles_raw8192_20260816"
DEFAULT_GSM_MAD = DEFAULT_GSM_ROOT / "01_mad/results.jsonl"
DEFAULT_GSM_AGENTVERSE = DEFAULT_GSM_ROOT / "02_agentverse/results.jsonl"
DEFAULT_GSM_AFLOW = DEFAULT_GSM_ROOT / "03_aflow/eval_results.jsonl"
DEFAULT_GSM_GPTSWARM = DEFAULT_GSM_ROOT / "04_gptswarm/results.jsonl"
DEFAULT_GSM_JCA = GSM_JCA
DEFAULT_GSM_ZEROSHOT = GSM_ZEROSHOT
DEFAULT_MULTIPL_E_ZERO_SHOT = MULTIPL_E_ZERO_SHOT
DEFAULT_MULTIPL_E_JCA = MULTIPL_E_JCA
DEFAULT_MULTIPL_E_MAD = MULTIPL_E_MAD
DEFAULT_MULTIPL_E_AGENTVERSE = MULTIPL_E_AGENTVERSE
DEFAULT_MULTIPL_E_GPTSWARM = MULTIPL_E_GPTSWARM
DEFAULT_MULTIPL_E_AFLOW = MULTIPL_E_AFLOW
DEFAULT_MULTIPL_E_SELF_RL = MULTIPL_E_SELF_RL
DEFAULT_MATH_MAD = MATH_MAD
DEFAULT_MATH_AGENTVERSE = MATH_AGENTVERSE
DEFAULT_MATH_GPTSWARM = MATH_GPTSWARM
DEFAULT_MATH_AFLOW = MATH_AFLOW
DEFAULT_MATH_ZERO_SHOT = MATH_ZERO_SHOT
DEFAULT_MATH_JCA = MATH_JCA

# 2026-09-14 新增 Conifer。五个 arm 的 trajectory schema 完全一致（都是
# run_conifer_mas.py 产出的 MAS trajectory），因此复用 `protocol_units("jca", ...)`。
# 只有走 baseline_queue 的 AFlow 一条带完整 raw_outputs ledger；另外四条是
# forever-loop 产物，每个 step 只落一个 raw_output、round 目录里也没有 vLLM access
# log，含重试的调用数不可恢复，故以 retry_ledger=False 登记，只报逻辑调用。
DEFAULT_CONIFER_JCA = CONIFER_JCA
DEFAULT_CONIFER_MAD = CONIFER_MAD
DEFAULT_CONIFER_AGENTVERSE = CONIFER_AGENTVERSE
DEFAULT_CONIFER_GPTSWARM = CONIFER_GPTSWARM
DEFAULT_CONIFER_AFLOW = CONIFER_AFLOW

# MuSiQue JCA 仍是唯一需要 access-log 回退的 deployment：它落盘时没有 raw_outputs。
# MuSiQue 的 MAD / AgentVerse / GPTSwarm 在 2026-09-11 换成带 ledger 的新 run 之后不再
# 需要 server log，对应的 *_LOG_DIR 常量已随之删除。
MUSIQUE_JCA_LOG_DIR = ROOT / "logs/mas_eval_concurrent/rl_sft_mas_0717_1130"
GSM_JCA_LOG_DIR = (
    ROOT
    / "logs/gsm_eval/role_batched/gsm_judge_rl_v13_seed_sweep_infinite_20260809/"
    "gsm_judge_rl_v13_seed_sweep_infinite_20260809_seed43_dev132_"
    "thinking_hidden_self_handoff_ctx40960"
)

# AFlow search provenance.  2026-09-14 起 MuSiQue 的 search state 也由
# benchmark_paths 提供：070_musique_aflow_repeat5 是 MODE=search + 10 次 eval，
# search 与 deployment 同源，state 里带 raw_outputs/attempts ledger，不再需要
# 「server 总数减 deployment ledger」那条回退路径，MUSIQUE_AFLOW_SEARCH_LOG_DIR
# 随之删除。config 在 search 子目录自带（MODE=search）。
# 2026-09-14 前此处为旧的 42.16% 单点所配的 search，原值逐字如下：
#   MUSIQUE_AFLOW_SEARCH_STATE = ROOT / (
#       "baseline/MuSiQue/AFlow/search_runs/20260812_185216_mcts_iters20_dev20/"
#       "state.json"
#   )
#   MUSIQUE_AFLOW_SEARCH_LOG_DIR = ROOT / (
#       "baseline/MuSiQue/AFlow/logs/aflow_hetero_search20_dev20_full2417"
#   )
#   MUSIQUE_AFLOW_SEARCH_CONFIG = MUSIQUE_AFLOW_SEARCH_LOG_DIR / "config.env"
MUSIQUE_AFLOW_SEARCH_CONFIG = MUSIQUE_AFLOW_SEARCH_STATE.parent / "config.env"
GSM_AFLOW_SEARCH_STATE = ROOT / (
    "logs/gsm_hard_baselines/"
    "gsmhard_aflow_hetero_search20_eval132_20260814_113713/search/state.json"
)
GSM_AFLOW_SEARCH_LOG_DIR = ROOT / (
    "logs/gsm_hard_baselines/"
    "gsmhard_aflow_hetero_search20_eval132_20260814_113713/logs/"
    "gsmhard_aflow_hetero_search20_eval132_20260814_113713_search"
)
GSM_AFLOW_SEARCH_CONFIG = GSM_AFLOW_SEARCH_LOG_DIR / "config.env"
MULTIPL_E_AFLOW_SEARCH_STATE = ROOT / (
    "logs/multipl_e_8lang_baselines/aflow/"
    "multipl_e_8lang_aflow_full_20x20_test30_20260820/search/state.json"
)
MULTIPL_E_AFLOW_SEARCH_CONFIG = ROOT / (
    "logs/multipl_e_8lang_baselines/aflow/"
    "multipl_e_8lang_aflow_full_20x20_test30_20260820/config.env"
)
# MATH 的 search state 由 benchmark_paths 提供（2026-09-11 起指向 020_math_aflow_both
# 自带的 search）。该 run 是 MODE=both，search 子目录没有独立 config.env，配置在 run 根。
MATH_AFLOW_SEARCH_CONFIG = (
    MATH_AFLOW_SEARCH_STATE.parent.parent / "config.env"
)
# Conifer 的 search 是独立批次 030_conifer_aflow_search（MODE=search，20×20，
# no-thinking）。130_conifer_aflow_eval_x5 的 summary.json 里
# `reused_search.selected_source` 指向该 search 的 round_01_proposed.py，
# 所以 deployment 与 search 同源，可按常规摊销。search 子目录自带 config.env。
CONIFER_AFLOW_SEARCH_CONFIG = CONIFER_AFLOW_SEARCH_STATE.parent / "config.env"

DEFAULT_OUTPUT = OUTPUT_DIR / "model_call_count_report.md"
MODEL_SIZES = ("1.7B", "4B", "8B", "14B", "unknown")
METHOD_ORDER = {
    "JCA": 0,
    "zero-shot": 1,
    "MAD": 2,
    "AgentVerse": 3,
    "AFlow": 4,
    "GPTSwarm": 5,
    "self-RL": 6,
}
GPTSWARM_TYPE_TO_SIZE = {
    "IO": "1.7B",
    "CoT": "4B",
    "Debate": "8B",
    "Aggregator": "8B",
}


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


def agent_size(value: Any, agent_size_map: dict[str, str] | None = None) -> str:
    text = str(value or "")
    # 2026-09-13：agent_size_map 优先。槽位名不再必然对应固定模型规模——reversed
    # capacity 的 run 里 A1 是 8B、A3 是 1.7B。查不到才退回旧的 A1=1.7B 约定。
    if agent_size_map:
        slot = text.split("_", 1)[0]
        if slot in agent_size_map:
            return agent_size_map[slot]
    if text == "A1" or text.startswith("A1_"):
        return "1.7B"
    if text == "A2" or text.startswith("A2_"):
        return "4B"
    if text == "A3" or text.startswith("A3_"):
        return "8B"
    lowered = text.lower()
    if "14b" in lowered or "sas" in lowered:
        return "14B"
    if "1.7b" in lowered:
        return "1.7B"
    if "8b" in lowered:
        return "8B"
    if "4b" in lowered:
        return "4B"
    return "unknown"


@dataclass(frozen=True)
class LogicalCall:
    model_size: str
    payload: dict[str, Any]

    @property
    def recorded_attempts(self) -> int | None:
        attempts = self.payload.get("attempts")
        if isinstance(attempts, list):
            return len(attempts)
        raw_outputs = raw_attempts(self.payload)
        return len(raw_outputs) if raw_outputs is not None else None


def standardized_recruiter(agentverse: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "raw_output": agentverse.get("recruit_raw_output"),
        "retried": agentverse.get("recruit_retried"),
    }
    if "recruit_raw_outputs" in agentverse:
        payload["raw_outputs"] = agentverse.get("recruit_raw_outputs")
    return payload


def logical_calls(
    method: str, record: dict[str, Any], agent_size_map: dict[str, str] | None = None
) -> list[LogicalCall]:
    calls: list[LogicalCall] = []
    for unit in protocol_units(method, record):
        payload = unit.payload
        size = agent_size(unit.agent, agent_size_map)
        if method == "AgentVerse" and unit.kind in {"recruit", "evaluation"} and size == "unknown":
            size = "8B"
        if method == "GPTSwarm" and size == "unknown":
            size = GPTSWARM_TYPE_TO_SIZE.get(str(payload.get("node_type") or ""), "unknown")
        calls.append(LogicalCall(size, payload))
    return calls


@dataclass(frozen=True)
class ServerLog:
    model_size: str
    path: Path


@dataclass(frozen=True)
class ServerRequestCounts:
    completed: Counter[str]
    rejected: Counter[str]


def count_server_requests(logs: Sequence[ServerLog]) -> ServerRequestCounts:
    completed: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for server_log in logs:
        if not server_log.path.is_file():
            raise FileNotFoundError(f"server log does not exist: {server_log.path}")
        with server_log.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "POST /v1/chat/completions" not in line:
                    continue
                if 'HTTP/1.1" 200' in line:
                    completed[server_log.model_size] += 1
                else:
                    rejected[server_log.model_size] += 1
    return ServerRequestCounts(completed=completed, rejected=rejected)


def fixed_server_logs(directory: Path, include_meta: bool = False) -> list[ServerLog]:
    logs = [
        ServerLog("1.7B", directory / "A1_server.log"),
        ServerLog("4B", directory / "A2_server.log"),
        ServerLog("8B", directory / "A3_server.log"),
    ]
    if include_meta:
        logs.append(ServerLog("8B", directory / "META_server.log"))
    return logs


def aflow_search_server_logs(directory: Path) -> tuple[ServerLog, ...]:
    return tuple(
        [
            *fixed_server_logs(directory),
            ServerLog("14B", directory / "OPT_server.log"),
        ]
    )


def gsm_jca_server_logs(directory: Path) -> list[ServerLog]:
    logs: list[ServerLog] = []
    for path in sorted(directory.glob("stage_*_server.log")):
        size = "unknown"
        for agent in ("A1", "A2", "A3"):
            if f"_{agent}_server.log" in path.name:
                size = agent_size(agent)
                break
        logs.append(ServerLog(size, path))
    if not logs:
        raise FileNotFoundError(f"no GSM JCA stage server logs found under {directory}")
    return logs


def logs_for_default_path(
    actual_path: Path,
    default_path: Path,
    logs: Sequence[ServerLog],
) -> Sequence[ServerLog] | None:
    return logs if actual_path.resolve() == default_path.resolve() else None


def _correct_for_record(dataset: str, method: str, record: dict[str, Any]) -> bool:
    if dataset == "MATH" and method == "JCA":
        projection = math_jca_answer_projection(record)
        problem = record.get("problem") or {}
        gold = str(record.get("gold_answer") or problem.get("gold_answer") or "")
        return bool(compute_math_em(projection["projected_answer"], gold))
    if dataset == "Conifer":
        # Conifer 没有 EM。连续分是 hard_score（conifer_hard_score_v1），唯一的二值
        # 判据是 final_checks.all_explicit_passed（所有显式约束全部满足）。本报告的
        # 「EM」列填的是这个二值通过率，连续 hard_score 见 W4 token 报告与
        # benchmark_readiness_report.md 的备注列。
        checks = record.get("final_checks")
        if not isinstance(checks, dict) or "all_explicit_passed" not in checks:
            raise ValueError("Conifer record lacks final_checks.all_explicit_passed")
        return bool(checks["all_explicit_passed"])
    if "correct" in record:
        return bool(record.get("correct"))
    return float(record.get("em", 0.0) or 0.0) >= 1.0


@dataclass(frozen=True)
class AFlowSearchSpec:
    """Provenance and recovery settings for one AFlow workflow search."""

    dataset: str
    state_path: Path
    thinking_mode: str
    note: str
    config_path: Path | None = None
    server_logs: tuple[ServerLog, ...] = ()
    server_logs_include_deployment: bool = False


@dataclass
class AFlowSearchStats:
    """Logical calls and completed generations made while searching."""

    dataset: str
    path: Path
    n_nodes: int
    n_dev_evaluations: int
    executor_logical_total: int
    optimizer_logical_total: int
    logical_by_size: Counter[str]
    executor_request_total: int
    optimizer_request_total: int
    request_by_size: Counter[str]
    rejected_request_total: int
    request_source: str
    thinking_mode: str
    note: str
    config_path: Path | None = None

    @property
    def logical_total(self) -> int:
        return self.executor_logical_total + self.optimizer_logical_total

    @property
    def request_total(self) -> int:
        return self.executor_request_total + self.optimizer_request_total


def _empty_size_counter() -> Counter[str]:
    return Counter({size: 0 for size in MODEL_SIZES})


@dataclass
class RunStats:
    dataset: str
    method: str
    path: Path
    n_questions: int
    correct: int
    em: float
    logical_total: int
    logical_by_size: Counter[str]
    logical_per_question: list[int]
    request_total: int
    request_by_size: Counter[str]
    rejected_request_total: int | None
    request_source: str
    search: AFlowSearchStats | None = None
    unit_limit: int | None = None
    # 2026-09-14：False 表示这条 run 既没有 raw_outputs ledger 也没有 server log，
    # 含重试的完成生成数无法恢复。此时 request_* 字段被填成逻辑调用数（保证下游
    # 算术不炸），但报告里所有"完成生成/重生成"列一律打印 n/a，不得当作实测值。
    retry_recoverable: bool = True

    @property
    def mean_logical_calls(self) -> float:
        return self.logical_total / self.n_questions if self.n_questions else math.nan

    @property
    def mean_requests(self) -> float:
        return self.request_total / self.n_questions if self.n_questions else math.nan

    @property
    def search_logical_total(self) -> int:
        return self.search.logical_total if self.search is not None else 0

    @property
    def search_request_total(self) -> int:
        return self.search.request_total if self.search is not None else 0

    @property
    def pipeline_logical_total(self) -> int:
        return self.logical_total + self.search_logical_total

    @property
    def pipeline_request_total(self) -> int:
        return self.request_total + self.search_request_total

    @property
    def pipeline_retry_overhead(self) -> int:
        return self.pipeline_request_total - self.pipeline_logical_total

    @property
    def mean_search_logical_calls(self) -> float:
        return self.search_logical_total / self.n_questions if self.n_questions else math.nan

    @property
    def mean_search_requests(self) -> float:
        return self.search_request_total / self.n_questions if self.n_questions else math.nan

    @property
    def mean_pipeline_logical_calls(self) -> float:
        return self.pipeline_logical_total / self.n_questions if self.n_questions else math.nan

    @property
    def mean_pipeline_requests(self) -> float:
        return self.pipeline_request_total / self.n_questions if self.n_questions else math.nan

    @property
    def retry_overhead(self) -> int:
        return self.request_total - self.logical_total

    @property
    def mean_retry_overhead(self) -> float:
        return self.retry_overhead / self.n_questions if self.n_questions else math.nan

    def mean_requests_for_size(self, size: str) -> float:
        return self.request_by_size[size] / self.n_questions if self.n_questions else math.nan

    def search_requests_for_size(self, size: str) -> int:
        if self.search is None:
            return 0
        return self.search.request_by_size[size]

    def pipeline_requests_for_size(self, size: str) -> int:
        return self.request_by_size[size] + self.search_requests_for_size(size)

    def mean_pipeline_requests_for_size(self, size: str) -> float:
        return self.pipeline_requests_for_size(size) / self.n_questions if self.n_questions else math.nan

    # 2026-09-14 新增：逻辑调用的模型拆分。完成生成的拆分对没有重试账本的 run
    # （Conifer 的 JCA/MAD/AgentVerse/GPTSwarm）是 n/a，逻辑调用这一份则始终可用。
    def search_logical_for_size(self, size: str) -> int:
        if self.search is None:
            return 0
        return self.search.logical_by_size[size]

    def pipeline_logical_for_size(self, size: str) -> int:
        return self.logical_by_size[size] + self.search_logical_for_size(size)

    def mean_pipeline_logical_for_size(self, size: str) -> float:
        return self.pipeline_logical_for_size(size) / self.n_questions if self.n_questions else math.nan

    @property
    def median_logical_calls(self) -> float:
        return median(self.logical_per_question) if self.logical_per_question else math.nan

    @property
    def min_logical_calls(self) -> int:
        return min(self.logical_per_question) if self.logical_per_question else 0

    @property
    def max_logical_calls(self) -> int:
        return max(self.logical_per_question) if self.logical_per_question else 0


def _search_nodes(value: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        nodes = value
    elif isinstance(value, dict) and isinstance(value.get("nodes"), list):
        nodes = value["nodes"]
    else:
        raise ValueError(f"{path}: AFlow search state must contain a node list")
    if not all(isinstance(node, dict) for node in nodes):
        raise ValueError(f"{path}: AFlow search state contains a non-object node")
    return nodes


def _search_node_is_optimizer(node: dict[str, Any]) -> bool:
    if str(node.get("proposed_by") or "").lower() == "optimizer":
        return True
    try:
        return int(node.get("round_id", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _attempt_counts(container: dict[str, Any]) -> tuple[int, int, int]:
    """Return (completed responses, persisted attempts, rejected attempts)."""
    raw_outputs = container.get("raw_outputs")
    if isinstance(raw_outputs, list):
        # A raw_outputs entry is written only after the HTTP response arrived.
        return len(raw_outputs), len(raw_outputs), 0

    attempts = container.get("attempts")
    if isinstance(attempts, list):
        rejected = 0
        for attempt in attempts:
            if isinstance(attempt, dict) and (
                attempt.get("request_ok") is False
                or attempt.get("exception") not in (None, "")
            ):
                rejected += 1
        return len(attempts) - rejected, len(attempts), rejected
    return 0, 0, 0


def _optimizer_attempt_counts(node: dict[str, Any]) -> tuple[int, int, int] | None:
    if isinstance(node.get("optimizer_raw_outputs"), list):
        return _attempt_counts({"raw_outputs": node["optimizer_raw_outputs"]})
    if isinstance(node.get("optimizer_attempts"), list):
        return _attempt_counts({"attempts": node["optimizer_attempts"]})
    return None


def _search_op_size(operation: dict[str, Any], solve_index: int) -> str:
    size = agent_size(
        operation.get("caller_id")
        or operation.get("agent")
        or operation.get("api_model")
        or operation.get("model")
    )
    if size != "unknown":
        return size
    kind = str(operation.get("op") or "").lower()
    if kind == "solve":
        return MODEL_SIZES[solve_index % 3]
    if kind in {"ensemble", "answer_generate"}:
        return "8B"
    return "unknown"


def _inferred_search_sizes(kinds: Any) -> Counter[str]:
    sizes = _empty_size_counter()
    solve_index = 0
    if not isinstance(kinds, list):
        return sizes
    for kind in kinds:
        operation = {"op": kind} if not isinstance(kind, dict) else kind
        size = _search_op_size(operation, solve_index)
        sizes[size] += 1
        if str(operation.get("op") or "").lower() == "solve":
            solve_index += 1
    return sizes


def _subtract_size_counters(
    total: Counter[str], deployment: Counter[str], *, label: str
) -> Counter[str]:
    result = _empty_size_counter()
    for size in MODEL_SIZES:
        value = total[size] - deployment[size]
        if value < 0:
            raise ValueError(
                f"{label}: server count for {size} is smaller than deployment ledger "
                f"({total[size]} < {deployment[size]})"
            )
        result[size] = value
    return result


def analyze_aflow_search(
    spec: AFlowSearchSpec,
    deployment: RunStats,
) -> AFlowSearchStats:
    """Count AFlow's one-time workflow-search calls.

    Most search states persist every response attempt.  The oldest MuSiQue
    state persisted only ``n_ops``/``op_kinds``; for that run the accompanying
    server logs contain search and deployment traffic together, so completed
    search responses are recovered by subtracting the deployment ledger.
    """
    if not spec.state_path.is_file():
        raise FileNotFoundError(f"{spec.dataset} AFlow search state missing: {spec.state_path}")

    value = json.loads(spec.state_path.read_text(encoding="utf-8"))
    nodes = _search_nodes(value, spec.state_path)
    logical_by_size = _empty_size_counter()
    request_by_size = _empty_size_counter()
    executor_logical_total = 0
    optimizer_logical_total = 0
    executor_request_total = 0
    optimizer_request_total = 0
    rejected_request_total = 0
    ledger_complete = True
    n_dev_evaluations = 0

    for node in nodes:
        is_optimizer = _search_node_is_optimizer(node)
        if is_optimizer:
            optimizer_logical_total += 1
            optimizer_attempts = _optimizer_attempt_counts(node)
            if optimizer_attempts is None:
                ledger_complete = False
            else:
                completed, _persisted, rejected = optimizer_attempts
                optimizer_request_total += completed
                rejected_request_total += rejected
                request_by_size["14B"] += completed
                logical_by_size["14B"] += 1

        dev_records = node.get("dev_records") or []
        if not isinstance(dev_records, list):
            raise ValueError(f"{spec.state_path}: node dev_records is not a list")
        for evaluation in dev_records:
            if not isinstance(evaluation, dict):
                raise ValueError(f"{spec.state_path}: search evaluation is not an object")
            n_dev_evaluations += 1
            operations = evaluation.get("op_records")
            if not isinstance(operations, list):
                operations = evaluation.get("op_calls")
            if isinstance(operations, list) and operations:
                solve_index = 0
                for operation in operations:
                    if not isinstance(operation, dict):
                        raise ValueError(f"{spec.state_path}: search operation is not an object")
                    size = _search_op_size(operation, solve_index)
                    logical_by_size[size] += 1
                    executor_logical_total += 1
                    if str(operation.get("op") or "").lower() == "solve":
                        solve_index += 1
                    completed, _persisted, rejected = _attempt_counts(operation)
                    if "raw_outputs" not in operation and "attempts" not in operation:
                        ledger_complete = False
                    executor_request_total += completed
                    rejected_request_total += rejected
                    request_by_size[size] += completed
            else:
                declared_ops = evaluation.get("n_ops")
                if declared_ops is None:
                    declared_ops = len(evaluation.get("op_kinds") or [])
                try:
                    declared_ops = int(declared_ops or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{spec.state_path}: invalid n_ops={declared_ops!r}"
                    ) from exc
                if declared_ops > 0:
                    executor_logical_total += declared_ops
                    inferred_sizes = _inferred_search_sizes(evaluation.get("op_kinds"))
                    inferred_total = sum(inferred_sizes.values())
                    if inferred_total < declared_ops:
                        inferred_sizes["unknown"] += declared_ops - inferred_total
                    logical_by_size.update(inferred_sizes)
                    if not evaluation.get("op_records") and not evaluation.get("op_calls"):
                        ledger_complete = False

    if ledger_complete:
        request_source = "raw_outputs/attempts ledger"
    elif spec.server_logs:
        server_counts = count_server_requests(spec.server_logs)
        if spec.server_logs_include_deployment:
            request_by_size = _subtract_size_counters(
                server_counts.completed,
                deployment.request_by_size,
                label=f"{spec.dataset} AFlow search",
            )
            rejected_request_total = sum(server_counts.rejected.values())
        else:
            request_by_size = Counter(server_counts.completed)
            rejected_request_total = sum(server_counts.rejected.values())
        executor_request_total = sum(
            request_by_size[size] for size in MODEL_SIZES if size != "14B"
        )
        optimizer_request_total = request_by_size["14B"]
        request_source = "vLLM access log（减去 deployment ledger）" if spec.server_logs_include_deployment else "vLLM access log"
    else:
        raise ValueError(
            f"{spec.dataset} AFlow search has incomplete response ledger and no server logs: "
            f"{spec.state_path}"
        )

    return AFlowSearchStats(
        dataset=spec.dataset,
        path=spec.state_path,
        n_nodes=len(nodes),
        n_dev_evaluations=n_dev_evaluations,
        executor_logical_total=executor_logical_total,
        optimizer_logical_total=optimizer_logical_total,
        logical_by_size=logical_by_size,
        executor_request_total=executor_request_total,
        optimizer_request_total=optimizer_request_total,
        request_by_size=request_by_size,
        rejected_request_total=rejected_request_total,
        request_source=request_source,
        thinking_mode=spec.thinking_mode,
        note=spec.note,
        config_path=spec.config_path,
    )


def aflow_search_specs() -> dict[str, AFlowSearchSpec]:
    """Return the search run paired with each canonical AFlow deployment."""
    return {
        "MuSiQue": AFlowSearchSpec(
            dataset="MuSiQue",
            state_path=MUSIQUE_AFLOW_SEARCH_STATE,
            thinking_mode="disabled",
            config_path=MUSIQUE_AFLOW_SEARCH_CONFIG,
            note=(
                "MODE=search，本 run 自带的 20×20 no-thinking search；"
                "与同一 run 的 deployment 同源"
            ),
        ),
        "GSM-Hard": AFlowSearchSpec(
            dataset="GSM-Hard",
            state_path=GSM_AFLOW_SEARCH_STATE,
            thinking_mode="enabled",
            config_path=GSM_AFLOW_SEARCH_CONFIG,
            note="20 轮 search；state 保存 optimizer 与 executor 的响应重试",
        ),
        "MultiPL-E-8Lang": AFlowSearchSpec(
            dataset="MultiPL-E-8Lang",
            state_path=MULTIPL_E_AFLOW_SEARCH_STATE,
            thinking_mode="disabled",
            config_path=MULTIPL_E_AFLOW_SEARCH_CONFIG,
            note="20×20 workflow search；thinking 关闭",
        ),
        "MATH": AFlowSearchSpec(
            dataset="MATH",
            state_path=MATH_AFLOW_SEARCH_STATE,
            thinking_mode="disabled",
            config_path=MATH_AFLOW_SEARCH_CONFIG,
            note=(
                "MODE=both，本 run 自带的 20×20 no-thinking search；"
                "与同一 run 的 deployment 同源"
            ),
        ),
        "Conifer": AFlowSearchSpec(
            dataset="Conifer",
            state_path=CONIFER_AFLOW_SEARCH_STATE,
            thinking_mode="disabled",
            config_path=CONIFER_AFLOW_SEARCH_CONFIG,
            note=(
                "030_conifer_aflow_search 的 20×20 no-thinking search；"
                "130 批次的 deployment 复用其 round_01_proposed.py，两端同源"
            ),
        ),
    }


def analyze_run(
    dataset: str,
    method: str,
    path: Path,
    server_logs: Sequence[ServerLog] | None = None,
    max_units: int | None = None,
    # 2026-09-13：默认 None = 旧的 A1=1.7B/A2=4B/A3=8B 槽位约定，既有调用全部不变。
    # reversed capacity 的 run（config.env 里 MODEL_A1=Qwen3-8B）必须显式传入。
    agent_size_map: tuple[tuple[str, str], ...] | None = None,
    # 2026-09-14：显式声明这条 run 是否存在可用的重试账本。默认 True = 旧行为
    # （必须能走 ledger 或 server log，否则抛错）。传 False 表示已核实两者都没有：
    # 此时只统计逻辑调用，完成生成一律记为不可恢复，而不是拿 raw_output 的
    # "每个 step 恰好一次生成"冒充零重试。
    retry_ledger: bool = True,
    # 2026-09-14：轨迹 schema 与方法名脱钩。Conifer 的五个 arm 全部由
    # conifer_training_hub/03_rollout/run_conifer_mas.py 产出，落的是同一套 MAS
    # trajectory（trajectory.steps + active_agent），与 baseline/ 下各方法自己的
    # 格式无关，因此必须按 "jca" 解析，不能按 method 名去猜。默认 None = 用 method。
    protocol: str | None = None,
    # 2026-09-14：只把真正产生过模型生成的 protocol unit 计为逻辑调用。Conifer 的
    # MAD 与 AgentVerse 每题末尾各有一个 harness 自己写的 confirm_stop step
    # （MAD 是"controller selected the final draft by majority vote"，AgentVerse 是
    # "controller finalized the last recruited team answer"），raw_output 为空串、
    # 没有任何请求对应。不过滤的话 MAD 会记成 10.000 次/题而不是协议规定的 9.000。
    require_generation: bool = False,
) -> RunStats:
    if not path.is_file():
        raise FileNotFoundError(f"{dataset}/{method} input does not exist: {path}")
    if max_units is not None and max_units < 0:
        raise ValueError("max_units must be non-negative")

    rows = list(iter_jsonl(path))
    per_question: list[int] = []
    logical_by_size: Counter[str] = Counter()
    ledger_requests: Counter[str] = Counter()
    ledger_complete = True
    correct = 0
    em_values: list[float] = []
    has_persisted_accuracy = False

    for record in rows:
        if dataset == "MATH" and method == "JCA":
            projected = math_jca_answer_projection(record)
            problem = record.get("problem") or {}
            gold = str(record.get("gold_answer") or problem.get("gold_answer") or "")
            projected_em = float(compute_math_em(projected["projected_answer"], gold))
            em_values.append(projected_em)
            correct += int(projected_em >= 1.0)
            has_persisted_accuracy = True
        elif dataset == "Conifer":
            # Conifer 无 EM；「EM」列填二值口径 final_checks.all_explicit_passed。
            passed = _correct_for_record(dataset, method, record)
            em_values.append(float(passed))
            correct += int(passed)
            has_persisted_accuracy = True
        else:
            has_persisted_accuracy = has_persisted_accuracy or "em" in record or "correct" in record
            em = float(record.get("em", 0.0) or 0.0)
            em_values.append(em)
            correct += int(_correct_for_record(dataset, method, record))
        calls: list[LogicalCall] = []
        if dataset == "MATH" and method in {"JCA", "zero-shot"}:
            bootstrap = math_jca_bootstrap_component(record)
            if bootstrap is not None:
                bootstrap_model, bootstrap_attempts = bootstrap
                calls.append(
                    LogicalCall(
                        agent_size(bootstrap_model),
                        {"attempts": bootstrap_attempts},
                    )
                )
        protocol_calls = logical_calls(
            protocol or method, record, dict(agent_size_map) if agent_size_map else None
        )
        if require_generation:
            protocol_calls = [
                call for call in protocol_calls if call.recorded_attempts is not None
            ]
        if max_units is not None:
            protocol_calls = protocol_calls[:max_units]
        calls.extend(protocol_calls)
        per_question.append(len(calls))
        for call in calls:
            logical_by_size[call.model_size] += 1
            attempts = call.recorded_attempts
            if attempts is None:
                ledger_complete = False
            else:
                ledger_requests[call.model_size] += attempts

    retry_recoverable = True
    if not retry_ledger:
        if server_logs is not None:
            raise ValueError(
                f"{dataset}/{method}: retry_ledger=False conflicts with supplied server logs"
            )
        # 没有任何重试证据：把完成生成数填成逻辑调用数只是为了让下游算术有定义，
        # 报告会把这些列打成 n/a。
        request_by_size = Counter(logical_by_size)
        rejected_request_total: int | None = None
        request_source = "无 ledger / 无 server log（仅逻辑调用）"
        retry_recoverable = False
    elif server_logs is not None:
        server_counts = count_server_requests(server_logs)
        request_by_size = server_counts.completed
        rejected_request_total = sum(server_counts.rejected.values())
        request_source = "vLLM access log"
    elif ledger_complete:
        request_by_size = ledger_requests
        rejected_request_total = None
        request_source = "raw_outputs ledger"
    else:
        raise ValueError(
            f"{dataset}/{method} has no complete raw_outputs ledger and no server logs"
        )

    if not has_persisted_accuracy:
        summary_path = path.parent / "summary.txt"
        summary_text = summary_path.read_text(encoding="utf-8", errors="replace") if summary_path.is_file() else ""
        match = re.search(r"Overall correct:\s*(\d+)", summary_text)
        if match:
            correct = int(match.group(1))
            em = correct / len(rows) if rows else math.nan
        else:
            em = math.nan
    else:
        em = fmean(em_values) if em_values else math.nan

    logical_total = sum(per_question)
    request_total = sum(request_by_size.values())
    if request_total < logical_total:
        raise ValueError(
            f"{dataset}/{method}: recovered requests {request_total} < logical calls {logical_total}"
        )
    return RunStats(
        dataset=dataset,
        method=method,
        path=path,
        n_questions=len(rows),
        correct=correct,
        em=em,
        logical_total=logical_total,
        logical_by_size=logical_by_size,
        logical_per_question=per_question,
        request_total=request_total,
        request_by_size=request_by_size,
        rejected_request_total=rejected_request_total,
        request_source=request_source,
        unit_limit=max_units,
        retry_recoverable=retry_recoverable,
    )


def _req_cell(run: RunStats, value: float | int, spec: str = ".3f") -> str:
    """Format a retry-inclusive cell, or `n/a` when the run has no retry ledger."""
    if not run.retry_recoverable:
        return "n/a"
    return format(value, spec)


def render_report(runs: list[RunStats], output_path: Path) -> str:
    ordered = sorted(runs, key=lambda run: (run.dataset, METHOD_ORDER[run.method]))
    lines = [
        "# 各 Benchmark 平均模型调用次数",
        "",
        "本报告统计已完成正式 eval 的 JCA、异构 zero-shot、四个 training-free baseline 与 MultiPL-E SAS self-RL 的模型调用次数。AFlow 的端到端总量明确包含一次性 workflow search；同时保留 deployment 与 search 拆分，避免把两种成本混淆。MATH zero-shot 与 JCA 的部署调用都包含独立 8B bootstrap solver；zero-shot 统计完整 trajectory，JCA 只取后续前 3 个 trajectory unit。MATH 14B self-RL 仍不纳入。2026-09-14 加入 Conifer（5 行）；它的口径与另外四个数据集有三处系统性差异，见下节最后一条。",
        "",
        "## 口径",
        "",
        "- **完成模型生成（主指标）**：得到模型响应的请求次数，包含 parser/transport 重试；优先读取完整 `raw_outputs`/`attempts` ledger，旧格式运行使用 vLLM access log 中 HTTP 200 的 chat-completion 请求。失败 HTTP 请求不混入该主指标，search 表中单列失败请求。2026-09-11 起，deployment 里只剩 MuSiQue JCA 与 GSM-Hard JCA 两条走 access log，其余全部走 ledger（逐条来源见文末表）。",
        "- **逻辑调用**：协议中一个 agent turn、MAD turn、AgentVerse recruiter/solver/evaluator、AFlow op 或 GPTSwarm node 各记一次，不包含重试。",
        "- **AFlow search**：每个非初始 workflow node 记一个 14B optimizer 逻辑调用，每个 search `op_record` 记一个 executor 逻辑调用；所有 search 响应（含 retry）加入端到端总数。search 是一次性成本，按对应 deployment eval 的题目数摊销。",
        "- **Conifer 的三处口径差异（必须随数字一起引用）**：（1）它没有 EM，「EM」列填的是二值口径 `final_checks.all_explicit_passed`（显式约束全部满足），连续的 `hard_score` 不在本报告，见 W4 token 报告与 `benchmark_readiness_report.md`；（2）JCA/MAD/AgentVerse/GPTSwarm 来自 forever-loop 落盘，每个 step 只有单个 `raw_output`、没有 `raw_outputs` ledger，round 目录里也没有 vLLM access log，**含重试的完成生成数不可恢复**，这四行所有「完成生成 / 重生成」列一律为 `n/a`，只有逻辑调用可用；不要把 step 数当成零重试的完成生成数。只有走 baseline_queue 的 AFlow 一条 ledger 完整（8,412/8,412 个 step）；（3）Conifer 的 JSON 走 vLLM guided decoding（`--json-transport json_schema` + `response_format`），不是另外四个数据集的 prompt-only，解析失败率接近 0，因此即使补齐 ledger，它的重试率也**不能**与另外四个数据集横向比较。",
        "- 所有平均值分母均为该正式结果文件的题目数；运行失败的题目也计入分母。AgentVerse recruiter/evaluator 按记录中的 `meta_agent` 归类；固定 Meta-8B 旧格式回退为 8B。MATH zero-shot 与 JCA 均先计每题一次独立 8B bootstrap solver（包括其 retry）；zero-shot 再计完整 trajectory，JCA 再计前 3 个 trajectory unit。bootstrap 不属于协作 turn；完整协议轨迹长度另见 Turn 报告。",
        "",
        "## 汇总",
        "",
        "| 数据集 | 方法 | 题目数 | EM | 部署完成生成/题 | search 完成生成/题 | **总完成生成/题** | 部署逻辑调用/题 | search 逻辑调用/题 | **总逻辑调用/题** | 总重生成/题 | 总完成生成数 | 成本窗口 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in ordered:
        lines.append(
            f"| {run.dataset} | {run.method} | {run.n_questions} | {run.em:.2%} "
            f"| {_req_cell(run, run.mean_requests)} | {_req_cell(run, run.mean_search_requests)} "
            f"| **{_req_cell(run, run.mean_pipeline_requests)}** | {run.mean_logical_calls:.3f} "
            f"| {run.mean_search_logical_calls:.3f} | **{run.mean_pipeline_logical_calls:.3f}** "
            f"| {_req_cell(run, run.pipeline_retry_overhead / run.n_questions)} "
            f"| {_req_cell(run, run.pipeline_request_total, ',')} | "
            f"{'8B bootstrap + 前 3 个 unit' if run.unit_limit is not None else ('8B bootstrap + 完整 trajectory' if run.dataset == 'MATH' and run.method == 'zero-shot' else '完整 deployment')} |"
        )

    lines.extend([
        "",
        "其中“总重生成开销”是 `部署+search 完成模型生成数 − 部署+search 逻辑调用数`，包括 parser 未接受后触发的重生成。非 AFlow 方法没有 search，因此 search 列为 0。Conifer 的 JCA/MAD/AgentVerse/GPTSwarm 四行该列为 `n/a`——不是 0，而是无从恢复。",
        "",
        "MuSiQue JCA 的 server log 还记录了 1,284 次 HTTP 400（0.531 次/题）。这些请求没有形成模型响应或输出 token，因此不计入主表的完成模型生成；若论文希望统计所有 API POST 尝试，应将其单列。SAS self-RL 是 one-shot 生成，每题按一条逻辑调用统计；它没有 MAS 的 turn/handoff 结构。",
        "",
        "## 主要观察",
        "",
        "- JCA 与 baseline 的部署比较可看部署列；若比较获得 AFlow workflow 的完整端到端代价，应看包含 search 的总列。",
        "- AFlow search 的调用量并不随 deployment 题数线性重复；这里将一次 search 成本摊入当前正式 eval，扩展到更多题目时平均 search 成本会下降。",
        "- GSM-Hard 的 MAD、AgentVerse 和 AFlow 因长 thinking 输出或解析重试，完成生成数明显高于逻辑调用数。",
        "",
        "## 按模型大小拆分（端到端）",
        "",
        "以下为 deployment 与 AFlow search 合并后的完成生成；AFlow 的 14B 列来自 optimizer search。",
        "",
        "| 数据集 | 方法 | 1.7B calls/题 | 4B calls/题 | 8B calls/题 | 14B calls/题 | 未识别 calls/题 | 总 calls/题 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in ordered:
        lines.append(
            f"| {run.dataset} | {run.method} "
            f"| {_req_cell(run, run.mean_pipeline_requests_for_size('1.7B'))} "
            f"| {_req_cell(run, run.mean_pipeline_requests_for_size('4B'))} "
            f"| {_req_cell(run, run.mean_pipeline_requests_for_size('8B'))} "
            f"| {_req_cell(run, run.mean_pipeline_requests_for_size('14B'))} "
            f"| {_req_cell(run, run.mean_pipeline_requests_for_size('unknown'))} "
            f"| {_req_cell(run, run.mean_pipeline_requests)} |"
        )

    lines.extend([
        "",
        "## 按模型大小拆分（端到端逻辑调用）",
        "",
        "与上表同口径但不含重试。上表对没有重试账本的运行（Conifer 的 JCA/MAD/AgentVerse/GPTSwarm）整行为 n/a，本表则对所有运行都可用，因此模型配比只能从这里读。",
        "",
        "| 数据集 | 方法 | 1.7B 调用/题 | 4B 调用/题 | 8B 调用/题 | 14B 调用/题 | 未识别 调用/题 | 总调用/题 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in ordered:
        lines.append(
            f"| {run.dataset} | {run.method} "
            f"| {run.mean_pipeline_logical_for_size('1.7B'):.3f} "
            f"| {run.mean_pipeline_logical_for_size('4B'):.3f} "
            f"| {run.mean_pipeline_logical_for_size('8B'):.3f} "
            f"| {run.mean_pipeline_logical_for_size('14B'):.3f} "
            f"| {run.mean_pipeline_logical_for_size('unknown'):.3f} "
            f"| {run.mean_pipeline_logical_calls:.3f} |"
        )

    lines.extend([
        "",
        "## AFlow search 拆分",
        "",
        "search 的响应数全部来自 state 中持久化的 raw output/attempt ledger。2026-09-14 MuSiQue AFlow 换成自带 search 的重跑之后，四个数据集的 search 都不再需要「server 总请求数减 deployment ledger」这条恢复路径。",
        "",
        "| 数据集 | search thinking | nodes | dev eval 数 | executor 逻辑调用 | optimizer 逻辑调用 | search 逻辑总数 | 1.7B 完成生成 | 4B 完成生成 | 8B 完成生成 | 14B 完成生成 | search 完成总数 | search 失败请求 | 计数来源 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for run in ordered:
        if run.search is None:
            continue
        search = run.search
        lines.append(
            f"| {search.dataset} | {search.thinking_mode} | {search.n_nodes} "
            f"| {search.n_dev_evaluations:,} | {search.executor_logical_total:,} "
            f"| {search.optimizer_logical_total:,} | {search.logical_total:,} "
            f"| {search.request_by_size['1.7B']:,} | {search.request_by_size['4B']:,} "
            f"| {search.request_by_size['8B']:,} | {search.request_by_size['14B']:,} "
            f"| {search.request_total:,} | {search.rejected_request_total:,} "
            f"| {search.request_source} |"
        )

    lines.extend([
        "",
        "## 逻辑调用长度分布",
        "",
        "search 是一次性 workflow 构建过程，无法从 search state 还原到 deployment 每道题的调用链，因此中位数/最小/最大仍针对 deployment 逐题逻辑调用；总平均同时列出 search 摊销。",
        "",
        "| 数据集 | 方法 | 部署平均 | search 平均摊销 | 端到端平均 | 部署中位数 | 部署最少 | 部署最多 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in ordered:
        lines.append(
            f"| {run.dataset} | {run.method} | {run.mean_logical_calls:.3f} "
            f"| {run.mean_search_logical_calls:.3f} | {run.mean_pipeline_logical_calls:.3f} "
            f"| {run.median_logical_calls:.1f} | {run.min_logical_calls} "
            f"| {run.max_logical_calls} |"
        )

    lines.extend([
        "",
        "## 输入、search 与计数来源",
        "",
        "| 数据集 | 方法 | deployment 结果文件 | deployment 生成计数来源 | search state | search 配置 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ])
    for run in ordered:
        search_path = f"`{run.search.path}`" if run.search is not None else "—"
        search_config = (
            f"`{run.search.config_path}`" if run.search is not None and run.search.config_path else "—"
        )
        note = run.search.note if run.search is not None else "—"
        lines.append(
            f"| {run.dataset} | {run.method} | `{run.path}` | {run.request_source} "
            f"| {search_path} | {search_config} | {note} |"
        )

    lines.extend([
        "",
        "MuSiQue 四个 baseline 已于 2026-09-11 全部重跑，deployment 与 search 的计数都走完整 ledger，不再依赖 access log 反推。AFlow 取 `070_musique_aflow_repeat5` 官方 5-roll 中 EM 最高的 `run_01`（`1011/2417 = 41.83%`）：该 run 先 `MODE=search` 搜出 `round_07_proposed.py` 再用它部署，search 与 deployment 同源。它替换掉的是 42.16% 的旧单点（search 响应数只能由 server 总数减 deployment 恢复），而旧汇总报告里的 44.19% 重搜版本一直未采用。新 workflow 每题 18.911 个 op、旧的为 7.000，因此 AFlow 的调用数与 token 明显高于旧值。GSM-Hard JCA 使用正式 seed=43、`95/132 = 71.97%` 运行。MultiPL-E AgentVerse 沿用 formal aggregate 选定的 repeat-5 `run_03`（801/1352）；不使用 full_test30 的其他运行。MATH deployment 使用共同 `shard_04`（500 题，thinking 关闭）；zero-shot 与 JCA 的调用均包含独立 8B bootstrap solver。MATH GPTSwarm 与 AFlow 也在 2026-09-11 重跑：GPTSwarm 的 `TEMPERATURE` 由 0.0 改为 0.7、与其余数据集对齐，AFlow 由 `MODE=eval`（评的是手写 `round_00_initial.py`、却摊一份 thinking-enabled 旧 search）改为 `MODE=both`，deployment 与 search 首次同源且同为 no-thinking。",
        "",
        "AFlow search 的 thinking 配置：MuSiQue=disabled，GSM-Hard=enabled，MultiPL-E-8Lang=disabled，MATH=disabled，Conifer=disabled。配置分别记录在上表的 `config.env` 中。",
        "",
        "Conifer 于 2026-09-14 加入，5 行。JCA 取用户指定的 coverage 88.90 / explicit 95.90 一轮（`rl_mas_eval_forever_20260906/round_4`，同时是 111 轮里 hard_score 最高且仍在盘上的一轮）；MAD/AgentVerse/GPTSwarm 取三臂最新同时存活的 `four_arm_eval_forever_20260909/round_27`；AFlow 沿用 MuSiQue AFlow 的惯例取官方 5-roll 最高的 `130_conifer_aflow_eval_x5/r04`，其 search 是独立批次 `030_conifer_aflow_search`（20×20，no-thinking，21 nodes / 420 次 dev 评估），`130` 的 `summary.json` 里 `reused_search.selected_source` 指向该 search 的 `round_01_proposed.py`，两端同源。Conifer self-RL（sas14b_rl）27 轮分数俱全但没有任何一轮的 `trajectories.jsonl` 留在盘上，Conifer zero-shot 的 5 个输出目录全部为空，两者都不入表，也不以别的 round 代填。",
        "",
        "Conifer 的协议形状还有两处与 `baseline/` 下同名方法不同，读调用数时要注意。其一，MAD 与 AgentVerse 的每条轨迹末尾都有一个 harness 自己写的 `confirm_stop` step（MAD 是 controller 按多数票选定终稿，AgentVerse 是 controller 采纳最后一次组队答案），`raw_output` 为空串、不对应任何请求，已从逻辑调用中剔除；不剔除的话 MAD 会记成 10.000 次/题而不是协议规定的 9.000。其二，Conifer 的 AgentVerse 没有单独的 recruiter 调用，每轮就是 3 个 expert 加 1 次 evaluator，所以每题落在 4 / 8 / 12 次而不是另外四个数据集的 5 / 9 / 13 次。另外 Conifer JCA 有 4 道题整条轨迹为空（彻底失败，`hard_score = 0`），按惯例仍计入分母。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "/data/conda_envs/qwen35/bin/python analysis/coordination_fingerprints/model_call_count_analysis.py",
        "```",
        "",
        f"输出文件：`{output_path}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
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
    parser.add_argument("--multipl-e-zero-shot", type=Path, default=DEFAULT_MULTIPL_E_ZERO_SHOT)
    parser.add_argument("--multipl-e-jca", type=Path, default=DEFAULT_MULTIPL_E_JCA)
    parser.add_argument("--multipl-e-mad", type=Path, default=DEFAULT_MULTIPL_E_MAD)
    parser.add_argument("--multipl-e-agentverse", type=Path, default=DEFAULT_MULTIPL_E_AGENTVERSE)
    parser.add_argument("--multipl-e-gptswarm", type=Path, default=DEFAULT_MULTIPL_E_GPTSWARM)
    parser.add_argument("--multipl-e-aflow", type=Path, default=DEFAULT_MULTIPL_E_AFLOW)
    parser.add_argument("--multipl-e-self-rl", type=Path, default=DEFAULT_MULTIPL_E_SELF_RL)
    parser.add_argument("--math-mad", type=Path, default=DEFAULT_MATH_MAD)
    parser.add_argument("--math-agentverse", type=Path, default=DEFAULT_MATH_AGENTVERSE)
    parser.add_argument("--math-gptswarm", type=Path, default=DEFAULT_MATH_GPTSWARM)
    parser.add_argument("--math-aflow", type=Path, default=DEFAULT_MATH_AFLOW)
    parser.add_argument("--math-zero-shot", type=Path, default=DEFAULT_MATH_ZERO_SHOT)
    parser.add_argument("--math-jca", type=Path, default=DEFAULT_MATH_JCA)
    parser.add_argument("--math-jca-max-units", type=int, default=3)
    parser.add_argument("--conifer-jca", type=Path, default=DEFAULT_CONIFER_JCA)
    parser.add_argument("--conifer-mad", type=Path, default=DEFAULT_CONIFER_MAD)
    parser.add_argument("--conifer-agentverse", type=Path, default=DEFAULT_CONIFER_AGENTVERSE)
    parser.add_argument("--conifer-gptswarm", type=Path, default=DEFAULT_CONIFER_GPTSWARM)
    parser.add_argument("--conifer-aflow", type=Path, default=DEFAULT_CONIFER_AFLOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.math_jca_max_units < 0:
        parser.error("--math-jca-max-units must be non-negative")

    runs = [
        analyze_run(
            "MuSiQue",
            "JCA",
            args.musique_jca,
            logs_for_default_path(
                args.musique_jca,
                DEFAULT_MUSIQUE_JCA,
                fixed_server_logs(MUSIQUE_JCA_LOG_DIR),
            ),
        ),
        # 2026-09-11 起 MAD / AgentVerse / GPTSwarm 都是带 raw_outputs ledger 的新 run，
        # 不再需要 server-log 回退；JCA 那条（上面）仍然需要。
        analyze_run("MuSiQue", "MAD", args.musique_mad),
        analyze_run("MuSiQue", "AgentVerse", args.musique_agentverse),
        analyze_run("MuSiQue", "AFlow", args.musique_aflow),
        analyze_run("MuSiQue", "GPTSwarm", args.musique_gptswarm),
        analyze_run(
            "GSM-Hard",
            "JCA",
            args.gsm_jca,
            logs_for_default_path(
                args.gsm_jca,
                DEFAULT_GSM_JCA,
                gsm_jca_server_logs(GSM_JCA_LOG_DIR),
            ),
        ),
        analyze_run("GSM-Hard", "MAD", args.gsm_mad),
        analyze_run("GSM-Hard", "AgentVerse", args.gsm_agentverse),
        analyze_run("GSM-Hard", "AFlow", args.gsm_aflow),
        analyze_run("GSM-Hard", "GPTSwarm", args.gsm_gptswarm),
        analyze_run("MultiPL-E-8Lang", "zero-shot", args.multipl_e_zero_shot),
        # 该 run 的 config.env 是 MODEL_A1=Qwen3-8B / A2=Qwen3-4B / A3=Qwen3-1.7B，
        # 与 launcher 默认相反，不传映射会把 8B 的调用记到 1.7B 列。
        analyze_run(
            "MultiPL-E-8Lang", "JCA", args.multipl_e_jca,
            agent_size_map=(("A1", "8B"), ("A2", "4B"), ("A3", "1.7B")),
        ),
        analyze_run("MultiPL-E-8Lang", "MAD", args.multipl_e_mad),
        analyze_run("MultiPL-E-8Lang", "AgentVerse", args.multipl_e_agentverse),
        analyze_run("MultiPL-E-8Lang", "AFlow", args.multipl_e_aflow),
        analyze_run("MultiPL-E-8Lang", "GPTSwarm", args.multipl_e_gptswarm),
        analyze_run("MultiPL-E-8Lang", "self-RL", args.multipl_e_self_rl),
        analyze_run("MATH", "zero-shot", args.math_zero_shot),
        analyze_run("MATH", "JCA", args.math_jca, max_units=args.math_jca_max_units),
        analyze_run("MATH", "MAD", args.math_mad),
        analyze_run("MATH", "AgentVerse", args.math_agentverse),
        analyze_run("MATH", "AFlow", args.math_aflow),
        analyze_run("MATH", "GPTSwarm", args.math_gptswarm),
        # Conifer（2026-09-14 加入）。五个 arm 的 trajectory 都是 run_conifer_mas.py
        # 的统一 MAS schema，所以 protocol 全部按 "jca" 解析。槽位映射从各 run 自己的
        # sampling.model_paths 核对过，是常规的 A1=1.7B / A2=4B / A3=8B（MultiPL-E 那种
        # 反转只在那一批出现）。JCA/MAD/AgentVerse/GPTSwarm 走 forever-loop 落盘，
        # 没有 raw_outputs ledger 也没有 round 级 server log，故 retry_ledger=False。
        analyze_run(
            "Conifer", "JCA", args.conifer_jca,
            retry_ledger=False, protocol="jca", require_generation=True,
        ),
        analyze_run(
            "Conifer", "MAD", args.conifer_mad,
            retry_ledger=False, protocol="jca", require_generation=True,
        ),
        analyze_run(
            "Conifer", "AgentVerse", args.conifer_agentverse,
            retry_ledger=False, protocol="jca", require_generation=True,
        ),
        analyze_run(
            "Conifer", "GPTSwarm", args.conifer_gptswarm,
            retry_ledger=False, protocol="jca", require_generation=True,
        ),
        # AFlow 走 baseline_queue，8412/8412 个 step 都有 raw_outputs，ledger 完整。
        analyze_run(
            "Conifer", "AFlow", args.conifer_aflow,
            protocol="jca", require_generation=True,
        ),
    ]
    search_specs = aflow_search_specs()
    for run in runs:
        if run.method != "AFlow":
            continue
        search_spec = search_specs.get(run.dataset)
        if search_spec is None:
            raise ValueError(f"no AFlow search provenance registered for {run.dataset}")
        run.search = analyze_aflow_search(search_spec, run)
        print(
            f"[search] {run.dataset}: logical={run.search.logical_total} "
            f"completed={run.search.request_total} ({run.search.request_source})",
            flush=True,
        )
    report = render_report(runs, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[report] {args.output}")


if __name__ == "__main__":
    main()
