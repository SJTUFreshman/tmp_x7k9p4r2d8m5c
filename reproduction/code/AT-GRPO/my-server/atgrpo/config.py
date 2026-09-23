"""Configuration: dataclasses, YAML loading, and validation.

Every knob carries an ``origin`` in ``configs/_base.yaml``. The MAGRPO paper's
algorithm box and hyperparameter table were not retrievable in this environment,
so every numeric default here is *ours*, chosen from GRPO-standard practice, and
must not be attributed to Liu et al. See ``PAPER_NOTES.md``.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

AGENTS = ("A1", "A2", "A3")


def serving_replica_groups(spec: str) -> tuple[str, ...]:
    """Return replica groups; commas inside one group mean tensor parallelism."""
    if not isinstance(spec, str):
        raise ValueError(f"invalid GPU specification: {spec!r}")
    raw_groups = spec.split(";")
    if not raw_groups or any(not group.strip() for group in raw_groups):
        raise ValueError(f"invalid GPU specification: {spec!r}")
    groups = tuple(group.strip() for group in raw_groups)
    if any(any(not gpu.strip() for gpu in group.split(",")) for group in groups):
        raise ValueError(f"invalid GPU specification: {spec!r}")
    return groups


def gpu_ids(spec: str) -> tuple[str, ...]:
    return tuple(
        gpu.strip()
        for group in serving_replica_groups(spec)
        for gpu in group.split(",")
        if gpu.strip()
    )


def cpu_affinity_groups(spec: str) -> tuple[str, ...]:
    """Return one ``taskset -c`` CPU list per serving replica."""
    if not isinstance(spec, str):
        raise ValueError(f"invalid CPU affinity specification: {spec!r}")
    raw_groups = spec.split(";")
    if not raw_groups or any(not group.strip() for group in raw_groups):
        raise ValueError(f"invalid CPU affinity specification: {spec!r}")
    groups = tuple(group.strip() for group in raw_groups)
    for group in groups:
        cpu_ids(group)
    return groups


def cpu_ids(spec: str) -> tuple[int, ...]:
    """Expand a taskset-style comma-separated list of CPU IDs and ranges."""
    if not isinstance(spec, str) or not spec.strip() or ";" in spec:
        raise ValueError(f"invalid CPU affinity specification: {spec!r}")
    ids: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid CPU affinity specification: {spec!r}")
        bounds = part.split("-")
        if len(bounds) == 1 and bounds[0].isdigit():
            ids.append(int(bounds[0]))
        elif (
            len(bounds) == 2
            and bounds[0].isdigit()
            and bounds[1].isdigit()
            and int(bounds[0]) <= int(bounds[1])
        ):
            ids.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError(f"invalid CPU affinity specification: {spec!r}")
    return tuple(ids)


@dataclass
class AtgrpoConfig:
    """AT-GRPO knobs. Reference: Zhao et al., arXiv:2510.11062 (ICLR 2026)."""

    agent_num: int = 3
    round_num: int = 3
    # Candidates sampled per (agent, turn) branch point -- the GRPO group size.
    group_size_K: int = 4
    prompts_per_iter: int = 24
    # spine: one group per turn, cost T*K. full_tree: K^T leaves.
    # The paper does not pin this down; see PAPER_NOTES.
    branch_mode: str = "spine"
    inner_epochs: int = 1
    clip_epsilon: float = 0.2
    kl_coef: float = 0.02
    clip_advantage: float = 4.0
    loss_agg: str = "seq_mean"
    std_floor: float = 1e-8
    # A group whose members disagree on the prompt is not a valid GRPO group.
    strict_fingerprint: bool = True
    max_degenerate_frac: float = 0.90
    degenerate_patience: int = 3


@dataclass
class LoraConfig:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class TrainConfig:
    learning_rate: dict[str, float] = field(
        default_factory=lambda: {"A1": 2.0e-5, "A2": 1.5e-5, "A3": 1.0e-5}
    )
    per_device_batch_size: dict[str, int] = field(
        default_factory=lambda: {"A1": 4, "A2": 2, "A3": 1}
    )
    lora: LoraConfig = field(default_factory=LoraConfig)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # Constant, not cosine: an online loop must be resumable, and cosine needs a
    # horizon known up front.
    lr_schedule: str = "constant_with_warmup"
    warmup_iters: int = 5
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    enable_thinking: bool = False
    daemon_timeout_s: int = 1800
    max_encode_skip_frac: float = 0.10
    ratio_tolerance: float = 1e-3


@dataclass
class RolloutConfig:
    joint_mode: str = "synchronous"  # synchronous | sequential
    t_max: int = 4
    temperature: float = 1.0  # must be > 0: group variance is the whole signal
    top_p: float = 1.0  # truncated sampling biases the importance ratio
    max_new_tokens: int = 1024
    step_retries: int = 1
    max_concurrency: int = 32
    start_agent: str = "A1"


@dataclass
class ServingConfig:
    gpu_plan: str = "split44"  # split44 | split53 | timeshare | restart
    lora_naming: str = "inplace"  # inplace | versioned
    enable_prefix_caching: bool = False
    ports: dict[str, int] = field(
        default_factory=lambda: {"A1": 8401, "A2": 8402, "A3": 8403}
    )
    host: str = "127.0.0.1"
    models: dict[str, str] = field(
        default_factory=lambda: {
            "A1": "/data/wangyuheng/models/Qwen3-1.7B",
            "A2": "/data/wangyuheng/models/Qwen3-4B",
            "A3": "/data/wangyuheng/models/Qwen3-8B",
        }
    )
    serve_gpus: dict[str, str] = field(
        default_factory=lambda: {"A1": "0", "A2": "1", "A3": "2,3"}
    )
    train_gpus: dict[str, str] = field(
        default_factory=lambda: {"A1": "4", "A2": "5", "A3": "6,7"}
    )
    gpu_memory_utilization: dict[str, float] = field(
        default_factory=lambda: {"A1": 0.32, "A2": 0.38, "A3": 0.42}
    )
    cpu_threads_per_server: int = 8
    cpu_affinity: dict[str, str] = field(
        default_factory=lambda: {
            "A1": "0-7",
            "A2": "8-15",
            "A3": "16-31",
        }
    )
    max_model_len: int = 8192
    startup_timeout_s: int = 900
    vllm_python: str = "/data/conda_envs/drb_py311_clean/bin/python"


@dataclass
class PoolConfig:
    prescreen: bool = True
    keep_band: tuple[int, int] = (1, 7)
    prescreen_sample: int = 3000
    shuffle_seed: int = 20260911


@dataclass
class BudgetConfig:
    max_iterations: int = 120
    max_wall_clock_hours: float = 24.0
    eval_every: int = 10
    eval_subset: int = 200
    save_every: int = 10
    keep_last: int = 3


@dataclass
class Config:
    task: str = "gsm_hard"
    run_id: str = "dev"
    seed: int = 20260911
    reward_mode: str = "execution"
    task_options: dict[str, Any] = field(default_factory=dict)
    atgrpo: AtgrpoConfig = field(default_factory=AtgrpoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    mock: bool = False

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        errors: list[str] = []
        if self.atgrpo.group_size_K < 2:
            errors.append(
                "atgrpo.group_size_K must be >= 2: a group of one makes the "
                "baseline the sample itself, which is the degeneracy AT-GRPO's "
                "tree sampling exists to prevent"
            )
        if self.atgrpo.branch_mode not in {"spine", "full_tree"}:
            errors.append("atgrpo.branch_mode must be spine|full_tree")
        if self.rollout.temperature <= 0:
            errors.append(
                "rollout.temperature must be > 0: with greedy decoding all K "
                "candidates sampled from one frozen prompt are identical, so "
                "every (agent, turn) group has zero variance and zero advantage"
            )
        if self.atgrpo.inner_epochs < 1:
            errors.append("atgrpo.inner_epochs must be >= 1")
        if self.budget.max_wall_clock_hours < 0:
            errors.append(
                "budget.max_wall_clock_hours must be >= 0 (use 0 to disable it)"
            )
        if self.atgrpo.loss_agg not in {"seq_mean", "token_mean"}:
            errors.append("atgrpo.loss_agg must be seq_mean|token_mean")
        if self.rollout.joint_mode not in {"synchronous", "sequential"}:
            errors.append("rollout.joint_mode must be synchronous|sequential")
        if self.serving.lora_naming not in {"inplace", "versioned"}:
            errors.append("serving.lora_naming must be inplace|versioned")
        # The stale-KV hazard: vLLM keys prefix-cache blocks on the LoRA *name*
        # only, so an in-place swap lets blocks from the previous adapter be
        # reused by the next iteration's rollout -- silently off-policy.
        if self.serving.lora_naming == "inplace" and self.serving.enable_prefix_caching:
            errors.append(
                "serving.enable_prefix_caching must be false when lora_naming="
                "'inplace': vLLM hashes KV blocks on lora_name only, so an "
                "in-place hot-swap would reuse the previous policy's cached KV. "
                "Use lora_naming='versioned' to enable prefix caching safely."
            )
        if self.serving.gpu_plan not in {
            "split44", "split53", "timeshare", "restart"
        }:
            errors.append(
                "serving.gpu_plan must be split44|split53|timeshare|restart"
            )
        try:
            serve_assignments = [
                gpu
                for agent in AGENTS
                for gpu in gpu_ids(self.serving.serve_gpus[agent])
            ]
            train_assignments = [
                gpu
                for agent in AGENTS
                for gpu in gpu_ids(self.serving.train_gpus[agent])
            ]
            serve = set(serve_assignments)
            train = set(train_assignments)
            duplicate_serve = sorted(
                gpu for gpu in serve if serve_assignments.count(gpu) > 1
            )
            duplicate_train = sorted(
                gpu for gpu in train if train_assignments.count(gpu) > 1
            )
            if duplicate_serve:
                errors.append(
                    f"serving GPUs assigned more than once: {duplicate_serve}"
                )
            if duplicate_train:
                errors.append(
                    f"training GPUs assigned more than once: {duplicate_train}"
                )
            if serve & train:
                errors.append(
                    f"serving and training GPUs overlap: {sorted(serve & train)}"
                )
            if self.serving.gpu_plan == "split53" and (
                len(serve_assignments) != 5 or len(train_assignments) != 3
            ):
                errors.append(
                    "serving.gpu_plan='split53' requires exactly 5 serving GPU "
                    f"assignments and 3 training GPU assignments, got "
                    f"{len(serve_assignments)} and {len(train_assignments)}"
                )
        except (KeyError, ValueError) as exc:
            errors.append(f"invalid serving GPU map: {exc}")
        if self.serving.cpu_threads_per_server < 1:
            errors.append("serving.cpu_threads_per_server must be >= 1")
        try:
            affinity_owners: dict[int, list[str]] = {}
            for agent in AGENTS:
                replica_groups = serving_replica_groups(
                    self.serving.serve_gpus[agent]
                )
                affinity_groups = cpu_affinity_groups(
                    self.serving.cpu_affinity[agent]
                )
                if len(affinity_groups) != len(replica_groups):
                    errors.append(
                        f"serving.cpu_affinity[{agent!r}] must define exactly "
                        f"{len(replica_groups)} group(s), one per serving replica; "
                        f"got {len(affinity_groups)}"
                    )
                for replica_index, (replica_group, affinity) in enumerate(
                    zip(replica_groups, affinity_groups)
                ):
                    assigned = cpu_ids(affinity)
                    unique = set(assigned)
                    tensor_parallel_size = len(
                        [gpu for gpu in replica_group.split(",")]
                    )
                    required_cpus = (
                        self.serving.cpu_threads_per_server * tensor_parallel_size
                    )
                    if len(unique) != len(assigned):
                        errors.append(
                            f"serving.cpu_affinity[{agent!r}] replica "
                            f"{replica_index} repeats CPU IDs"
                        )
                    if len(unique) < required_cpus:
                        errors.append(
                            f"serving.cpu_affinity[{agent!r}] replica "
                            f"{replica_index} assigns {len(unique)} CPU(s), fewer "
                            f"than the required {required_cpus} "
                            "(serving.cpu_threads_per_server="
                            f"{self.serving.cpu_threads_per_server} x "
                            f"tensor_parallel_size={tensor_parallel_size})"
                        )
                    for cpu in unique:
                        affinity_owners.setdefault(cpu, []).append(
                            f"{agent}[{replica_index}]"
                        )
            overlaps = {
                cpu: owners
                for cpu, owners in affinity_owners.items()
                if len(owners) > 1
            }
            if overlaps:
                errors.append(
                    "serving CPU affinity groups overlap: "
                    + ", ".join(
                        f"CPU {cpu}={owners}" for cpu, owners in sorted(overlaps.items())
                    )
                )
        except (KeyError, ValueError) as exc:
            errors.append(f"invalid serving CPU affinity map: {exc}")
        if self.reward_mode not in {"execution", "proxy"}:
            errors.append("reward_mode must be execution|proxy")
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _strip_origins(node: Any) -> Any:
    """Drop the ``origin:`` annotations before building dataclasses."""
    if isinstance(node, dict):
        if set(node) == {"value", "origin"} or (
            "value" in node and "origin" in node and len(node) == 2
        ):
            return _strip_origins(node["value"])
        return {k: _strip_origins(v) for k, v in node.items() if k != "origin"}
    if isinstance(node, list):
        return [_strip_origins(v) for v in node]
    return node


def _build(cls, payload: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} for {cls.__name__}")
        field_type = known[key].type
        if key == "lora" and isinstance(value, dict):
            kwargs[key] = _build(LoraConfig, value)
        elif isinstance(value, list) and "tuple" in str(field_type):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, following a single ``extends:`` chain."""
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = payload.pop("extends", None)
    if parent:
        base_path = (path.parent / parent).resolve()
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        base.pop("extends", None)
        payload = _merge(base, payload)
    payload = _strip_origins(payload)
    if overrides:
        payload = _merge(payload, overrides)

    sections = {
        "atgrpo": AtgrpoConfig,
        "train": TrainConfig,
        "rollout": RolloutConfig,
        "serving": ServingConfig,
        "pool": PoolConfig,
        "budget": BudgetConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in sections:
            kwargs[key] = _build(sections[key], value or {})
        else:
            kwargs[key] = value
    config = Config(**kwargs)
    config.validate()
    return config
