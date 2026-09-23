"""Configuration for the MAPoRL baseline.

Every knob carries an ``origin`` tag in ``configs/_base.yaml``:

  official          -- the value from MAPoRL's GSM8k config, carried over
  official-adapted  -- the official mechanism, value changed for our setup
  ours              -- our choice (agent count, models, datasets, batch sizes)
  unrecoverable     -- not determinable from the released code; see PAPER_NOTES

``validate()`` is where the non-obvious constraints live. Several of them
encode failure modes of the official code that would otherwise be silent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

AGENTS = ("A1", "A2", "A3")

LEGAL_RULE_MODES = {
    ("last", "all"),
    ("last", "individual"),
    ("discounted_sum", "all"),
    ("discounted_sum", "individual"),
    ("current", "all"),
    ("current", "individual"),
}


@dataclass
class MaporlConfig:
    agent_num: int = 3                      # official: 2
    round_num: int = 3                      # official
    rule_horizon: str = "discounted_sum"    # official
    rule_agent_share: str = "all"           # official
    rule_discount: float = 0.3              # official
    # 4 components: [persuaded, self-generated, influence-same, influence-diff].
    # The only attested sweep is a broadcast scalar over {0, 0.5, 1, 2}.
    alpha: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    correct_threshold: float = 0.5          # official (bonus_rule default)
    wrong_threshold: float = 0.5            # official
    # "dead" reproduces the official strict comparison, which discards the bonus
    # when the peers' mean lands exactly on 0.5 -- only reachable with >2 agents.
    others_tiebreak: str = "dead"
    forward_sign: str = "official"          # official | corrected
    binarize: str = "strict"                # Conifer only: strict | split
    adapter_mode: str = "collaboration"     # collaboration | per_turn | shared
    task_training: bool = False             # official: turn 0 frozen
    value_simplification: bool = True       # official
    # The real ground-truth switch. `no_reward_model` in the official config is
    # dead code (assigned at :368, never read).
    reward_feedback: bool = False
    verifier: str | None = None             # documented future extension
    no_early_stopping: bool = True          # criteria_* = 1.1 upstream


@dataclass
class LoraConfig:
    r: int = 8                              # official
    alpha: int = 16                         # official
    dropout: float = 0.05                   # official
    target_modules: tuple[str, ...] = (
        "up_proj", "down_proj", "gate_proj",
        "k_proj", "q_proj", "v_proj", "o_proj",
    )


@dataclass
class PPOConfig:
    learning_rate: float = 1.0e-5           # official
    kl_coef: float = 0.002                  # official
    vf_coef: float = 0.1                    # official
    num_ppo_epochs: int = 4                 # official
    parallel_agent_updates: bool = False    # ours: independent agents on separate GPUs
    cliprange: float = 0.2                  # official
    cliprange_value: float = 0.2            # official
    gamma: float = 1.0                      # official
    lam: float = 0.95                       # official
    whiten_rewards: bool = False            # official
    num_mini_batches: int = 8               # official
    gradient_accumulation_steps: int = 4    # official
    per_device_train_batch_size: int = 1    # official
    accumulate_mode: str = "official"       # official | true_accumulation
    max_grad_norm: float | None = None      # official does not clip
    ratio_tolerance: float = 1.0e-3


@dataclass
class RolloutConfig:
    temperature: float = 0.7000001          # official (0.7 + 1e-7)
    top_k: int = 0                          # official
    top_p: float = 1.0                      # official
    response_length: int = 512              # official-adapted (official 300)
    min_output_length: int = 50             # official
    # Official forces min_new_tokens == max_new_tokens, which suppresses EOS and
    # trains on the post-answer tail. Off by default; strict_official.yaml turns
    # it on for a fidelity ablation.
    force_fixed_length: bool = False
    penalty_reward_value: float = -10.0     # official
    non_eos_penalty: bool = True            # official-adapted (reloadF's False NameErrors)
    prompts_per_iter: int = 24              # ours
    max_concurrency: int = 32               # ours
    step_retries: int = 1                   # ours


@dataclass
class TrainConfig:
    lora: LoraConfig = field(default_factory=LoraConfig)
    value_lora: LoraConfig = field(
        default_factory=lambda: LoraConfig(r=8, alpha=16, dropout=0.0)
    )
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    enable_thinking: bool = False
    quantization: str = "none"              # ours (official: 4-bit NF4)
    max_encode_skip_frac: float = 0.10


@dataclass
class ServingConfig:
    gpu_plan: str = "split44"
    lora_naming: str = "inplace"
    enable_prefix_caching: bool = False
    host: str = "127.0.0.1"
    # 8401-8403 so MAPoRL and MAGRPO can run side by side.
    ports: dict[str, int] = field(
        default_factory=lambda: {"A1": 8401, "A2": 8402, "A3": 8403}
    )
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
    max_model_len: int = 8192
    startup_timeout_s: int = 900
    vllm_python: str = "/data/conda_envs/drb_py311_clean/bin/python"


@dataclass
class PoolConfig:
    prescreen: bool = False
    keep_band: tuple[int, int] = (1, 2)
    prescreen_sample: int = 3000
    shuffle_seed: int = 20260914


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
    seed: int = 20260914
    reward_mode: str = "execution"          # MultiPL-E only
    task_options: dict[str, Any] = field(default_factory=dict)
    maporl: MaporlConfig = field(default_factory=MaporlConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    mock: bool = False

    def validate(self) -> None:
        errors: list[str] = []
        m = self.maporl

        if len(m.alpha) != 4:
            errors.append(
                f"maporl.alpha must have exactly 4 components, got {len(m.alpha)}"
            )
        elif any(a < 0 for a in m.alpha):
            errors.append(f"maporl.alpha components must be non-negative: {m.alpha}")

        if m.agent_num < 1:
            errors.append("maporl.agent_num must be >= 1")
        if m.round_num < 1:
            errors.append("maporl.round_num must be >= 1")
        if m.agent_num > len(AGENTS):
            errors.append(f"maporl.agent_num exceeds the {len(AGENTS)} configured agents")

        if (m.rule_horizon, m.rule_agent_share) not in LEGAL_RULE_MODES:
            errors.append(
                f"illegal rule mode {(m.rule_horizon, m.rule_agent_share)}; "
                f"legal: {sorted(LEGAL_RULE_MODES)}"
            )
        if m.others_tiebreak not in {"dead", "majority"}:
            errors.append("maporl.others_tiebreak must be dead|majority")
        if m.forward_sign not in {"official", "corrected"}:
            errors.append("maporl.forward_sign must be official|corrected")
        if m.binarize not in {"strict", "split"}:
            errors.append("maporl.binarize must be strict|split")
        if m.adapter_mode not in {"collaboration", "per_turn", "shared"}:
            errors.append("maporl.adapter_mode must be collaboration|per_turn|shared")
        # utils/utils_cooperateLLM.py:500-519 -- with neither task_training nor
        # policy separation, grad_requires_setting falls through and marks
        # nothing trainable.
        if m.adapter_mode == "shared" and not m.task_training:
            errors.append(
                "adapter_mode='shared' with task_training=False leaves no trainable "
                "parameters (the official grad_requires_setting has no else branch)"
            )
        if m.reward_feedback or m.verifier is not None:
            errors.append(
                "verifier mode is not implemented: the official repo ships no "
                "verifier checkpoint and no training data. Keep "
                "reward_feedback=false and verifier=null."
            )
        if not m.no_early_stopping:
            errors.append(
                "early stopping must stay off: the official config disables it "
                "(criteria_* = 1.1), and score_rule's discounted_sum branch "
                "raises on an empty stack when a question finishes early"
            )

        if self.rollout.temperature <= 0:
            errors.append(
                "rollout.temperature must be > 0; the debate's only source of "
                "diversity at turn 0 is sampling"
            )
        if self.rollout.min_output_length < 0:
            errors.append("rollout.min_output_length must be >= 0")

        if self.ppo.num_ppo_epochs < 1:
            errors.append("ppo.num_ppo_epochs must be >= 1")
        if self.ppo.accumulate_mode not in {"official", "true_accumulation"}:
            errors.append("ppo.accumulate_mode must be official|true_accumulation")

        if self.serving.lora_naming not in {"inplace", "versioned"}:
            errors.append("serving.lora_naming must be inplace|versioned")
        # vLLM hashes prefix-cache blocks on the LoRA name only, so an in-place
        # swap lets the previous policy's KV leak into the next rollout.
        if self.serving.lora_naming == "inplace" and self.serving.enable_prefix_caching:
            errors.append(
                "serving.enable_prefix_caching must be false when lora_naming="
                "'inplace': vLLM keys KV blocks on lora_name alone, so an "
                "in-place hot-swap reuses the previous policy's cached KV. "
                "Use lora_naming='versioned' to enable caching safely."
            )
        if self.serving.gpu_plan not in {"split44", "split66", "timeshare", "restart"}:
            errors.append("serving.gpu_plan must be split44|split66|timeshare|restart")
        if self.reward_mode not in {"execution", "proxy"}:
            errors.append("reward_mode must be execution|proxy")

        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))

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
    if isinstance(node, dict):
        if "value" in node and "origin" in node and len(node) == 2:
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
        if key in {"lora", "value_lora"} and isinstance(value, dict):
            kwargs[key] = _build(LoraConfig, value)
        elif isinstance(value, list) and "tuple" in str(known[key].type):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
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
        "maporl": MaporlConfig,
        "ppo": PPOConfig,
        "rollout": RolloutConfig,
        "train": TrainConfig,
        "serving": ServingConfig,
        "pool": PoolConfig,
        "budget": BudgetConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        kwargs[key] = _build(sections[key], value or {}) if key in sections else value
    config = Config(**kwargs)
    config.validate()
    return config
