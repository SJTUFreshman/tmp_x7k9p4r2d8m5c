"""Single-device MAPoRL policy and per-turn value training."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .encode import apply_chat_template
from .ppo import masked_mean, ppo_losses, prepare_advantages
from .trajectory import TurnRecord


@dataclass
class PreparedTurn:
    turn: int
    rows: list[dict[str, Any]]
    old_logprobs: Any
    ref_logprobs: Any
    values: Any
    advantages: Any
    returns: Any
    padding_mask: Any
    padding_mask_p1: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class AgentTrainer:
    def __init__(
        self,
        config: Config,
        agent: str,
        device: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config.validate()
        if config.maporl.adapter_mode != "collaboration" or config.maporl.task_training:
            raise ValueError("this runtime requires collaboration adapters and frozen turn 0")
        if config.train.quantization != "none":
            raise ValueError("the PPO runtime requires the unquantized base model")
        if config.ppo.per_device_train_batch_size != 1:
            raise ValueError("the sequential PPO runtime requires per-device batch size 1")
        if config.rollout.top_k not in (0, -1) or config.rollout.top_p != 1.0:
            raise ValueError("PPO requires the full temperature-scaled sampling distribution")
        if config.ppo.gradient_accumulation_steps < 1 or config.ppo.num_mini_batches < 1:
            raise ValueError("PPO accumulation and mini-batch counts must be positive")
        self.config = config
        self.agent = agent
        gpu = config.serving.train_gpus[agent]
        if device is None and "," in gpu:
            raise ValueError("AgentTrainer expects one logical training GPU")
        self.device = torch.device(device or f"cuda:{gpu}")
        self.model_path = config.serving.models[agent]
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="sdpa",
                device_map={"": str(self.device)},
            )
        else:
            model = model.to(self.device)
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise TypeError("PPO runtime requires a decoder model with model and lm_head")
        model.config.use_cache = False
        seed = config.seed + 1009 * int(agent.removeprefix("A"))
        torch.random.default_generator.manual_seed(seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(seed)

        def lora_config(values: Any) -> Any:
            return LoraConfig(
                r=values.r,
                lora_alpha=values.alpha,
                lora_dropout=0.0,
                target_modules=list(values.target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

        self.model = get_peft_model(model, lora_config(config.train.lora), adapter_name="policy")
        for turn in range(config.maporl.round_num):
            self.model.add_adapter(f"value_{turn}", lora_config(config.train.value_lora))
        hidden_size = model.config.hidden_size
        self.value_heads = torch.nn.ModuleDict(
            {
                str(turn): torch.nn.Linear(hidden_size, 1, bias=False)
                for turn in range(config.maporl.round_num)
            }
        ).to(device=self.device, dtype=torch.float32)
        for head in self.value_heads.values():
            torch.nn.init.normal_(head.weight, std=1.0 / math.sqrt(hidden_size + 1))
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            for attribute in ("attention_dropout", "hidden_dropout", "activation_dropout"):
                if isinstance(getattr(module, attribute, None), (int, float)):
                    setattr(module, attribute, 0.0)
        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        self.model.train()
        self.value_heads.train()
        self.parameters = {
            name: parameter for name, parameter in self.model.named_parameters()
            if ".lora_" in name
        }
        self.parameters.update(
            {f"value_heads.{name}": parameter for name, parameter in self.value_heads.named_parameters()}
        )
        self.optimizer = torch.optim.AdamW(
            list(self.parameters.values()), lr=config.ppo.learning_rate, weight_decay=0.0
        )
        self.optimizer_steps = 0
        self.accumulation_microsteps = 0
        self.last_prepare_stats: dict[str, Any] = {}
        self._activate("policy")

    def policy_key(self, turn: int) -> str | None:
        if not 0 <= turn < self.config.maporl.round_num:
            raise ValueError(f"invalid turn {turn}")
        return None if turn == 0 else "policy"

    def _activate(self, adapter: str, value_turn: int | None = None) -> None:
        self.model.set_adapter(adapter)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(".lora_" in name and f".{adapter}." in name)
        for turn, head in self.value_heads.items():
            head.requires_grad_(value_turn is not None and int(turn) == value_turn)

    def _inputs(self, token_ids: list[int]) -> dict[str, Any]:
        import torch

        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "use_cache": False}

    def _hidden(self, token_ids: list[int]) -> Any:
        return self.model.get_base_model().model(**self._inputs(token_ids)).last_hidden_state

    def _policy_logprobs(self, row: dict[str, Any], *, reference: bool = False) -> Any:
        import torch
        import torch.nn.functional as functional

        key = self.policy_key(row["turn"])
        self._activate(key or "policy")
        context = self.model.disable_adapter() if reference or key is None else nullcontext()
        with context:
            hidden = self._hidden(row["prompt_ids"] + row["response_ids"])
            prefix = len(row["prompt_ids"])
            count = len(row["response_ids"])
            predicting = hidden[:, prefix - 1:prefix + count - 1]
            targets = torch.tensor(row["response_ids"], dtype=torch.long, device=self.device)
            logprobs = []
            for start in range(0, count, 64):
                end = min(start + 64, count)
                logits = self.model.get_base_model().lm_head(predicting[:, start:end]).float()
                logits = logits / row["temperature"]
                if row["min_tokens"]:
                    stop_ids = row["stop_token_ids"]
                    if start < row["min_tokens"] and stop_ids:
                        stop_until = min(end, row["min_tokens"]) - start
                        logits[:, :stop_until, stop_ids] = -float("inf")
                logprobs.append(
                    -functional.cross_entropy(logits[0], targets[start:end], reduction="none")
                )
            return torch.cat(logprobs)

    def _values(self, row: dict[str, Any]) -> Any:
        self._activate(f"value_{row['turn']}", row["turn"])
        hidden = self._hidden(row["critic_prompt_ids"] + row["response_ids"])
        prefix = len(row["critic_prompt_ids"])
        response_states = hidden[:, prefix - 1:prefix + len(row["response_ids"])]
        return self.value_heads[str(row["turn"])](response_states.float())[0, :, 0]

    def _encode_record(self, record: TurnRecord) -> dict[str, Any] | None:
        if getattr(record, "guided_decoding", False):
            raise ValueError("guided decoding changes the behavior distribution and is unsupported in PPO")
        prompt_ids = getattr(record, "prompt_token_ids", None)
        response_ids = getattr(record, "response_token_ids", None)
        if not prompt_ids or not response_ids:
            raise ValueError("PPO requires the exact generated prompt and response token IDs")
        if any(not isinstance(token, int) or token < 0 for token in prompt_ids + response_ids):
            raise ValueError("invalid exact token IDs")
        vocab_size = self.model.get_base_model().config.vocab_size
        if max(prompt_ids + response_ids) >= vocab_size:
            raise ValueError("generation token IDs do not belong to this model vocabulary")
        turn = int(record.turn)
        self.policy_key(turn)
        model_name = getattr(record, "model_name", None)
        if turn == 0 and model_name != f"{self.agent}_base":
            raise ValueError(f"turn 0 must be sampled from {self.agent}_base, got {model_name!r}")
        if turn > 0 and not (
            model_name == self.agent or
            isinstance(model_name, str) and model_name.startswith(f"{self.agent}_it")
        ):
            raise ValueError(f"collaboration turn must use the {self.agent} policy, got {model_name!r}")
        rollout_logprobs = getattr(record, "response_logprobs", None)
        if rollout_logprobs is None:
            raise ValueError("PPO requires the actual behavior-policy token logprobs")
        if len(rollout_logprobs) != len(response_ids):
            raise ValueError("generation token IDs and logprobs have different lengths")
        generation = getattr(record, "generation_config", {}) or {}
        if len(response_ids) > self.config.rollout.response_length:
            raise ValueError("response exceeds the configured generation width")
        if generation.get("max_tokens", self.config.rollout.response_length) != self.config.rollout.response_length:
            raise ValueError("rollout generation width does not match PPO configuration")
        temperature = float(generation.get("temperature", self.config.rollout.temperature))
        if abs(temperature - self.config.rollout.temperature) > 1e-6:
            raise ValueError("rollout temperature does not match the PPO configuration")
        if generation.get("top_k", 0) not in (-1, 0) or generation.get("top_p", 1.0) != 1.0:
            raise ValueError("truncated sampling distributions are unsupported in PPO")
        if any(generation.get(key, default) != default for key, default in (
            ("min_p", 0.0), ("repetition_penalty", 1.0),
            ("presence_penalty", 0.0), ("frequency_penalty", 0.0),
        )):
            raise ValueError("PPO cannot reconstruct generation logits processors")
        min_tokens = int(generation.get("min_tokens", 0))
        generation_config = self.model.get_base_model().generation_config
        stop_ids = generation.get("stop_token_ids") or generation_config.eos_token_id or []
        if isinstance(stop_ids, int):
            stop_ids = [stop_ids]
        if min_tokens and not stop_ids:
            raise ValueError("cannot reconstruct min_tokens without stop token IDs")
        if record.reward is None or not math.isfinite(float(record.reward)):
            raise ValueError("PPO requires a finite shaped reward")
        if self.config.train.enable_thinking != generation.get(
            "enable_thinking", self.config.train.enable_thinking
        ):
            raise ValueError("rollout thinking mode does not match training")
        if self.config.maporl.value_simplification:
            original_messages = []
            for message in record.prompt_messages:
                original_messages.append(message)
                if message["role"] == "user":
                    break
            if not original_messages or original_messages[-1]["role"] != "user":
                raise ValueError("value simplification needs the original user question")
            critic_text = apply_chat_template(
                self.tokenizer, original_messages, add_generation_prompt=True,
                enable_thinking=self.config.train.enable_thinking,
            )
            critic_ids = list(self.tokenizer(critic_text, add_special_tokens=False)["input_ids"])
        else:
            critic_ids = list(prompt_ids)
        if not critic_ids:
            raise ValueError("the critic has an empty conditioning prompt")
        if max(len(prompt_ids), len(critic_ids)) + len(response_ids) > self.config.train.max_seq_length:
            return None
        return {
            "turn": turn,
            "prompt_ids": list(prompt_ids),
            "response_ids": list(response_ids),
            "critic_prompt_ids": critic_ids,
            "reward": float(record.reward),
            "temperature": temperature,
            "min_tokens": min_tokens,
            "stop_token_ids": list(stop_ids),
            "rollout_logprobs": rollout_logprobs,
            "logprobs_mode": generation.get("logprobs_mode", "processed_logprobs"),
        }

    def prepare(self, records: list[TurnRecord], iteration: int | None = None) -> list[PreparedTurn]:
        import torch

        selected = [record for record in records if record.agent == self.agent]
        if not selected:
            raise ValueError(f"no records for {self.agent}")
        groups: dict[int, list[dict[str, Any]]] = {}
        turn_counts: dict[int, int] = {}
        skipped = 0
        for record in selected:
            if iteration is not None:
                expected_version = "base" if record.turn == 0 else str(iteration)
                if record.adapter_version != expected_version:
                    raise ValueError(
                        f"stale behavior adapter for turn {record.turn}: "
                        f"expected {expected_version!r}, got {record.adapter_version!r}"
                    )
            sample_index = turn_counts.get(record.turn, 0)
            turn_counts[record.turn] = sample_index + 1
            row = self._encode_record(record)
            if row is None:
                skipped += 1
            else:
                row["sample_index"] = sample_index
                groups.setdefault(row["turn"], []).append(row)
        if len(set(turn_counts.values())) != 1:
            raise ValueError("debate turns must contain aligned question counts")
        self.last_prepare_stats = {
            "received_records": len(selected), "encoded_records": len(selected) - skipped,
            "overlength_records": skipped, "encode_skip_frac": skipped / len(selected),
            "dropout": 0.0,
        }
        if skipped / len(selected) > self.config.train.max_encode_skip_frac:
            raise RuntimeError(f"too many exact on-policy contexts exceed max_seq_length: {self.last_prepare_stats}")
        if set(groups) != set(range(self.config.maporl.round_num)):
            raise ValueError("PPO requires usable records for every debate turn")
        prepared = []
        engine_differences = []
        behavior_ratios = []
        policy_behavior_ratios = []
        identity_errors = []
        for turn, rows in sorted(groups.items()):
            width = min(
                max(len(row["response_ids"]) for row in rows) + 1,
                self.config.rollout.response_length,
            )
            shape = (len(rows), width)
            old_logprobs = torch.zeros(shape, dtype=torch.float32)
            ref_logprobs = torch.zeros(shape, dtype=torch.float32)
            values = torch.zeros(shape, dtype=torch.float32)
            padding_mask = torch.ones(shape, dtype=torch.bool)
            padding_mask_p1 = torch.ones(shape, dtype=torch.bool)
            lengths = torch.tensor([len(row["response_ids"]) - 1 for row in rows])
            scores = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
            with torch.no_grad():
                for index, row in enumerate(rows):
                    count = len(row["response_ids"])
                    old = self._policy_logprobs(row).detach().cpu()
                    if not torch.isfinite(old).all():
                        raise RuntimeError("nonfinite old policy logprobs")
                    if index == 0:
                        unchanged = self._policy_logprobs(row).detach().cpu()
                        identity_errors.append(float((torch.exp(unchanged - old) - 1).abs().max()))
                    ref_logprobs[index, :count] = self._policy_logprobs(row, reference=True).detach().cpu()
                    value_count = min(count + 1, width)
                    values[index, :value_count] = self._values(row)[:value_count].detach().cpu()
                    padding_mask[index, :count] = False
                    padding_mask_p1[index, :count + 1] = False
                    if row["logprobs_mode"] != "processed_logprobs":
                        raise ValueError("rollout logprob validation requires processed_logprobs")
                    sampled = torch.tensor(row["rollout_logprobs"], dtype=torch.float32)
                    if not torch.isfinite(sampled).all() or (sampled > 1e-6).any():
                        raise ValueError("rollout contains invalid sampled token logprobs")
                    old_logprobs[index, :count] = sampled
                    engine_differences.append((sampled - old).abs())
                    behavior_ratios.append(torch.exp(old - sampled))
                    if self.policy_key(turn) is not None:
                        policy_behavior_ratios.append(torch.exp(old - sampled))
            if max(identity_errors) > self.config.ppo.ratio_tolerance:
                raise RuntimeError(
                    f"the unchanged policy failed the on-policy ratio check: {max(identity_errors)}"
                )
            advantages = prepare_advantages(
                old_logprobs - ref_logprobs, values, scores, lengths, lengths + 1,
                padding_mask, padding_mask_p1,
                kl_coef=self.config.ppo.kl_coef, gamma=self.config.ppo.gamma,
                lam=self.config.ppo.lam, whiten_rewards=self.config.ppo.whiten_rewards,
            )
            if not all(torch.isfinite(value).all() for value in (values, ref_logprobs, advantages["returns"])):
                raise RuntimeError("nonfinite PPO values or returns")
            prepared.append(PreparedTurn(
                turn, rows, old_logprobs, ref_logprobs, values,
                advantages["advantages"], advantages["returns"], padding_mask, padding_mask_p1,
            ))
        self.last_prepare_stats["unchanged_ratio_max_error"] = max(identity_errors)
        if engine_differences:
            differences = torch.cat(engine_differences)
            ratios = torch.cat(behavior_ratios)
            limits = {"mae": 0.05, "p99": 0.5, "max": 16.0}
            policy_ratios = torch.cat(policy_behavior_ratios)
            cliprange = self.config.ppo.cliprange
            self.last_prepare_stats.update(
                rollout_hf_logprob_mae=float(differences.mean()),
                rollout_hf_logprob_max_error=float(differences.max()),
                rollout_hf_logprob_p99_error=float(torch.quantile(differences, 0.99)),
                behavior_ratio={
                    "mean": float(ratios.mean()), "min": float(ratios.min()),
                    "max": float(ratios.max()),
                    "p99_absolute_error": float(torch.quantile((ratios - 1.0).abs(), 0.99)),
                },
                behavior_logprob_limits=limits,
                policy_ratio_clip_frac=float((
                    (policy_ratios < 1.0 - cliprange) | (policy_ratios > 1.0 + cliprange)
                ).float().mean()),
                old_logprob_source="rollout_processed_logprobs",
                unchanged_ratio_check="repeated_hf_forward",
            )
            if (
                self.last_prepare_stats["rollout_hf_logprob_mae"] > limits["mae"] or
                self.last_prepare_stats["rollout_hf_logprob_p99_error"] > limits["p99"] or
                self.last_prepare_stats["rollout_hf_logprob_max_error"] > limits["max"]
            ):
                raise RuntimeError(
                    f"behavior-policy logprob mismatch exceeds numerical tolerance: {self.last_prepare_stats}"
                )
        return prepared

    def _one_backward(self, prepared: PreparedTurn, index: int, loss_scale: float) -> dict[str, float]:
        import torch

        row = prepared.rows[index]
        count = len(row["response_ids"])
        width = min(count + 1, prepared.old_logprobs.shape[1])
        old = prepared.old_logprobs[index:index + 1, :width].to(self.device)
        advantage = prepared.advantages[index:index + 1, :width].to(self.device)
        old_values = prepared.values[index:index + 1, :width].to(self.device)
        returns = prepared.returns[index:index + 1, :width].to(self.device)
        mask = prepared.padding_mask[index:index + 1, :width].to(self.device)
        value_mask = prepared.padding_mask_p1[index:index + 1, :width].to(self.device)
        train_policy = self.policy_key(prepared.turn) is not None
        with nullcontext() if train_policy else torch.no_grad():
            policy_logprobs = self._policy_logprobs(row)
            new = torch.cat((policy_logprobs, policy_logprobs.new_zeros(width - count))).unsqueeze(0)
            policy_loss, stats = ppo_losses(
                new, old, advantage, old_values, old_values, returns, mask, value_mask,
                cliprange=self.config.ppo.cliprange,
                cliprange_value=self.config.ppo.cliprange_value, vf_coef=0.0,
            )
        if train_policy:
            if not torch.isfinite(policy_loss):
                raise RuntimeError("nonfinite PPO policy loss")
            (policy_loss * loss_scale).backward()
        predicted_values = self._values(row)[:width].unsqueeze(0)
        clipped_values = torch.clamp(
            predicted_values, old_values - self.config.ppo.cliprange_value,
            old_values + self.config.ppo.cliprange_value,
        )
        value_loss = 0.5 * masked_mean(
            torch.maximum((predicted_values - returns).square(), (clipped_values - returns).square()),
            ~value_mask,
        )
        if not torch.isfinite(value_loss):
            raise RuntimeError("nonfinite PPO value loss")
        (self.config.ppo.vf_coef * value_loss * loss_scale).backward()
        stats.update(
            vf_loss=float(value_loss.detach()),
            loss=float(policy_loss.detach()) + self.config.ppo.vf_coef * float(value_loss.detach()),
            policy_trained=float(train_policy),
            vf_clipfrac=float(masked_mean(
                ((clipped_values - returns).square() > (predicted_values - returns).square()).float(),
                ~value_mask,
            ).detach()),
        )
        return stats

    def _step_optimizer(self) -> None:
        import torch

        parameters = [parameter for parameter in self.parameters.values() if parameter.grad is not None]
        if not parameters or not all(torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise RuntimeError("missing or nonfinite PPO gradients")
        if self.config.ppo.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.ppo.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_steps += 1

    def _fingerprints(self) -> dict[str, str]:
        hashes = {"policy": hashlib.sha256()}
        hashes.update({f"value_{turn}": hashlib.sha256() for turn in range(self.config.maporl.round_num)})
        hashes.update({f"head_{turn}": hashlib.sha256() for turn in range(self.config.maporl.round_num)})
        for name, parameter in sorted(self.parameters.items()):
            if name.startswith("value_heads."):
                key = f"head_{name.split('.')[1]}"
            else:
                key = name.split(".lora_", 1)[1].split(".")[1]
            hashes[key].update(name.encode())
            hashes[key].update(parameter.detach().cpu().float().numpy().tobytes())
        return {name: digest.hexdigest() for name, digest in hashes.items()}

    def update(self, records: list[TurnRecord], iteration: int = 0) -> dict[str, Any]:
        prepared_turns = self.prepare(records, iteration=iteration)
        before = self._fingerprints()
        statistics: list[dict[str, float]] = []
        per_turn: dict[int, list[dict[str, float]]] = {}
        previous_steps = self.optimizer_steps
        generator = random.Random(self.config.seed + iteration * 10007 + int(self.agent[1:]))
        accumulation = self.config.ppo.gradient_accumulation_steps
        indexed_turns = [
            (prepared, {row["sample_index"]: index for index, row in enumerate(prepared.rows)})
            for prepared in prepared_turns
        ]
        sample_indices = sorted({sample for _, indices in indexed_turns for sample in indices})
        for epoch in range(self.config.ppo.num_ppo_epochs):
            order = list(sample_indices)
            generator.shuffle(order)
            mini_size = max(1, math.ceil(len(order) / self.config.ppo.num_mini_batches))
            for mini_start in range(0, len(order), mini_size):
                mini_batch = order[mini_start:mini_start + mini_size]
                for sample_index in mini_batch:
                    self.accumulation_microsteps += 1
                    synchronized = self.accumulation_microsteps % accumulation == 0
                    for prepared, indices in indexed_turns:
                        if sample_index not in indices:
                            continue
                        stats = self._one_backward(prepared, indices[sample_index], 1.0 / accumulation)
                        statistics.append(stats)
                        per_turn.setdefault(prepared.turn, []).append(stats)
                        if synchronized and self.config.ppo.accumulate_mode == "official":
                            self._step_optimizer()
                    if synchronized and self.config.ppo.accumulate_mode == "true_accumulation":
                        self._step_optimizer()
        means = {
            key: sum(stat[key] for stat in statistics) / len(statistics)
            for key in statistics[0]
        }
        after = self._fingerprints()
        return {
            **self.last_prepare_stats, **means,
            "agent": self.agent, "iteration": iteration,
            "optimizer_steps": self.optimizer_steps - previous_steps,
            "optimizer_steps_total": self.optimizer_steps,
            "accumulation_microsteps_total": self.accumulation_microsteps,
            "pending_accumulation_microsteps": self.accumulation_microsteps % accumulation,
            "policy_changed": before["policy"] != after["policy"],
            "critic_changed": {
                str(turn): (
                    before[f"value_{turn}"] != after[f"value_{turn}"] and
                    before[f"head_{turn}"] != after[f"head_{turn}"]
                )
                for turn in range(self.config.maporl.round_num)
            },
            "critic_adapter_changed": {
                str(turn): before[f"value_{turn}"] != after[f"value_{turn}"]
                for turn in range(self.config.maporl.round_num)
            },
            "critic_head_changed": {
                str(turn): before[f"head_{turn}"] != after[f"head_{turn}"]
                for turn in range(self.config.maporl.round_num)
            },
            "policy_updates": sum(int(item["policy_trained"]) for item in statistics),
            "critic_updates": len(statistics),
            "critic_updates_by_turn": {str(turn): len(items) for turn, items in per_turn.items()},
            "all_gradients_finite": True,
            "base_frozen": all(
                not parameter.requires_grad for name, parameter in self.model.named_parameters()
                if ".lora_" not in name
            ),
            "turn0_policy_frozen": self.policy_key(0) is None,
            "turns": {
                str(turn): {
                    key: sum(item[key] for item in items) / len(items)
                    for key in items[0]
                }
                for turn, items in per_turn.items()
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        import torch

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        payload = destination / "runtime.pt"
        state = {
            "parameters": {name: parameter.detach().cpu() for name, parameter in self.parameters.items()},
            "optimizer": _cpu_tree(self.optimizer.state_dict()),
            "optimizer_steps": self.optimizer_steps,
            "accumulation_microsteps": self.accumulation_microsteps,
            "pending_gradients": {
                name: parameter.grad.detach().cpu()
                for name, parameter in self.parameters.items() if parameter.grad is not None
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
        }
        torch.save(state, payload)
        metadata = {
            "schema_version": 1, "agent": self.agent, "model_path": self.model_path,
            "config_sha256": self.config.sha256(), "optimizer_steps": self.optimizer_steps,
            "adapters": list(self.model.peft_config), "dropout": 0.0,
            "files": {"runtime.pt": _sha256(payload)},
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in self.parameters.items()},
            "parameter_dtypes": {name: str(parameter.dtype) for name, parameter in self.parameters.items()},
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def load_checkpoint(self, path: str | Path) -> None:
        import torch

        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata["agent"] != self.agent or metadata["model_path"] != self.model_path:
            raise ValueError("checkpoint agent or base model mismatch")
        if metadata["config_sha256"] != self.config.sha256():
            raise ValueError("checkpoint configuration hash mismatch")
        payload = source / "runtime.pt"
        if _sha256(payload) != metadata["files"]["runtime.pt"]:
            raise ValueError("checkpoint checksum mismatch")
        state = torch.load(payload, map_location="cpu", weights_only=True)
        if set(state["parameters"]) != set(self.parameters):
            raise ValueError("checkpoint adapter or value-head parameters mismatch")
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                saved = state["parameters"][name]
                if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint parameter shape or dtype mismatch: {name}")
                parameter.copy_(saved.to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_steps = int(state["optimizer_steps"])
        self.accumulation_microsteps = int(state.get("accumulation_microsteps", 0))
        torch.set_rng_state(state["torch_rng_state"])
        if self.device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for name, gradient in state.get("pending_gradients", {}).items():
            if name not in self.parameters or gradient.shape != self.parameters[name].shape:
                raise ValueError(f"checkpoint pending gradient mismatch: {name}")
            self.parameters[name].grad = gradient.to(self.device)

    def export_policy(self, path: str | Path, turn: int = 1) -> Path | None:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        key = self.policy_key(turn)
        if key is None:
            return None
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(self.model, adapter_name=key)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in state.items()},
            str(destination / "adapter_model.safetensors"), metadata={"format": "pt"},
        )
        adapter_config = copy.deepcopy(self.model.peft_config[key])
        adapter_config.inference_mode = True
        adapter_config.base_model_name_or_path = self.model_path
        adapter_config.save_pretrained(destination)
        return destination

    def close(self) -> None:
        import torch

        self.optimizer.zero_grad(set_to_none=True)
        del self.optimizer, self.parameters, self.value_heads, self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
