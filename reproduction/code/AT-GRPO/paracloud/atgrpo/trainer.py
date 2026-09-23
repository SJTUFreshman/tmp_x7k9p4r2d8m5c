"""One agent's LoRA update under the shared team advantage.

Each agent owns a separate LoRA over its own frozen base, so the three policies
cannot share a parameter tensor or a process group. What makes this multi-agent
rather than three independent GRPO runs is that every row handed to every agent
carries the advantage of the *joint* rollout it came from.

The reference policy for the KL anchor is the frozen base, obtained by disabling
the PEFT adapter rather than loading a second copy of the model (the trick from
``scripts/rl_train.py:1492``). It stays fixed for the whole run, so the KL keeps
bounding drift across iterations instead of chasing the previous checkpoint.
"""
from __future__ import annotations

import contextlib
import json
import math
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .encode import collate, encode_turn
from .loss import magrpo_token_loss, masked_token_logprobs


@dataclass
class TrainStats:
    agent: str
    iteration: int
    n_rows: int = 0
    n_tokens: int = 0
    n_skipped: int = 0
    loss: float = 0.0
    policy_loss: float = 0.0
    kl: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    ratio_mean: float = 1.0
    ratio_max: float = 1.0
    clip_frac: float = 0.0
    adv_abs_mean: float = 0.0
    seconds: float = 0.0
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def disable_dropout(model) -> int:
    """Zero every dropout probability in the model.

    Required for correctness, not just determinism. The importance ratio
    ``exp(log pi_theta - log pi_old)`` is only meaningful if both log-probs come
    from the same function. With dropout active each forward pass samples a
    different mask, so even at identical weights the ratio drifts off 1.0 and
    the clipped surrogate starts responding to noise instead of to policy
    change. (``mas_grpo_probe/10_train_agent_grpo_lora.py:618`` does the same
    thing for the same reason.)
    """
    import torch

    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    return changed


@contextlib.contextmanager
def adapter_disabled(model):
    """Expose the frozen base for reference log-probs, with no second model."""
    inner = getattr(model, "module", model)
    disable = getattr(inner, "disable_adapter", None)
    if disable is None:
        yield
        return
    was_training = inner.training
    with disable():
        inner.eval()
        try:
            yield
        finally:
            inner.train(was_training)


class AgentTrainer:
    """Holds one agent's model/optimizer across iterations.

    Kept resident on purpose: re-loading the 8B per iteration would cost more
    wall-clock in model loading than in gradient steps.
    """

    def __init__(
        self,
        agent: str,
        *,
        base_model: str,
        lora: Any,
        learning_rate: float,
        max_seq_length: int,
        per_device_batch_size: int,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        warmup_iters: int = 5,
        gradient_checkpointing: bool = True,
        enable_thinking: bool = False,
        device: str | None = None,
        init_adapter: str | None = None,
    ) -> None:
        import torch
        from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.agent = agent
        self.max_seq_length = max_seq_length
        self.per_device_batch_size = max(1, per_device_batch_size)
        self.max_grad_norm = max_grad_norm
        self.warmup_iters = max(0, warmup_iters)
        self.enable_thinking = enable_thinking
        self.base_lr = learning_rate

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, trust_remote_code=True
        )
        if init_adapter:
            model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        else:
            model = get_peft_model(
                model,
                PeftLoraConfig(
                    r=lora.r,
                    lora_alpha=lora.alpha,
                    lora_dropout=lora.dropout,
                    target_modules=list(lora.target_modules),
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        if gradient_checkpointing and self.device.startswith("cuda"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        self.model = model.to(self.device)
        self.model.train()
        # Must happen before any log-prob is computed; see disable_dropout.
        self.dropout_disabled = disable_dropout(self.model)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"{agent}: no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )
        self.optimizer_restored = False
        if init_adapter:
            optimizer_path = Path(init_adapter) / "optimizer.pt"
            if optimizer_path.is_file():
                state = torch.load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(state)
                self.optimizer_restored = True
            else:
                warnings.warn(
                    f"{agent}: {optimizer_path} is missing; adapter weights were "
                    "restored but Adam moments restart from zero",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # -- data ---------------------------------------------------------------
    def encode_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict], int]:
        encoded, skipped = [], 0
        for row in rows:
            item = encode_turn(
                row["prompt_messages"],
                row["response"],
                self.tokenizer,
                max_seq_length=self.max_seq_length,
                enable_thinking=self.enable_thinking,
            )
            if item is None:
                skipped += 1
                continue
            item["advantage"] = float(row.get("advantage", 0.0))
            encoded.append(item)
        return encoded, skipped

    def _lr_for(self, iteration: int) -> float:
        if self.warmup_iters and iteration < self.warmup_iters:
            return self.base_lr * (iteration + 1) / self.warmup_iters
        return self.base_lr

    # -- update -------------------------------------------------------------
    def step(
        self,
        rows: list[dict[str, Any]],
        *,
        iteration: int,
        clip_epsilon: float,
        kl_coef: float,
        loss_agg: str = "seq_mean",
        inner_epochs: int = 1,
        max_encode_skip_frac: float = 0.10,
        ratio_tolerance: float = 1e-3,
    ) -> TrainStats:
        import time

        import torch

        started = time.time()
        stats = TrainStats(agent=self.agent, iteration=iteration)
        if not rows:
            # Sequential mode can leave an agent with nothing to learn from.
            # Skip the optimizer AND the schedule; do not fabricate a step.
            stats.skipped_reason = "no rows"
            stats.seconds = time.time() - started
            return stats

        encoded, skipped = self.encode_rows(rows)
        stats.n_skipped = skipped
        # Checked before the emptiness test on purpose: a 100% encode failure is
        # strictly worse than the threshold, so it must raise rather than look
        # like a benign skip.
        skip_frac = skipped / max(1, len(rows))
        if skip_frac > max_encode_skip_frac:
            raise RuntimeError(
                f"{self.agent}: {skip_frac:.1%} of rows failed to encode "
                f"(limit {max_encode_skip_frac:.1%}); training on the remainder "
                "would silently bias the update"
            )
        if not encoded:
            stats.skipped_reason = "all rows failed to encode"
            stats.seconds = time.time() - started
            return stats

        lr = self._lr_for(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        stats.lr = lr

        pad_id = self.tokenizer.pad_token_id or 0
        batches = [
            encoded[i : i + self.per_device_batch_size]
            for i in range(0, len(encoded), self.per_device_batch_size)
        ]

        # pi_old is snapshotted once, before any update. With inner_epochs == 1
        # it equals pi_theta and the ratio is identically 1; with more epochs the
        # later passes are genuinely off-policy and the clip starts working.
        old_logps: list[Any] = []
        for batch in batches:
            payload = collate(batch, pad_id)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=payload["input_ids"].to(self.device),
                    attention_mask=payload["attention_mask"].to(self.device),
                )
                logp, _ = masked_token_logprobs(
                    outputs.logits, payload["labels"].to(self.device)
                )
            old_logps.append(logp.detach())

        ref_logps: list[Any] = []
        if kl_coef > 0:
            with adapter_disabled(self.model):
                for batch in batches:
                    payload = collate(batch, pad_id)
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=payload["input_ids"].to(self.device),
                            attention_mask=payload["attention_mask"].to(self.device),
                        )
                        logp, _ = masked_token_logprobs(
                            outputs.logits, payload["labels"].to(self.device)
                        )
                    ref_logps.append(logp.detach())

        totals = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "tokens": 0}
        diag_accum = {"ratio_mean": 0.0, "ratio_max": 1.0, "clip_frac": 0.0, "adv": 0.0}
        n_micro = 0

        for epoch in range(inner_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(batches):
                payload = collate(batch, pad_id)
                outputs = self.model(
                    input_ids=payload["input_ids"].to(self.device),
                    attention_mask=payload["attention_mask"].to(self.device),
                )
                token_logp, mask = masked_token_logprobs(
                    outputs.logits, payload["labels"].to(self.device)
                )
                loss, policy_loss, kl, diag = magrpo_token_loss(
                    token_logp,
                    old_logps[index],
                    ref_logps[index] if ref_logps else None,
                    mask,
                    payload["advantages"].to(self.device),
                    clip_epsilon=clip_epsilon,
                    kl_coef=kl_coef,
                    agg=loss_agg,
                )
                # One optimizer step per iteration: accumulate over all batches.
                (loss / len(batches)).backward()

                totals["loss"] += float(loss)
                totals["policy"] += float(policy_loss)
                totals["kl"] += float(kl)
                totals["tokens"] += diag["n_tokens"]
                diag_accum["ratio_mean"] += diag["ratio_mean"]
                diag_accum["ratio_max"] = max(diag_accum["ratio_max"], diag["ratio_max"])
                diag_accum["clip_frac"] += diag["clip_frac"]
                diag_accum["adv"] += diag["adv_abs_mean"]
                n_micro += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            stats.grad_norm = float(grad_norm)

        stats.n_rows = len(encoded)
        stats.n_tokens = totals["tokens"]
        stats.loss = totals["loss"] / max(1, n_micro)
        stats.policy_loss = totals["policy"] / max(1, n_micro)
        stats.kl = totals["kl"] / max(1, n_micro)
        stats.ratio_mean = diag_accum["ratio_mean"] / max(1, n_micro)
        stats.ratio_max = diag_accum["ratio_max"]
        stats.clip_frac = diag_accum["clip_frac"] / max(1, n_micro)
        stats.adv_abs_mean = diag_accum["adv"] / max(1, n_micro)
        stats.seconds = time.time() - started

        # The canary: on-policy means ratio == 1. Any real deviation says the
        # rollout prompt and the training prompt disagree (chat template,
        # tokenizer, or a stale adapter), which would corrupt every update.
        if inner_epochs == 1 and not math.isclose(
            stats.ratio_mean, 1.0, abs_tol=max(ratio_tolerance, 1e-4)
        ):
            raise RuntimeError(
                f"{self.agent}: on-policy ratio_mean={stats.ratio_mean:.6f} "
                f"deviates from 1.0 by more than {ratio_tolerance}; the rollout "
                "and training prompts do not match"
            )
        return stats

    # -- publishing ---------------------------------------------------------
    def save_adapter(self, destination: Path) -> Path:
        """Save adapter and optimizer atomically for exact resume."""
        import torch

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(destination.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        self.model.save_pretrained(str(staging))
        torch.save(self.optimizer.state_dict(), staging / "optimizer.pt")
        shutil.rmtree(destination, ignore_errors=True)
        staging.replace(destination)
        return destination
