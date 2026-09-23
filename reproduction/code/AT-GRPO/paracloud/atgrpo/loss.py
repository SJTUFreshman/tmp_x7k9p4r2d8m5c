"""MAGRPO token-level objective.

Lifted from ``mas_grpo_probe/10_train_agent_grpo_lora.py:370 grpo_token_loss`` with
two corrections that matter once the loop is online rather than a single offline
pass:

1. **pi_old and pi_ref are separate tensors.** The original fills
   ``old_token_logprobs`` from the reference model and then uses that one tensor as
   both the importance-ratio denominator and the KL anchor. That is self-consistent
   only while pi_behavior == pi_ref, i.e. for one offline pass. Here the policy
   drifts across iterations: the ratio must be anchored to the *rollout-time*
   policy, while the KL must stay anchored to the *frozen base*, or the
   regularizer silently stops regularizing.

2. **Aggregation is selectable.** ``seq_mean`` reproduces the lifted behaviour
   (per-sequence token mean, then batch mean). ``token_mean`` divides by the global
   token count instead, removing the length bias that makes short responses carry
   more per-token weight (Dr.GRPO / DAPO style).

With ``inner_epochs == 1`` the rollout policy *is* the current policy, so the ratio
is identically 1 and the clip is inert -- the surrogate degenerates to the plain
on-policy policy gradient. We still compute the ratio, because
``diagnostics["ratio_mean"]`` deviating from 1.0 is the cheapest available detector
for a chat-template or tokenizer mismatch between rollout and training.
"""
from __future__ import annotations

from typing import Any, Literal

Aggregation = Literal["seq_mean", "token_mean"]


def masked_token_logprobs(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Return ``(token_logp, mask)`` aligned to next-token prediction.

    ``labels`` uses -100 for positions that are not response tokens.
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = (shift_labels != -100).to(shift_logits.dtype)
    safe_labels = shift_labels.clamp_min(0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logp * mask, mask


def _aggregate(values: Any, mask: Any, agg: Aggregation) -> Any:
    """Reduce ``[B, L]`` per-token values to a scalar under the chosen convention."""
    if agg == "token_mean":
        total = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / total
    counts = mask.sum(dim=1).clamp_min(1.0)
    return ((values * mask).sum(dim=1) / counts).mean()


def magrpo_token_loss(
    token_logp: Any,
    old_token_logp: Any,
    ref_token_logp: Any | None,
    token_mask: Any,
    advantages: Any,
    *,
    clip_epsilon: float,
    kl_coef: float,
    agg: Aggregation = "seq_mean",
) -> tuple[Any, Any, Any, dict[str, float]]:
    """Clipped group-relative surrogate plus a k3 KL anchor to the frozen base.

    Shapes: ``token_logp``/``old_token_logp``/``ref_token_logp``/``token_mask`` are
    ``[B, L-1]``; ``advantages`` is ``[B]``. Returns
    ``(loss, policy_loss, kl, diagnostics)``.
    """
    import torch

    mask = token_mask.to(token_logp.dtype)

    # --- policy term: ratio against the rollout-time policy -------------------
    log_ratio = (token_logp - old_token_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    adv = advantages.to(token_logp.dtype).unsqueeze(1)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    policy_loss = -_aggregate(surrogate, mask, agg)

    # --- KL term: k3 estimator against the frozen base ------------------------
    if ref_token_logp is None or kl_coef <= 0.0:
        kl = torch.zeros((), device=token_logp.device, dtype=token_logp.dtype)
    else:
        # k3 = exp(d) - d - 1 with d = log pi_ref - log pi_theta, an unbiased and
        # non-negative estimator of KL(pi_theta || pi_ref).
        ref_log_ratio = (ref_token_logp - token_logp).clamp(-20.0, 20.0)
        k3 = ref_log_ratio.exp() - ref_log_ratio - 1.0
        kl = _aggregate(k3, mask, agg)

    loss = policy_loss + kl_coef * kl

    with torch.no_grad():
        n_tokens = mask.sum()
        denom = n_tokens.clamp_min(1.0)
        clipped_frac = (
            ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).to(mask.dtype)
            * mask
        ).sum() / denom
        diagnostics = {
            "ratio_mean": float((ratio * mask).sum() / denom),
            "ratio_max": float((ratio * mask).max()) if float(n_tokens) > 0 else 1.0,
            "clip_frac": float(clipped_frac),
            "adv_abs_mean": float(advantages.abs().mean()),
            "n_tokens": int(n_tokens),
            "n_rows": int(advantages.shape[0]),
        }
    return loss, policy_loss, kl, diagnostics
