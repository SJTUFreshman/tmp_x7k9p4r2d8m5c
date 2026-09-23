"""PPO math, ported from the official multi-agent trainer.

Provenance: /tmp/maporl_ref/trl/trl/trainer/ppov2_trainer_multi_different_model.py
  KL + reward placement : :1630-1646
  GAE                   : :1648-1691
  advantage whitening   : :1692-1693
  clipped losses        : :1833-1866

Everything here is a pure function of tensors so the whole file is testable on
CPU against closed-form references.

The reward shape is worth stating explicitly, because it is unusual: the
per-(turn, agent) scalar from ``reward_rules`` is placed on **one token** -- the
last real token of the response -- while the KL penalty is spread densely over
every token. GAE then propagates the terminal scalar backwards. So the scalar
reward and the KL act on completely different timescales.
"""
from __future__ import annotations

from typing import Any


def masked_mean(values: Any, mask: Any, axis: int | None = None) -> Any:
    """Mean over ``mask``. TRL semantics."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis).clamp_min(1e-8)
    return (values * mask).sum() / mask.sum().clamp_min(1e-8)


def masked_var(values: Any, mask: Any, unbiased: bool = True) -> Any:
    import torch

    mean = masked_mean(values, mask)
    centered = (values - mean) ** 2
    variance = masked_mean(centered, mask)
    if unbiased:
        count = mask.sum()
        # Bessel correction, guarded for the degenerate 1-element case.
        correction = count / (count - 1).clamp_min(1.0)
        variance = variance * correction
    return variance


def masked_whiten(values: Any, mask: Any, shift_mean: bool = True) -> Any:
    """Standardize over the masked positions. TRL's implementation."""
    import torch

    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def place_terminal_reward(
    kl: Any,
    scores: Any,
    sequence_lengths: Any,
    sequence_lengths_p1: Any,
    *,
    kl_coef: float,
) -> tuple[Any, Any]:
    """Dense ``-kl_coef * KL`` everywhere, plus the scalar on one token.

    Port of :1630-1641. Returns ``(rewards, non_score_reward)``.

    The index is ``sequence_lengths_p1`` when that is inside the tensor and
    ``sequence_lengths`` otherwise, then clamped -- i.e. the token just past the
    response when there is room for it, else the last one.
    """
    import torch

    non_score_reward = -kl_coef * kl
    rewards = non_score_reward.clone()
    start = torch.arange(rewards.size(0), device=rewards.device)
    end = torch.where(
        sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths
    )
    end = torch.clamp(end, 0, rewards.size(1) - 1)
    rewards[start, end] += scores
    return rewards, non_score_reward


def gae(rewards: Any, values: Any, *, gamma: float = 1.0, lam: float = 0.95) -> tuple[Any, Any]:
    """Generalized advantage estimation. Port of :1648-1691.

    ``nextvalues`` is zero at the final position -- the official code does not
    bootstrap past the end of the response. Returns ``(advantages, returns)``.
    """
    import torch

    gen_length = rewards.shape[1]
    lastgaelam = 0.0
    reversed_advantages = []
    for t in reversed(range(gen_length)):
        nextvalues = (
            values[:, t + 1]
            if t < gen_length - 1
            else torch.zeros_like(values[:, 0])
        )
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        reversed_advantages.append(lastgaelam)
    advantages = torch.stack(reversed_advantages[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def prepare_advantages(
    kl: Any,
    values: Any,
    scores: Any,
    sequence_lengths: Any,
    sequence_lengths_p1: Any,
    padding_mask: Any,
    padding_mask_p1: Any,
    *,
    kl_coef: float,
    gamma: float = 1.0,
    lam: float = 0.95,
    whiten_rewards: bool = False,
) -> dict[str, Any]:
    """The full :1630-1693 sequence, in order.

    ``whiten_rewards`` is False in the official config; advantage whitening is
    unconditional.
    """
    import torch

    rewards, non_score_reward = place_terminal_reward(
        kl, scores, sequence_lengths, sequence_lengths_p1, kl_coef=kl_coef
    )
    if whiten_rewards:
        rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
        rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

    advantages, returns = gae(rewards, values, gamma=gamma, lam=lam)
    # :1692-1693 -- whiten, then zero the padding.
    advantages = masked_whiten(advantages, ~padding_mask)
    advantages = torch.masked_fill(advantages, padding_mask, 0)
    return {
        "advantages": advantages,
        "returns": returns,
        "rewards": rewards,
        "non_score_reward": non_score_reward,
    }


def ppo_losses(
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    vpred: Any,
    mb_values: Any,
    returns: Any,
    padding_mask: Any,
    padding_mask_p1: Any,
    *,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2,
    vf_coef: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Clipped surrogate + clipped value loss. Port of :1833-1866."""
    import torch

    vpredclipped = torch.clamp(
        vpred, mb_values - cliprange_value, mb_values + cliprange_value
    )
    vf_losses1 = torch.square(vpred - returns)
    vf_losses2 = torch.square(vpredclipped - returns)
    vf_loss_max = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1)

    logprobs_diff = new_logprobs - old_logprobs
    ratio = torch.exp(logprobs_diff)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss_max = torch.max(pg_losses1, pg_losses2)
    pg_loss = masked_mean(pg_loss_max, ~padding_mask)

    loss = pg_loss + vf_coef * vf_loss

    with torch.no_grad():
        mask = (~padding_mask).float()
        stats = {
            "pg_loss": float(pg_loss),
            "vf_loss": float(vf_loss),
            "loss": float(loss),
            "ratio_mean": float(masked_mean(ratio, ~padding_mask)),
            "ratio_max": float((ratio * mask).max()) if float(mask.sum()) else 1.0,
            # :1860-1861 -- the standard k1/2 approximation TRL logs.
            "approxkl": float(0.5 * masked_mean(logprobs_diff**2, ~padding_mask)),
            "pg_clipfrac": float(
                masked_mean((pg_losses2 > pg_losses1).float(), ~padding_mask)
            ),
            "vf_clipfrac": float(
                masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1)
            ),
            "n_tokens": int(mask.sum()),
        }
    return loss, stats


def entropy_from_logits(logits: Any) -> Any:
    """:1864-1866. Mean token entropy, for monitoring collapse."""
    import torch

    prob_dist = torch.nn.functional.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
