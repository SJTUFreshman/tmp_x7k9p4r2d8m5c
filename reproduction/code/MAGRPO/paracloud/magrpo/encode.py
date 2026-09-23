"""Turn a stored turn into ``input_ids``/``labels`` for the trainer.

Render the generation prompt exactly as the rollout server does, then append the
response tokens. Rendering the response as another assistant message can rewrite
its prefix or strip whitespace in Qwen3's template, changing the context under
which the response was generated.

Inputs above ``max_seq_length`` retain the existing front-trimming behavior.
Those rows use a shortened conditioning context; configuring sufficient training
sequence length is required when exact rollout-context fidelity matters.
"""
from __future__ import annotations

from typing import Any


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Tokenizers whose template does not accept enable_thinking.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def encode_turn(
    messages: list[dict[str, str]],
    response: str,
    tokenizer,
    *,
    max_seq_length: int,
    enable_thinking: bool = False,
) -> dict[str, Any] | None:
    """Return ``{input_ids, attention_mask, labels}``, or None if unusable.

    ``labels`` is -100 everywhere except the assistant response tokens, so the
    loss only ever sees tokens the policy actually generated.
    """
    if not isinstance(response, str) or not response.strip():
        return None
    if not messages:
        return None

    prompt_text = apply_chat_template(
        tokenizer, messages, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    response_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])
    if not prompt_ids or not response_ids or len(response_ids) >= max_seq_length:
        return None
    prompt_ids = prompt_ids[-(max_seq_length - len(response_ids)):]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "n_response_tokens": len(response_ids),
    }


def collate(rows: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad a batch. ``labels`` pads with -100 so padding never scores."""
    import torch

    width = max(len(r["input_ids"]) for r in rows)
    input_ids, attention, labels, advantages = [], [], [], []
    for row in rows:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        attention.append(row["attention_mask"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
        advantages.append(float(row.get("advantage", 0.0)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "advantages": torch.tensor(advantages, dtype=torch.float),
    }
