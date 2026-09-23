"""Turn a stored turn into ``input_ids``/``labels`` for the trainer.

The approach follows ``scripts/rl_train.py:506 encode_one``: render the full
conversation, render the prompt separately, and locate the final assistant span
by *character offsets* rather than assuming the prompt's token ids are a prefix
of the full conversation's.

That assumption is what ``mas_grpo_probe/10_train_agent_grpo_lora.py:187``
makes, and it fails for Qwen3 whenever earlier assistant turns exist -- the
template changes the assistant scaffold, the prefix check fails, and the row is
silently skipped. In a multi-turn MAS that would drop most of the data.
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

    full_messages = list(messages) + [{"role": "assistant", "content": response}]
    full_text = apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prompt_text = apply_chat_template(
        tokenizer, messages, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Locate the response inside the rendered conversation by character index.
    # rfind, not find: the same string may appear in the history.
    char_start = full_text.rfind(response)
    if char_start < 0:
        return None
    char_end = char_start + len(response)

    # Tokenize WITHOUT truncation first. Right-truncation would cut the tail of
    # the sequence -- which is exactly the response we need to train on -- and
    # the row would then be silently dropped. Over-long inputs are instead
    # trimmed from the FRONT of the prompt below, preserving the response.
    encoding = tokenizer(
        full_text, return_offsets_mapping=True, add_special_tokens=False
    )
    input_ids = list(encoding["input_ids"])
    offsets = list(encoding["offset_mapping"])
    if not input_ids:
        return None

    labels = [-100] * len(input_ids)
    n_response = 0
    for index, (start, end) in enumerate(offsets):
        if start == end:  # special tokens carry an empty span
            continue
        if start >= char_start and end <= char_end:
            labels[index] = input_ids[index]
            n_response += 1

    if len(input_ids) > max_seq_length:
        first_response = next(
            (i for i, label in enumerate(labels) if label != -100), None
        )
        if first_response is not None:
            n_response_tokens = len(input_ids) - first_response
            if n_response_tokens >= max_seq_length:
                # The response alone exceeds the window: nothing useful to learn
                # from a prompt-free fragment.
                return None
            cut = len(input_ids) - max_seq_length
            input_ids = input_ids[cut:]
            labels = labels[cut:]
        else:
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]
        n_response = sum(1 for label in labels if label != -100)

    if n_response == 0:
        # Offsets unusable (rare tokenizers): fall back to "everything after the
        # prompt's token length", which holds whenever the template did not
        # rewrite the prefix. Re-derive from the untruncated ids so the index
        # still lines up after any front-trimming above.
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        cut = min(len(prompt_ids), len(input_ids))
        for index in range(cut, len(input_ids)):
            labels[index] = input_ids[index]
            n_response += 1
        if n_response == 0:
            return None
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]
            n_response = sum(1 for label in labels if label != -100)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "n_response_tokens": n_response,
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
