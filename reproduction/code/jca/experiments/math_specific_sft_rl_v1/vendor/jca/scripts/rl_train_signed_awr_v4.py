#!/usr/bin/env python3
"""Protocol-safe conservative signed-AWR training for MATH collaboration."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DECISION_FIELDS = (
    "tentative_answer",
    "action",
    "handoff_target",
    "confirmed_answer",
)
PROTOCOL_FIELDS = (
    "reasoning",
    "tentative_answer",
    "action",
    "handoff_target",
    "handoff_note",
    "confirmed_answer",
)
PROTOCOL_FREE_CONTENT_FIELDS = (
    "reasoning",
    "tentative_answer",
    "handoff_note",
    "confirmed_answer",
)
ALGORITHM_NAME = "conservative_signed_awr_v4_protocol_floor_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one GSM agent with conservative signed AWR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", required=True, choices=["A1", "A2", "A3"])
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--validation-rollout", type=Path)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--advantage-field", default="train_weight")
    parser.add_argument("--mask-field", default="loss_mask_mode_v4")
    parser.add_argument(
        "--negative-mask-policy",
        choices=("role_aware", "uniform_full"),
        default="role_aware",
    )
    parser.add_argument(
        "--dangerous-decision-coef",
        type=float,
        default=0.0,
        help="Extra reference-margin loss on marked dangerous decision fields.",
    )
    parser.add_argument(
        "--positive-stop-decision-coef",
        type=float,
        default=0.0,
        help="Extra positive action loss on marked correct stop decisions.",
    )
    parser.add_argument("--reference-coef", type=float, default=0.05)
    parser.add_argument(
        "--protocol-reference-coef",
        type=float,
        default=1.0,
        help="Unweighted one-sided SFT-reference floor on protocol tokens.",
    )
    parser.add_argument(
        "--protocol-reference-margin",
        type=float,
        default=0.05,
        help="Allowed protocol-token log-prob drop below the SFT reference.",
    )
    parser.add_argument(
        "--negative-margin",
        type=float,
        default=0.5,
        help="Maximum mean log-prob suppression below the frozen SFT reference.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def setup_logging(out_dir: Path, agent: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"jca.rl.signed_awr_v4.{agent}")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            out_dir / f"rl_signed_awr_v4_{agent}.log",
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


@dataclass
class AWRRecord:
    problem_id: str
    turn: int
    agent_id: str
    messages: List[Dict[str, str]]
    response: str
    advantage: float
    mask_mode: str
    sample_source: str
    reward_class: str
    decision_fields: Tuple[str, ...]
    dangerous_decision_fields: Tuple[str, ...]
    positive_decision_fields: Tuple[str, ...]


def load_records(
    path: Path,
    agent: str,
    advantage_field: str,
    mask_field: str,
    negative_mask_policy: str = "role_aware",
) -> List[AWRRecord]:
    if negative_mask_policy not in {"role_aware", "uniform_full"}:
        raise ValueError(f"unknown negative mask policy: {negative_mask_policy}")
    records: List[AWRRecord] = []
    skipped = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] += 1
                continue
            if row.get("agent_id") != agent:
                continue
            if not row.get("messages") or not row.get("response"):
                skipped["missing_training_text"] += 1
                continue
            if advantage_field not in row:
                raise ValueError(
                    f"missing {advantage_field!r} at {path}:{line_number}"
                )
            try:
                advantage = float(row[advantage_field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {advantage_field!r} at {path}:{line_number}"
                ) from exc
            if not math.isfinite(advantage) or advantage == 0:
                raise ValueError(
                    f"signed AWR requires finite nonzero advantage at "
                    f"{path}:{line_number}: {advantage}"
                )
            mask_mode = str(row.get(mask_field) or "")
            if mask_mode not in {"full", "decision"}:
                raise ValueError(
                    f"invalid {mask_field!r} at {path}:{line_number}: {mask_mode}"
                )
            if advantage > 0 and mask_mode != "full":
                raise ValueError(
                    f"positive record must use full mask at {path}:{line_number}"
                )
            if advantage < 0:
                expected_negative_mask = (
                    "full" if negative_mask_policy == "uniform_full" else "decision"
                )
                if mask_mode != expected_negative_mask:
                    raise ValueError(
                        f"negative {agent} record must use {expected_negative_mask} "
                        f"mask at {path}:{line_number}"
                    )
            raw_decision_fields = row.get("decision_fields_v4", [])
            if not isinstance(raw_decision_fields, list) or not all(
                isinstance(field, str) and field in DECISION_FIELDS
                for field in raw_decision_fields
            ):
                raise ValueError(
                    f"invalid decision_fields_v4 at {path}:{line_number}"
                )
            if mask_mode == "decision" and not raw_decision_fields:
                raise ValueError(
                    f"decision-masked record has no fields at {path}:{line_number}"
                )
            raw_dangerous_fields = row.get(
                "dangerous_decision_fields_v4_3",
                row.get("dangerous_decision_fields_v4_2", []),
            )
            if not isinstance(raw_dangerous_fields, list) or not all(
                isinstance(field, str) and field in DECISION_FIELDS
                for field in raw_dangerous_fields
            ):
                raise ValueError(
                    f"invalid dangerous decision fields at "
                    f"{path}:{line_number}"
                )
            if raw_dangerous_fields and advantage >= 0:
                raise ValueError(
                    f"dangerous decision fields require a negative record at "
                    f"{path}:{line_number}"
                )
            raw_positive_fields = row.get("positive_decision_fields_v4_3", [])
            if not isinstance(raw_positive_fields, list) or not all(
                isinstance(field, str) and field in DECISION_FIELDS
                for field in raw_positive_fields
            ):
                raise ValueError(
                    f"invalid positive_decision_fields_v4_3 at "
                    f"{path}:{line_number}"
                )
            if raw_positive_fields and advantage <= 0:
                raise ValueError(
                    f"positive decision fields require a positive record at "
                    f"{path}:{line_number}"
                )
            records.append(AWRRecord(
                problem_id=str(row.get("problem_id") or ""),
                turn=int(row.get("turn", 0)),
                agent_id=agent,
                messages=row["messages"],
                response=str(row["response"]),
                advantage=advantage,
                mask_mode=mask_mode,
                sample_source=str(row.get("sample_source_v4") or "unknown"),
                reward_class=str(row.get("reward_class_v4") or "unknown"),
                decision_fields=tuple(raw_decision_fields),
                dangerous_decision_fields=tuple(raw_dangerous_fields),
                positive_decision_fields=tuple(raw_positive_fields),
            ))
    if skipped:
        print(f"[load_records] skipped={dict(skipped)}")
    return records


def json_field_value_spans(response: str) -> Dict[str, Tuple[int, int]]:
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    decoder = json.JSONDecoder()
    length = len(response)

    def skip_whitespace(index: int) -> int:
        while index < length and response[index].isspace():
            index += 1
        return index

    index = skip_whitespace(0)
    if index >= length or response[index] != "{":
        raise ValueError("response JSON does not start with an object")
    index = skip_whitespace(index + 1)
    spans: Dict[str, Tuple[int, int]] = {}
    if index < length and response[index] == "}":
        index = skip_whitespace(index + 1)
    else:
        while True:
            key, key_end = decoder.raw_decode(response, index)
            if not isinstance(key, str):
                raise ValueError("response JSON object has a non-string key")
            if key in spans:
                raise ValueError(f"response JSON repeats field {key!r}")
            index = skip_whitespace(key_end)
            if index >= length or response[index] != ":":
                raise ValueError(f"response JSON field {key!r} has no colon")
            value_start = skip_whitespace(index + 1)
            _value, value_end = decoder.raw_decode(response, value_start)
            spans[key] = (value_start, value_end)
            index = skip_whitespace(value_end)
            if index >= length:
                raise ValueError("response JSON object is not closed")
            if response[index] == "}":
                index = skip_whitespace(index + 1)
                break
            if response[index] != ",":
                raise ValueError(f"response JSON field {key!r} has no separator")
            index = skip_whitespace(index + 1)
    if index != length:
        raise ValueError("response JSON has trailing content")
    if set(spans) != set(parsed):
        raise ValueError("response JSON lexical fields do not match parsed fields")
    return spans


def decision_spans(
    response: str,
    fields: Sequence[str] = DECISION_FIELDS,
) -> List[Tuple[int, int]]:
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    value_spans = json_field_value_spans(response)
    spans: List[Tuple[int, int]] = []
    for field in fields:
        if field not in parsed:
            raise ValueError(f"response is missing decision field {field!r}")
        value_start, value_end = value_spans[field]
        value = parsed[field]
        if isinstance(value, str) and value_end - value_start >= 2:
            spans.append((value_start + 1, value_end - 1))
        else:
            spans.append((value_start, value_end))
    return spans


def protocol_spans(response: str) -> List[Tuple[int, int]]:
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    if set(parsed) != set(PROTOCOL_FIELDS):
        raise ValueError("response JSON fields do not match the protocol schema")
    value_spans = json_field_value_spans(response)
    free_spans = []
    for field in PROTOCOL_FREE_CONTENT_FIELDS:
        value_start, value_end = value_spans[field]
        if isinstance(parsed[field], str) and value_end - value_start >= 2:
            free_spans.append((value_start + 1, value_end - 1))
    protected_spans: List[Tuple[int, int]] = []
    cursor = 0
    for free_start, free_end in sorted(free_spans):
        if cursor < free_start:
            protected_spans.append((cursor, free_start))
        cursor = max(cursor, free_end)
    if cursor < len(response):
        protected_spans.append((cursor, len(response)))
    return protected_spans


def encode_one(
    record: AWRRecord,
    tokenizer,
    max_seq_length: int,
) -> Optional[Dict[str, Any]]:
    full_text = tokenizer.apply_chat_template(
        [*record.messages, {"role": "assistant", "content": record.response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        record.messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    full_ids = full_encoding["input_ids"]
    offsets = full_encoding.get("offset_mapping")
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=False,
    )["input_ids"]
    if len(full_ids) >= max_seq_length or len(prompt_ids) >= len(full_ids):
        return None
    completion_labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = list(completion_labels)
    if offsets is None:
        raise ValueError("fast tokenizer offset mapping is required")
    response_start = full_text.rfind(record.response)
    if response_start < 0:
        raise ValueError("assistant response is not present in chat template")
    response_end = response_start + len(record.response)
    protocol_global_spans = [
        (response_start + start, response_start + end)
        for start, end in protocol_spans(record.response)
    ]
    protocol_labels = [-100] * len(full_ids)
    for index, (token_start, token_end) in enumerate(offsets):
        if index < len(prompt_ids):
            continue
        outside_response = (
            token_end <= token_start
            or token_start < response_start
            or token_end > response_end
        )
        overlaps_protocol = any(
            token_end > span_start and token_start < span_end
            for span_start, span_end in protocol_global_spans
        )
        if outside_response or overlaps_protocol:
            protocol_labels[index] = full_ids[index]
    if all(label == -100 for label in protocol_labels):
        raise ValueError("protocol mask selected no completion tokens")

    def labels_for_fields(fields: Sequence[str]) -> List[int]:
        global_spans = [
            (response_start + start, response_start + end)
            for start, end in decision_spans(
                record.response,
                fields,
            )
        ]
        decision_labels = [-100] * len(full_ids)
        for index, (token_start, token_end) in enumerate(offsets):
            if index < len(prompt_ids) or token_end <= token_start:
                continue
            if any(
                token_end > span_start and token_start < span_end
                for span_start, span_end in global_spans
            ):
                decision_labels[index] = full_ids[index]
        if all(label == -100 for label in decision_labels):
            raise ValueError("decision mask selected no response tokens")
        return decision_labels

    if record.mask_mode == "decision":
        labels = labels_for_fields(record.decision_fields)
    if record.advantage < 0:
        labels = [
            label if protocol_label == -100 else -100
            for label, protocol_label in zip(labels, protocol_labels)
        ]
        if all(label == -100 for label in labels):
            raise ValueError("negative semantic mask selected no unprotected tokens")
    dangerous_labels = (
        labels_for_fields(record.dangerous_decision_fields)
        if record.dangerous_decision_fields
        else [-100] * len(full_ids)
    )
    positive_labels = (
        labels_for_fields(record.positive_decision_fields)
        if record.positive_decision_fields
        else [-100] * len(full_ids)
    )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "dangerous_labels": dangerous_labels,
        "positive_labels": positive_labels,
        "protocol_labels": protocol_labels,
        "completion_labels": completion_labels,
    }


def collate_batch(
    records: Sequence[AWRRecord],
    tokenizer,
    max_seq_length: int,
) -> Optional[Tuple[Any, Any, Any, Any, Any, Any, Any, Any]]:
    import torch

    encoded_records: List[Dict[str, Any]] = []
    advantages: List[float] = []
    for record in records:
        encoded = encode_one(record, tokenizer, max_seq_length)
        if encoded is None:
            continue
        encoded_records.append(encoded)
        advantages.append(record.advantage)
    if not encoded_records:
        return None
    maximum_length = max(len(encoded["input_ids"]) for encoded in encoded_records)
    pad_token_id = tokenizer.pad_token_id
    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []
    labels: List[List[int]] = []
    dangerous_labels: List[List[int]] = []
    positive_labels: List[List[int]] = []
    protocol_labels: List[List[int]] = []
    completion_labels: List[List[int]] = []
    for encoded in encoded_records:
        padding = maximum_length - len(encoded["input_ids"])
        input_ids.append(encoded["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(encoded["attention_mask"] + [0] * padding)
        labels.append(encoded["labels"] + [-100] * padding)
        dangerous_labels.append(encoded["dangerous_labels"] + [-100] * padding)
        positive_labels.append(encoded["positive_labels"] + [-100] * padding)
        protocol_labels.append(encoded["protocol_labels"] + [-100] * padding)
        completion_labels.append(encoded["completion_labels"] + [-100] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(dangerous_labels, dtype=torch.long),
        torch.tensor(positive_labels, dtype=torch.long),
        torch.tensor(protocol_labels, dtype=torch.long),
        torch.tensor(completion_labels, dtype=torch.long),
        torch.tensor(advantages, dtype=torch.float32),
    )


def sequence_log_probs_from_logits(logits, labels):
    import torch.nn.functional as functional

    shifted_labels = labels[:, 1:]
    shifted_mask = shifted_labels != -100
    log_probs = functional.log_softmax(logits, dim=-1)
    gathered_token_log_probs = log_probs.gather(
        2,
        shifted_labels.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = gathered_token_log_probs * shifted_mask.float()
    sequence_lengths = shifted_mask.sum(dim=1).clamp(min=1).float()
    sequence_values = token_log_probs.sum(dim=1) / sequence_lengths
    return (
        sequence_values,
        gathered_token_log_probs,
        shifted_mask,
        sequence_lengths,
    )


def sequence_log_probs(model, input_ids, attention_mask, labels):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :]
    return sequence_log_probs_from_logits(logits, labels)


def protocol_reference_statistics_from_logits(
    policy_logits,
    reference_logits,
    protocol_labels,
    margin: float,
):
    import torch
    import torch.nn.functional as functional

    if margin < 0:
        raise ValueError("protocol reference margin must be non-negative")
    _policy_sequence, policy_token_logp, protocol_mask, _lengths = (
        sequence_log_probs_from_logits(policy_logits, protocol_labels)
    )
    _reference_sequence, reference_token_logp, _mask, _lengths = (
        sequence_log_probs_from_logits(reference_logits, protocol_labels)
    )
    active = protocol_mask.float()
    protected_token_count = active.sum()
    denominator = protected_token_count.clamp(min=1.0)
    downward_drift = functional.relu(
        reference_token_logp.float() - policy_token_logp.float()
    ) * active
    floor_excess = functional.relu(downward_drift - margin) * active
    downward_square_sum = downward_drift.pow(2).sum()
    floor_sum = floor_excess.pow(2).sum()
    violation_count = ((floor_excess > 0) & protocol_mask).float().sum()
    return {
        "floor_loss": floor_sum / denominator,
        "floor_sum": floor_sum,
        "protected_token_count": protected_token_count,
        "downward_sum": downward_drift.sum(),
        "downward_square_sum": downward_square_sum,
        "downward_mean": downward_drift.sum() / denominator,
        "downward_rms": torch.sqrt(downward_square_sum / denominator),
        "downward_max": downward_drift.max(),
        "violation_count": violation_count,
        "violation_fraction": violation_count / denominator,
    }


def compute_signed_awr_loss(
    model,
    reference_model,
    batch_input_ids,
    batch_attention_mask,
    batch_labels,
    advantages,
    *,
    reference_coef: float,
    negative_margin: float = 0.5,
    negative_objective: str = "reference_margin",
    dangerous_labels=None,
    dangerous_decision_coef: float = 0.0,
    positive_labels=None,
    positive_stop_decision_coef: float = 0.0,
    protocol_labels=None,
    protocol_reference_coef: float = 0.0,
    protocol_reference_margin: float = 0.05,
):
    import torch
    import torch.nn.functional as functional

    policy_logits = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
    ).logits[:, :-1, :]
    sequence_logp, token_logp, token_mask, sequence_lengths = (
        sequence_log_probs_from_logits(
            policy_logits,
            batch_labels,
        )
    )
    advantages = advantages.to(sequence_logp.device)
    absolute_weight = advantages.abs()
    positive_mask = advantages > 0
    negative_mask = advantages < 0
    if dangerous_decision_coef < 0:
        raise ValueError("dangerous decision coefficient must be non-negative")
    if positive_stop_decision_coef < 0:
        raise ValueError("positive stop decision coefficient must be non-negative")
    if protocol_reference_coef < 0:
        raise ValueError("protocol reference coefficient must be non-negative")
    if protocol_reference_margin < 0:
        raise ValueError("protocol reference margin must be non-negative")
    if protocol_reference_coef > 0 and protocol_labels is None:
        raise ValueError("protocol reference loss requires protocol labels")
    dangerous_active = torch.zeros_like(sequence_logp, dtype=torch.bool)
    dangerous_sequence_logp = torch.zeros_like(sequence_logp)
    if dangerous_labels is not None and dangerous_decision_coef > 0:
        dangerous_labels = dangerous_labels.to(sequence_logp.device)
        dangerous_active = (dangerous_labels[:, 1:] != -100).any(dim=1)
        if (dangerous_active & ~negative_mask).any():
            raise ValueError("dangerous decision labels require negative advantages")
        dangerous_sequence_logp, _danger_token, _danger_mask, _danger_lengths = (
            sequence_log_probs_from_logits(
                policy_logits,
                dangerous_labels,
            )
        )
    positive_active = torch.zeros_like(sequence_logp, dtype=torch.bool)
    positive_sequence_logp = torch.zeros_like(sequence_logp)
    if positive_labels is not None and positive_stop_decision_coef > 0:
        positive_labels = positive_labels.to(sequence_logp.device)
        positive_active = (positive_labels[:, 1:] != -100).any(dim=1)
        if (positive_active & ~positive_mask).any():
            raise ValueError("positive decision labels require positive advantages")
        positive_sequence_logp, _positive_token, _positive_mask, _positive_lengths = (
            sequence_log_probs_from_logits(
                policy_logits,
                positive_labels,
            )
        )
    reference_sequence_logp = None
    reference_token_logp = None
    reference_logits = None
    if reference_model is not None:
        with torch.no_grad():
            reference_logits = reference_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            ).logits[:, :-1, :]
            reference_sequence_logp, reference_token_logp, _mask, _lengths = (
                sequence_log_probs_from_logits(
                    reference_logits,
                    batch_labels,
                )
            )

    if negative_objective != "reference_margin":
        raise ValueError(f"unknown negative objective: {negative_objective}")
    if negative_mask.any() and reference_sequence_logp is None:
        raise ValueError("reference-margin negatives require a reference model")
    positive_loss = torch.where(
        positive_mask,
        advantages * (-sequence_logp),
        torch.zeros_like(sequence_logp),
    )
    negative_objective_values = torch.zeros_like(sequence_logp)
    if reference_sequence_logp is not None:
        negative_objective_values = functional.relu(
            sequence_logp - reference_sequence_logp + negative_margin
        )
    negative_loss = torch.where(
        negative_mask,
        absolute_weight * negative_objective_values,
        torch.zeros_like(sequence_logp),
    )
    policy_loss = (positive_loss + negative_loss).mean()

    dangerous_objective_values = torch.zeros_like(sequence_logp)
    dangerous_loss = torch.tensor(0.0, device=sequence_logp.device)
    if dangerous_active.any():
        if reference_logits is None:
            raise ValueError("dangerous decision loss requires a reference model")
        reference_dangerous_logp, _token, _mask, _lengths = (
            sequence_log_probs_from_logits(
                reference_logits,
                dangerous_labels,
            )
        )
        dangerous_objective_values = functional.relu(
            dangerous_sequence_logp
            - reference_dangerous_logp
            + negative_margin
        )
        dangerous_loss = (
            absolute_weight[dangerous_active]
            * dangerous_objective_values[dangerous_active]
        ).sum() / sequence_logp.shape[0]
        policy_loss = policy_loss + dangerous_decision_coef * dangerous_loss

    positive_stop_loss = torch.tensor(0.0, device=sequence_logp.device)
    if positive_active.any():
        positive_stop_loss = (
            absolute_weight[positive_active]
            * (-positive_sequence_logp[positive_active])
        ).sum() / sequence_logp.shape[0]
        policy_loss = policy_loss + positive_stop_decision_coef * positive_stop_loss

    reference_loss = torch.tensor(0.0, device=sequence_logp.device)
    reference_token_logp_mse = torch.tensor(0.0, device=sequence_logp.device)
    weighted_reference_token_logp_mse = torch.tensor(
        0.0,
        device=sequence_logp.device,
    )
    if reference_coef > 0 and reference_token_logp is not None:
        sequence_delta_squared = (
            ((token_logp - reference_token_logp).pow(2) * token_mask.float()).sum(dim=1)
            / sequence_lengths
        )
        reference_token_logp_mse = sequence_delta_squared.mean()
        weighted_reference_token_logp_mse = (
            absolute_weight * sequence_delta_squared
        ).mean()
        reference_loss = reference_coef * weighted_reference_token_logp_mse

    protocol_reference_loss = torch.tensor(0.0, device=sequence_logp.device)
    protocol_floor_loss = torch.tensor(0.0, device=sequence_logp.device)
    protocol_token_count = torch.tensor(0.0, device=sequence_logp.device)
    protocol_downward_mean = torch.tensor(0.0, device=sequence_logp.device)
    protocol_downward_rms = torch.tensor(0.0, device=sequence_logp.device)
    protocol_downward_max = torch.tensor(0.0, device=sequence_logp.device)
    protocol_violation_fraction = torch.tensor(0.0, device=sequence_logp.device)
    if protocol_labels is not None:
        if reference_logits is None:
            raise ValueError("protocol reference loss requires a reference model")
        protocol_labels = protocol_labels.to(sequence_logp.device)
        protocol_stats = protocol_reference_statistics_from_logits(
            policy_logits,
            reference_logits,
            protocol_labels,
            protocol_reference_margin,
        )
        protocol_floor_loss = protocol_stats["floor_loss"]
        protocol_reference_loss = protocol_reference_coef * protocol_floor_loss
        protocol_token_count = protocol_stats["protected_token_count"]
        protocol_downward_mean = protocol_stats["downward_mean"]
        protocol_downward_rms = protocol_stats["downward_rms"]
        protocol_downward_max = protocol_stats["downward_max"]
        protocol_violation_fraction = protocol_stats["violation_fraction"]

    positive_nll = (
        -sequence_logp[positive_mask].mean()
        if positive_mask.any()
        else torch.tensor(0.0, device=sequence_logp.device)
    )
    negative_nll = (
        -sequence_logp[negative_mask].mean()
        if negative_mask.any()
        else torch.tensor(0.0, device=sequence_logp.device)
    )
    negative_objective_loss = (
        negative_objective_values[negative_mask].mean()
        if negative_mask.any()
        else torch.tensor(0.0, device=sequence_logp.device)
    )
    return policy_loss + reference_loss + protocol_reference_loss, {
        "policy_loss": policy_loss.detach(),
        "reference_loss": reference_loss.detach(),
        "reference_token_logp_mse": reference_token_logp_mse.detach(),
        "weighted_reference_token_logp_mse": (
            weighted_reference_token_logp_mse.detach()
        ),
        "protocol_reference_loss": protocol_reference_loss.detach(),
        "protocol_floor_loss": protocol_floor_loss.detach(),
        "protocol_token_count": protocol_token_count.detach(),
        "protocol_downward_drift_mean": protocol_downward_mean.detach(),
        "protocol_downward_drift_rms": protocol_downward_rms.detach(),
        "protocol_downward_drift_max": protocol_downward_max.detach(),
        "protocol_floor_violation_fraction": (
            protocol_violation_fraction.detach()
        ),
        "advantage_mean": advantages.mean().detach(),
        "advantage_abs_mean": absolute_weight.mean().detach(),
        "positive_nll": positive_nll.detach(),
        "negative_nll": negative_nll.detach(),
        "negative_objective_loss": negative_objective_loss.detach(),
        "dangerous_decision_loss": dangerous_loss.detach(),
        "dangerous_fraction": dangerous_active.float().mean().detach(),
        "dangerous_mass_fraction": (
            absolute_weight[dangerous_active].sum()
            / absolute_weight[negative_mask].sum().clamp(min=1e-6)
            if negative_mask.any()
            else torch.tensor(0.0, device=sequence_logp.device)
        ).detach(),
        "positive_stop_decision_loss": positive_stop_loss.detach(),
        "positive_stop_fraction": positive_active.float().mean().detach(),
        "positive_fraction": positive_mask.float().mean().detach(),
    }


def dummy_batch(tokenizer) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    import torch

    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.full((1, 8), pad_token_id, dtype=torch.long)
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    dangerous_labels = torch.full((1, 8), -100, dtype=torch.long)
    positive_labels = torch.full((1, 8), -100, dtype=torch.long)
    protocol_labels = torch.full((1, 8), -100, dtype=torch.long)
    completion_labels = torch.full((1, 8), -100, dtype=torch.long)
    advantages = torch.zeros((1,), dtype=torch.float32)
    return (
        input_ids,
        attention_mask,
        labels,
        dangerous_labels,
        positive_labels,
        protocol_labels,
        completion_labels,
        advantages,
    )


def shard_and_pad_records(
    records: Sequence[AWRRecord],
    *,
    world_size: int,
    rank: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> List[AWRRecord]:
    records_per_update = world_size * batch_size * gradient_accumulation_steps
    padded_count = math.ceil(len(records) / records_per_update) * records_per_update
    repeated = list(records)
    repeated.extend(
        records[index % len(records)]
        for index in range(padded_count - len(records))
    )
    return repeated[rank::world_size]


def save_adapter(accelerator, policy, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def evaluate_policy(
    policy,
    reference_model,
    records: Sequence[AWRRecord],
    tokenizer,
    accelerator,
    *,
    batch_size: int,
    max_seq_length: int,
    negative_margin: float,
    protocol_reference_margin: float,
) -> Dict[str, float]:
    import torch

    if not records:
        return {
            "signed_objective": float("nan"),
            "positive_nll": float("nan"),
            "negative_nll": float("nan"),
            "negative_margin_loss": float("nan"),
            "rows": 0.0,
        }
    rank_records = list(records)[accelerator.process_index :: accelerator.num_processes]
    totals = torch.zeros(20, device=accelerator.device)
    protocol_downward_max = torch.tensor(0.0, device=accelerator.device)
    policy.eval()
    with torch.no_grad():
        for offset in range(0, len(rank_records), batch_size):
            tensors = collate_batch(
                rank_records[offset : offset + batch_size],
                tokenizer,
                max_seq_length,
            )
            if tensors is None:
                continue
            (
                input_ids,
                attention_mask,
                labels,
                dangerous_labels,
                positive_labels,
                protocol_labels,
                completion_labels,
                advantages,
            ) = (tensor.to(accelerator.device) for tensor in tensors)
            policy_logits = policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]
            sequence_logp, token_logp, token_mask, sequence_lengths = (
                sequence_log_probs_from_logits(policy_logits, labels)
            )
            absolute = advantages.abs()
            nll = -sequence_logp
            positive = advantages > 0
            negative = advantages < 0
            reference_logits = reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]
            (
                reference_sequence_logp,
                reference_token_logp,
                _ref_mask,
                _ref_lengths,
            ) = sequence_log_probs_from_logits(reference_logits, labels)
            protocol_stats = protocol_reference_statistics_from_logits(
                policy_logits,
                reference_logits,
                protocol_labels,
                protocol_reference_margin,
            )
            margin_loss = torch.relu(
                sequence_logp - reference_sequence_logp + negative_margin
            )
            totals[0] += (advantages * nll).sum()
            totals[1] += absolute.sum()
            totals[2] += nll[positive].sum()
            totals[3] += positive.sum()
            totals[4] += nll[negative].sum()
            totals[5] += negative.sum()
            totals[6] += advantages.numel()
            totals[7] += margin_loss[negative].sum()
            sequence_delta_squared = (
                (
                    (token_logp - reference_token_logp).pow(2)
                    * token_mask.float()
                ).sum(dim=1)
                / sequence_lengths
            )
            totals[12] += sequence_delta_squared.sum()
            totals[13] += sequence_delta_squared.numel()
            totals[14] += protocol_stats["floor_sum"]
            totals[15] += protocol_stats["protected_token_count"]
            totals[16] += (completion_labels[:, 1:] != -100).sum()
            totals[17] += protocol_stats["downward_sum"]
            totals[18] += protocol_stats["downward_square_sum"]
            totals[19] += protocol_stats["violation_count"]
            protocol_downward_max = torch.maximum(
                protocol_downward_max,
                protocol_stats["downward_max"],
            )
            dangerous = (dangerous_labels[:, 1:] != -100).any(dim=1)
            if dangerous.any():
                dangerous_sequence_logp, _token, _mask, _lengths = (
                    sequence_log_probs_from_logits(
                        policy_logits,
                        dangerous_labels,
                    )
                )
                reference_dangerous_logp, _token, _mask, _lengths = (
                    sequence_log_probs_from_logits(
                        reference_logits,
                        dangerous_labels,
                    )
                )
                dangerous_margin = torch.relu(
                    dangerous_sequence_logp
                    - reference_dangerous_logp
                    + negative_margin
                )
                totals[8] += dangerous_margin[dangerous].sum()
                totals[9] += dangerous.sum()
            positive_stop = (positive_labels[:, 1:] != -100).any(dim=1)
            if positive_stop.any():
                positive_stop_sequence_logp, _token, _mask, _lengths = (
                    sequence_log_probs_from_logits(
                        policy_logits,
                        positive_labels,
                    )
                )
                totals[10] += (-positive_stop_sequence_logp[positive_stop]).sum()
                totals[11] += positive_stop.sum()
    totals = accelerator.reduce(totals, reduction="sum")
    protocol_downward_max = accelerator.gather(
        protocol_downward_max.reshape(1)
    ).max()
    policy.train()
    return {
        "signed_objective": (totals[0] / totals[1].clamp(min=1e-6)).item(),
        "positive_nll": (totals[2] / totals[3].clamp(min=1)).item(),
        "negative_nll": (totals[4] / totals[5].clamp(min=1)).item(),
        "negative_margin_loss": (
            totals[7] / totals[5].clamp(min=1)
        ).item(),
        "positive_rows": totals[3].item(),
        "negative_rows": totals[5].item(),
        "dangerous_decision_margin_loss": (
            totals[8] / totals[9].clamp(min=1)
        ).item(),
        "dangerous_decision_rows": totals[9].item(),
        "positive_stop_decision_loss": (
            totals[10] / totals[11].clamp(min=1)
        ).item(),
        "positive_stop_decision_rows": totals[11].item(),
        "reference_token_logp_mse": (
            totals[12] / totals[13].clamp(min=1)
        ).item(),
        "reference_token_logp_rms": (
            totals[12] / totals[13].clamp(min=1)
        ).sqrt().item(),
        "protocol_reference_floor_loss": (
            totals[14] / totals[15].clamp(min=1)
        ).item(),
        "protocol_protected_tokens": totals[15].item(),
        "protocol_completion_tokens": totals[16].item(),
        "protocol_protected_token_fraction": (
            totals[15] / totals[16].clamp(min=1)
        ).item(),
        "protocol_downward_drift_mean": (
            totals[17] / totals[15].clamp(min=1)
        ).item(),
        "protocol_downward_drift_rms": (
            totals[18] / totals[15].clamp(min=1)
        ).sqrt().item(),
        "protocol_downward_drift_max": protocol_downward_max.item(),
        "protocol_floor_violation_fraction": (
            totals[19] / totals[15].clamp(min=1)
        ).item(),
        "rows": totals[6].item(),
    }


def main() -> None:
    args = parse_args()
    if args.num_epochs <= 0:
        raise SystemExit("--num-epochs must be positive")
    if args.reference_coef < 0:
        raise SystemExit("--reference-coef must be non-negative")
    if args.protocol_reference_coef < 0:
        raise SystemExit("--protocol-reference-coef must be non-negative")
    if args.protocol_reference_margin < 0:
        raise SystemExit("--protocol-reference-margin must be non-negative")
    if args.negative_margin <= 0:
        raise SystemExit("--negative-margin must be positive")
    if args.dangerous_decision_coef < 0:
        raise SystemExit("--dangerous-decision-coef must be non-negative")
    if args.positive_stop_decision_coef < 0:
        raise SystemExit("--positive-stop-decision-coef must be non-negative")

    output_dir = Path(args.out_dir)
    logger = setup_logging(output_dir, args.agent)
    train_records = load_records(
        args.rollout,
        args.agent,
        args.advantage_field,
        args.mask_field,
        args.negative_mask_policy,
    )
    validation_records = (
        load_records(
            args.validation_rollout,
            args.agent,
            args.advantage_field,
            args.mask_field,
            args.negative_mask_policy,
        )
        if args.validation_rollout
        else []
    )
    if not train_records:
        raise SystemExit(f"no signed-AWR training records for {args.agent}")
    positive_count = sum(record.advantage > 0 for record in train_records)
    negative_count = sum(record.advantage < 0 for record in train_records)
    dangerous_count = sum(
        bool(record.dangerous_decision_fields) for record in train_records
    )
    positive_stop_count = sum(
        bool(record.positive_decision_fields) for record in train_records
    )
    if not positive_count or not negative_count:
        raise SystemExit(
            f"{args.agent} requires both reward signs: "
            f"positive={positive_count} negative={negative_count}"
        )
    logger.info(
        "Loaded %d train and %d holdout records; signs=+%d/-%d "
        "dangerous=%d positive_stop=%d sources=%s",
        len(train_records),
        len(validation_records),
        positive_count,
        negative_count,
        dangerous_count,
        positive_stop_count,
        dict(Counter(record.sample_source for record in train_records)),
    )
    logger.info(
        "Advantage min=%.4f mean=%.4f max=%.4f masks=%s",
        min(record.advantage for record in train_records),
        sum(record.advantage for record in train_records) / len(train_records),
        max(record.advantage for record in train_records),
        dict(Counter(record.mask_mode for record in train_records)),
    )

    import torch
    from accelerate import Accelerator
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    is_main = accelerator.is_main_process
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if not tokenizer.is_fast:
        raise SystemExit("signed-AWR decision masks require a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy = PeftModel.from_pretrained(
        base_model,
        str(args.sft_adapter),
        is_trainable=True,
    )
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        policy.enable_input_require_grads()
    if is_main:
        policy.print_trainable_parameters()

    logger.info(
        "Loading frozen SFT reference adapter for margin-bounded negatives "
        "and protocol-token floor"
    )
    reference_base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    reference_model = PeftModel.from_pretrained(
        reference_base,
        str(args.sft_adapter),
        is_trainable=False,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    effective_batch = (
        accelerator.num_processes
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(len(train_records) / effective_batch))
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    policy, optimizer = accelerator.prepare(policy, optimizer)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    reference_model = reference_model.to(accelerator.device)
    logger.info(
        "world_size=%d effective_batch=%d total_steps=%d lr=%.2e "
        "reference=%.4f protocol_reference=%.4f protocol_margin=%.3f "
        "negative_margin=%.3f negative_mask=%s "
        "dangerous_coef=%.3f positive_stop_coef=%.3f",
        accelerator.num_processes,
        effective_batch,
        total_steps,
        args.learning_rate,
        args.reference_coef,
        args.protocol_reference_coef,
        args.protocol_reference_margin,
        args.negative_margin,
        args.negative_mask_policy,
        args.dangerous_decision_coef,
        args.positive_stop_decision_coef,
    )

    random_generator = random.Random(args.seed)
    global_step = 0
    accumulated_metrics: Dict[str, float] = defaultdict(float)
    accumulated_count = 0
    optimizer.zero_grad()
    policy.train()
    for epoch in range(args.num_epochs):
        random_generator.shuffle(train_records)
        rank_records = shard_and_pad_records(
            train_records,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        logger.info(
            "Epoch %d/%d rank=%d records=%d",
            epoch + 1,
            args.num_epochs,
            accelerator.process_index,
            len(rank_records),
        )
        for offset in range(0, len(rank_records), args.per_device_batch_size):
            batch = rank_records[offset : offset + args.per_device_batch_size]
            tensors = collate_batch(batch, tokenizer, args.max_seq_length)
            if tensors is None:
                tensors = dummy_batch(tokenizer)
            (
                input_ids,
                attention_mask,
                labels,
                dangerous_labels,
                positive_labels,
                protocol_labels,
                _completion_labels,
                advantages,
            ) = (tensor.to(accelerator.device) for tensor in tensors)
            with accelerator.accumulate(policy):
                loss, metrics = compute_signed_awr_loss(
                    policy,
                    reference_model,
                    input_ids,
                    attention_mask,
                    labels,
                    advantages,
                    reference_coef=args.reference_coef,
                    negative_margin=args.negative_margin,
                    negative_objective="reference_margin",
                    dangerous_labels=dangerous_labels,
                    dangerous_decision_coef=args.dangerous_decision_coef,
                    positive_labels=positive_labels,
                    positive_stop_decision_coef=args.positive_stop_decision_coef,
                    protocol_labels=protocol_labels,
                    protocol_reference_coef=args.protocol_reference_coef,
                    protocol_reference_margin=args.protocol_reference_margin,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [
                            parameter
                            for parameter in policy.parameters()
                            if parameter.requires_grad
                        ],
                        max_norm=1.0,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad()

            for name, value in metrics.items():
                accumulated_metrics[name] += float(value.item())
            accumulated_count += 1
            if (
                accelerator.sync_gradients
                and global_step % args.logging_steps == 0
                and is_main
            ):
                denominator = max(accumulated_count, 1)
                logger.info(
                    "step=%d policy=%.4f reference=%.4f drift_mse=%.4f "
                    "drift_rms=%.4f protocol_ref=%.4f protocol_floor=%.4f "
                    "protocol_drop_mean=%.4f protocol_drop_rms=%.4f "
                    "protocol_drop_max_batch_mean=%.4f protocol_viol=%.3f adv=%.3f "
                    "pos_nll=%.4f neg_nll=%.4f neg_obj=%.4f "
                    "danger=%.4f danger_frac=%.3f danger_mass=%.3f "
                    "positive_stop=%.4f stop_frac=%.3f pos_frac=%.3f lr=%.2e",
                    global_step,
                    accumulated_metrics["policy_loss"] / denominator,
                    accumulated_metrics["reference_loss"] / denominator,
                    accumulated_metrics["reference_token_logp_mse"] / denominator,
                    math.sqrt(
                        max(
                            accumulated_metrics["reference_token_logp_mse"]
                            / denominator,
                            0.0,
                        )
                    ),
                    accumulated_metrics["protocol_reference_loss"] / denominator,
                    accumulated_metrics["protocol_floor_loss"] / denominator,
                    accumulated_metrics["protocol_downward_drift_mean"] / denominator,
                    accumulated_metrics["protocol_downward_drift_rms"] / denominator,
                    accumulated_metrics["protocol_downward_drift_max"] / denominator,
                    accumulated_metrics["protocol_floor_violation_fraction"] / denominator,
                    accumulated_metrics["advantage_mean"] / denominator,
                    accumulated_metrics["positive_nll"] / denominator,
                    accumulated_metrics["negative_nll"] / denominator,
                    accumulated_metrics["negative_objective_loss"] / denominator,
                    accumulated_metrics["dangerous_decision_loss"] / denominator,
                    accumulated_metrics["dangerous_fraction"] / denominator,
                    accumulated_metrics["dangerous_mass_fraction"] / denominator,
                    accumulated_metrics["positive_stop_decision_loss"] / denominator,
                    accumulated_metrics["positive_stop_fraction"] / denominator,
                    accumulated_metrics["positive_fraction"] / denominator,
                    scheduler.get_last_lr()[0],
                )
                accumulated_metrics = defaultdict(float)
                accumulated_count = 0
            if (
                accelerator.sync_gradients
                and global_step > 0
                and global_step % args.save_steps == 0
            ):
                accelerator.wait_for_everyone()
                if is_main:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    save_adapter(accelerator, policy, tokenizer, checkpoint_dir)
                    checkpoints = sorted(
                        output_dir.glob("checkpoint-*"),
                        key=lambda path: int(path.name.split("-")[1]),
                    )
                    for old_checkpoint in checkpoints[: -args.save_total_limit]:
                        shutil.rmtree(old_checkpoint, ignore_errors=True)
                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    holdout_metrics = evaluate_policy(
        policy,
        reference_model,
        validation_records,
        tokenizer,
        accelerator,
        batch_size=args.per_device_batch_size,
        max_seq_length=args.max_seq_length,
        negative_margin=args.negative_margin,
        protocol_reference_margin=args.protocol_reference_margin,
    )
    if is_main:
        final_dir = output_dir / "final"
        save_adapter(accelerator, policy, tokenizer, final_dir)
        summary = {
            "agent": args.agent,
            "algorithm": ALGORITHM_NAME,
            "global_step": global_step,
            "train_records": len(train_records),
            "positive_records": positive_count,
            "negative_records": negative_count,
            "dangerous_decision_records": dangerous_count,
            "positive_stop_decision_records": positive_stop_count,
            "validation_records": len(validation_records),
            "validation": holdout_metrics,
            "learning_rate": args.learning_rate,
            "reference_coef": args.reference_coef,
            "reference_regularizer": "advantage_weighted_sampled_token_logp_mse",
            "protocol_reference_coef": args.protocol_reference_coef,
            "protocol_reference_margin": args.protocol_reference_margin,
            "protocol_reference_regularizer": (
                "unweighted_one_sided_token_logp_hinge_square"
            ),
            "protocol_protected_fields": ["action", "handoff_target"],
            "protocol_free_content_fields": list(PROTOCOL_FREE_CONTENT_FIELDS),
            "negative_margin": args.negative_margin,
            "negative_mask_policy": args.negative_mask_policy,
            "dangerous_decision_coef": args.dangerous_decision_coef,
            "positive_stop_decision_coef": args.positive_stop_decision_coef,
            "num_epochs": args.num_epochs,
            "world_size": accelerator.num_processes,
            "mask_modes": dict(Counter(record.mask_mode for record in train_records)),
            "negative_objective": "reference_margin",
            "reward_classes": dict(
                Counter(record.reward_class for record in train_records)
            ),
        }
        (final_dir / "rl_train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Final adapter saved: %s", final_dir)
        logger.info("Holdout signed metrics: %s", holdout_metrics)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
