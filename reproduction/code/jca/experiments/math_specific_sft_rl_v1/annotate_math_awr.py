#!/usr/bin/env python3
"""Annotate freshly prepared MATH rows for conservative signed AWR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ANNOTATION_VERSION = "math_specific_conservative_awr_v3_protocol_floor"
ALGORITHM_NAME = "conservative_signed_awr_v4_protocol_floor_v1"
AGENTS = ("A1", "A2", "A3")
POSITIVE_CLASSES = {"proposal_correct", "wrong_to_correct", "correct_to_correct"}
NEGATIVE_CLASSES = {"proposal_wrong", "wrong_to_wrong", "correct_to_wrong"}
DECISION_FIELD_MAP = {
    "proposal_wrong": ["tentative_answer"],
    "wrong_to_wrong": ["tentative_answer"],
    "correct_to_wrong": ["tentative_answer", "confirmed_answer"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--holdout-input", type=Path, required=True)
    parser.add_argument("--base-stats", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--sft-stats", type=Path, required=True)
    parser.add_argument("--sft-a1", type=Path, required=True)
    parser.add_argument("--sft-a2", type=Path, required=True)
    parser.add_argument("--sft-a3", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--sample-source-v4",
        default="math_specific_sft_judged_signed_awr",
    )
    parser.add_argument(
        "--sampling-policy-lineage",
        default="MATH-specific SFT",
    )
    parser.add_argument(
        "--reuse-legacy-sampled-data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def adapter_meta(path: Path) -> dict[str, Any]:
    files = [path / "adapter_config.json", path / "adapter_model.safetensors"]
    for file_path in files:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    entries = []
    for file_path in files:
        entry = file_meta(file_path)
        entries.append(entry)
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "files": entries}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def protocol_object(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("protocol_response", row.get("response"))
    if not isinstance(payload, str):
        raise ValueError("missing protocol response")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("protocol response is not an object")
    required = {
        "reasoning",
        "tentative_answer",
        "action",
        "handoff_target",
        "handoff_note",
        "confirmed_answer",
    }
    if set(value) != required:
        raise ValueError("protocol response fields do not match the canonical schema")
    if value["action"] not in {"handoff", "confirm_stop"}:
        raise ValueError("invalid protocol action")
    return value


def transform_row(
    source: dict[str, Any],
    *,
    split: str,
    line_number: int,
    sample_source_v4: str = "math_specific_sft_judged_signed_awr",
    sampling_policy_lineage: str = "MATH-specific SFT",
    reuse_legacy_sampled_data: bool = False,
) -> dict[str, Any]:
    row = dict(source)
    agent = str(row.get("agent_id") or "")
    if agent not in AGENTS:
        raise ValueError(f"invalid agent at {split}:{line_number}")
    state = row.get("deterministic_state_v3")
    if not isinstance(state, dict):
        raise ValueError(f"missing deterministic state at {split}:{line_number}")
    reward_class = str(state.get("answer_state") or "")
    if reward_class not in POSITIVE_CLASSES | NEGATIVE_CLASSES:
        raise ValueError(f"unknown reward class at {split}:{line_number}: {reward_class}")
    try:
        weight = float(row["train_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid train_weight at {split}:{line_number}") from exc
    if not math.isfinite(weight) or weight == 0:
        raise ValueError(f"non-finite or zero train_weight at {split}:{line_number}")
    expected_positive = reward_class in POSITIVE_CLASSES
    if (weight > 0) is not expected_positive:
        raise ValueError(
            f"reward sign disagrees with {reward_class} at {split}:{line_number}"
        )
    protocol = protocol_object(row)
    action = str(protocol["action"])
    if row.get("action") not in {None, action}:
        raise ValueError(f"stored action disagrees at {split}:{line_number}")

    mask_mode = "decision" if weight < 0 else "full"
    decision_fields = list(DECISION_FIELD_MAP.get(reward_class, []))
    if mask_mode == "decision" and not decision_fields:
        raise ValueError(
            f"negative verifier class lacks decision fields at {split}:{line_number}"
        )
    for field in decision_fields:
        if field not in protocol:
            raise ValueError(f"protocol is missing {field} at {split}:{line_number}")
    dangerous_stop = weight < 0 and reward_class == "wrong_to_wrong" and action == "confirm_stop"
    positive_stop = weight > 0 and reward_class == "correct_to_correct" and action == "confirm_stop"

    row["annotation_version"] = ANNOTATION_VERSION
    row["sample_source_v4"] = sample_source_v4
    row["sampling_policy_lineage"] = sampling_policy_lineage
    row["legacy_sample_reuse"] = reuse_legacy_sampled_data
    row["reward_class_v4"] = reward_class
    row["loss_mask_mode_v4"] = mask_mode
    row["decision_fields_v4"] = decision_fields
    row["dangerous_stop_v4_3"] = dangerous_stop
    row["dangerous_decision_fields_v4_3"] = ["action"] if dangerous_stop else []
    row["positive_stop_v4_3"] = positive_stop
    row["positive_decision_fields_v4_3"] = ["action"] if positive_stop else []
    row["policy_initialization"] = "MATH-specific SFT"
    row["reference_policy"] = "frozen identical MATH-specific SFT"
    return row


def update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["rows"] += 1
    stats["agents"][row["agent_id"]] += 1
    sign = "positive" if float(row["train_weight"]) > 0 else "negative"
    stats["signs"][(row["agent_id"], sign)] += 1
    stats["reward_classes"][(row["agent_id"], row["reward_class_v4"])] += 1
    stats["masks"][(row["agent_id"], row["loss_mask_mode_v4"])] += 1
    stats["dangerous_stops"] += int(bool(row["dangerous_stop_v4_3"]))
    stats["positive_stops"] += int(bool(row["positive_stop_v4_3"]))


def new_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "agents": Counter(),
        "signs": Counter(),
        "reward_classes": Counter(),
        "masks": Counter(),
        "dangerous_stops": 0,
        "positive_stops": 0,
    }


def nested(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {agent: {} for agent in AGENTS}
    for (agent, name), count in sorted(counter.items()):
        output[agent][name] = count
    return output


def serializable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": stats["rows"],
        "agents": dict(sorted(stats["agents"].items())),
        "signs": nested(stats["signs"]),
        "reward_classes": nested(stats["reward_classes"]),
        "masks": nested(stats["masks"]),
        "dangerous_stops": stats["dangerous_stops"],
        "positive_stops": stats["positive_stops"],
    }


def transform_file(
    input_path: Path,
    temporary_output: Path,
    *,
    split: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], set[str]]:
    stats = new_stats()
    identities: set[tuple[str, int, int]] = set()
    problem_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as source, temporary_output.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank line at {input_path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {input_path}:{line_number}")
            identity = (
                str(value.get("problem_id") or ""),
                int(value.get("rollout_idx", -1)),
                int(value.get("turn", -1)),
            )
            if not identity[0] or min(identity[1:]) < 0 or identity in identities:
                raise ValueError(f"invalid or duplicate identity at {input_path}:{line_number}")
            identities.add(identity)
            problem_ids.add(identity[0])
            row = transform_row(
                value,
                split=split,
                line_number=line_number,
                sample_source_v4=args.sample_source_v4,
                sampling_policy_lineage=args.sampling_policy_lineage,
                reuse_legacy_sampled_data=args.reuse_legacy_sampled_data,
            )
            update_stats(stats, row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    if stats["rows"] == 0:
        raise ValueError(f"empty input: {input_path}")
    return serializable_stats(stats), problem_ids


def validate_train_signs(stats: dict[str, Any]) -> None:
    for agent in AGENTS:
        signs = stats["signs"].get(agent, {})
        if signs.get("positive", 0) <= 0 or signs.get("negative", 0) <= 0:
            raise ValueError(f"train split lacks both reward signs for {agent}: {signs}")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def input_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train": file_meta(args.train_input),
        "holdout": file_meta(args.holdout_input),
        "base_stats": file_meta(args.base_stats),
        "base_manifest": file_meta(args.base_manifest),
        "sft_stats": file_meta(args.sft_stats),
    }


def adapter_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "A1": adapter_meta(args.sft_a1),
        "A2": adapter_meta(args.sft_a2),
        "A3": adapter_meta(args.sft_a3),
    }


def validate_output(
    path: Path,
    *,
    split: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats = new_stats()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank output line at {path}:{line_number}")
            row = json.loads(line)
            if row.get("annotation_version") != ANNOTATION_VERSION:
                raise ValueError(f"stale annotation at {path}:{line_number}")
            if row.get("sample_source_v4") != args.sample_source_v4:
                raise ValueError(f"sample source changed at {path}:{line_number}")
            if row.get("sampling_policy_lineage") != args.sampling_policy_lineage:
                raise ValueError(f"sampling lineage changed at {path}:{line_number}")
            if row.get("legacy_sample_reuse") is not args.reuse_legacy_sampled_data:
                raise ValueError(f"sample reuse flag changed at {path}:{line_number}")
            protocol_object(row)
            update_stats(stats, row)
    result = serializable_stats(stats)
    if result["rows"] == 0:
        raise ValueError(f"empty output: {path}")
    if split == "train":
        validate_train_signs(result)
    return result


def validate_existing(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest_output)
    stats = read_json(args.stats_output)
    if manifest.get("annotation_version") != ANNOTATION_VERSION:
        raise SystemExit("annotation manifest version mismatch")
    if manifest.get("inputs") != input_metadata(args):
        raise SystemExit("annotation inputs changed; choose a new TAG")
    if manifest.get("sft_adapters") != adapter_metadata(args):
        raise SystemExit("MATH SFT adapter fingerprints changed; choose a new TAG")
    if manifest.get("sampling_provenance") != sampling_provenance(args):
        raise SystemExit("sampling provenance changed; choose a new TAG")
    if stats.get("sampling_provenance") != sampling_provenance(args):
        raise SystemExit("stored sampling provenance changed; choose a new TAG")
    train_stats = validate_output(args.train_output, split="train", args=args)
    holdout_stats = validate_output(args.holdout_output, split="holdout", args=args)
    if stats.get("splits") != {"train": train_stats, "holdout": holdout_stats}:
        raise SystemExit("stored annotation stats do not match outputs")
    expected_outputs = {
        "train": file_meta(args.train_output),
        "holdout": file_meta(args.holdout_output),
        "stats": file_meta(args.stats_output),
    }
    if manifest.get("outputs") != expected_outputs:
        raise SystemExit("annotation output fingerprints do not match")
    print(json.dumps({"status": "valid", "train_rows": train_stats["rows"]}))


def sampling_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_source_v4": args.sample_source_v4,
        "sampling_policy_lineage": args.sampling_policy_lineage,
        "legacy_sample_reuse": args.reuse_legacy_sampled_data,
    }


def main() -> None:
    args = parse_args()
    args.sample_source_v4 = args.sample_source_v4.strip()
    args.sampling_policy_lineage = args.sampling_policy_lineage.strip()
    if not args.sample_source_v4 or not args.sampling_policy_lineage:
        raise SystemExit("sampling provenance labels cannot be empty")
    if args.reuse_legacy_sampled_data:
        if args.sampling_policy_lineage == "MATH-specific SFT":
            raise SystemExit("legacy sample reuse requires its actual sampling lineage")
    elif args.sampling_policy_lineage != "MATH-specific SFT":
        raise SystemExit("non-legacy data must use the MATH-specific SFT lineage")
    required = [
        args.train_input,
        args.holdout_input,
        args.base_stats,
        args.base_manifest,
        args.sft_stats,
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")
    if args.validate_existing:
        validate_existing(args)
        return
    outputs = (
        args.train_output,
        args.holdout_output,
        args.stats_output,
        args.manifest_output,
    )
    if any(path.exists() for path in outputs):
        raise SystemExit("AWR output exists; validate it or choose a new TAG")
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = args.train_output.with_name(f".{args.train_output.name}.tmp.{os.getpid()}")
    holdout_temporary = args.holdout_output.with_name(
        f".{args.holdout_output.name}.tmp.{os.getpid()}"
    )
    try:
        train_stats, train_problem_ids = transform_file(
            args.train_input,
            train_temporary,
            split="train",
            args=args,
        )
        holdout_stats, holdout_problem_ids = transform_file(
            args.holdout_input,
            holdout_temporary,
            split="holdout",
            args=args,
        )
        validate_train_signs(train_stats)
        overlap = train_problem_ids & holdout_problem_ids
        if overlap:
            raise ValueError(f"problem-level train/holdout leakage: {len(overlap)} IDs")
        os.replace(train_temporary, args.train_output)
        os.replace(holdout_temporary, args.holdout_output)
    finally:
        train_temporary.unlink(missing_ok=True)
        holdout_temporary.unlink(missing_ok=True)

    stats = {
        "annotation_version": ANNOTATION_VERSION,
        "dataset": "MATH",
        "source_split": "train",
        "objective": {
            "algorithm": ALGORITHM_NAME,
            "negative_mask_policy": "all negative rows use semantic decision fields",
            "negative_protocol_policy": "protocol tokens excluded from negative masks",
            "protocol_reference_policy": (
                "unweighted one-sided token log-prob floor against identical MATH SFT"
            ),
            "protocol_control_fields": ["action", "handoff_target"],
            "dangerous_stop_policy": "wrong_to_wrong_confirm_stop",
            "positive_stop_policy": "correct_to_correct_confirm_stop",
            "policy_initialization": "MATH-specific SFT",
            "frozen_reference": "the identical MATH-specific SFT adapter",
        },
        "splits": {"train": train_stats, "holdout": holdout_stats},
        "sampling_provenance": sampling_provenance(args),
        "base_prepare_stats": read_json(args.base_stats),
    }
    atomic_write_json(args.stats_output, stats)
    experiment = (
        "MATH SFT -> legacy judged MATH rollout reuse -> MATH RL"
        if args.reuse_legacy_sampled_data
        else "MATH-only reproduction: MATH SFT -> MATH rollout -> MATH RL"
    )
    forbidden_provenance = (
        ["GSM dataset", "direct corrected-data patch"]
        if args.reuse_legacy_sampled_data
        else ["GSM SFT adapter", "GSM rollout", "direct corrected-data patch"]
    )
    manifest = {
        "experiment": experiment,
        "annotation_version": ANNOTATION_VERSION,
        "dataset": {"name": "MATH", "sft_split": "train", "rl_split": "train"},
        "forbidden_provenance": forbidden_provenance,
        "sampling_provenance": sampling_provenance(args),
        "inputs": input_metadata(args),
        "sft_adapters": adapter_metadata(args),
        "base_prepare_manifest": read_json(args.base_manifest),
        "training": stats["objective"],
        "outputs": {
            "train": file_meta(args.train_output),
            "holdout": file_meta(args.holdout_output),
            "stats": file_meta(args.stats_output),
        },
    }
    atomic_write_json(args.manifest_output, manifest)
    validate_existing(args)


if __name__ == "__main__":
    main()
