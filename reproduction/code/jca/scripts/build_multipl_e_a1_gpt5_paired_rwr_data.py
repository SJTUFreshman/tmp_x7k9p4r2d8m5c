#!/usr/bin/env python3
"""Build a high-confidence, within-task paired signed-RWR dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


LANGUAGES = ("py", "cpp", "java", "php", "ts", "cs", "sh", "js")
PROTOCOL_KEYS = {
    "reasoning",
    "tentative_completion",
    "action",
    "handoff_target",
    "handoff_note",
    "confirmed_completion",
}
POLICY_VERSION = "multipl_e_gpt5_within_task_paired_rwr_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conflict-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--pairs-per-language", type=int, default=120)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", choices=("A1", "A2", "A3"), default="A1")
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument(
        "--reward-source",
        choices=("judged", "execution"),
        default="judged",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    payload = "\0".join(
        (
            str(seed),
            str(row.get("root_dataset", "")),
            str(row.get("language", "")),
            str(row.get("problem_id", "")),
            str(row.get("rollout_idx", "")),
            str(row.get("turn", "")),
            str(row.get("response", "")),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("root_dataset", "")),
        str(row.get("language", "")),
        str(row.get("problem_id", "")),
    )


def valid_agent_response(row: dict[str, Any]) -> bool:
    try:
        response = json.loads(str(row.get("response", "")))
    except json.JSONDecodeError:
        return False
    if not isinstance(response, dict) or not PROTOCOL_KEYS.issubset(response):
        return False
    action = response.get("action")
    if action == "handoff":
        return bool(str(response.get("tentative_completion") or "").strip())
    if action == "confirm_stop":
        return bool(str(response.get("confirmed_completion") or response.get("tentative_completion") or "").strip())
    return False


def valid_turn_zero_response(row: dict[str, Any]) -> bool:
    return int(row.get("turn", -1)) == 0 and valid_agent_response(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    if args.pairs_per_language <= 0:
        raise SystemExit("--pairs-per-language must be positive")
    if not 0 < args.min_confidence <= 1:
        raise SystemExit("--min-confidence must be in (0, 1]")

    rows = read_jsonl(args.input)
    rejected = Counter()
    conflicts: list[dict[str, Any]] = []
    pools: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: {1: [], -1: []}
    )

    for row in rows:
        if row.get("agent_id") != args.agent:
            rejected["not_target_agent"] += 1
            continue
        if int(row.get("turn", -1)) != args.turn:
            rejected["not_target_turn"] += 1
            continue
        if row.get("data_split") != "train":
            rejected["not_train_split"] += 1
            continue
        language = str(row.get("language", ""))
        if language not in LANGUAGES:
            rejected["unsupported_language"] += 1
            continue
        if args.reward_source == "judged" and row.get("judge_failed") is True:
            rejected["judge_failed"] += 1
            continue
        try:
            task_reward = float(row["task_reward"])
        except (KeyError, TypeError, ValueError):
            rejected["invalid_reward"] += 1
            continue

        execution_sign = 1 if task_reward > 0 else -1
        if args.reward_source == "judged":
            try:
                reward = float(row["reward"])
            except (KeyError, TypeError, ValueError):
                rejected["invalid_reward"] += 1
                continue
            mixed_sign = 1 if reward > 0 else -1 if reward < 0 else 0
            if mixed_sign != execution_sign:
                conflict = dict(row)
                conflict["reserve_reason"] = "execution_mixed_reward_sign_conflict"
                conflict["selection_policy"] = POLICY_VERSION
                conflicts.append(conflict)
                rejected["conflict_reserve"] += 1
                continue
        else:
            row = dict(row)
            row["reward"] = task_reward
            row["reward_semantics"] = "per_turn_execution_v2"
            row["reward_source"] = "execution_only"
            row["judge_failed"] = False
            row["judge_status"] = "not_applicable_execution_only"
        if not row.get("messages") or not str(row.get("response", "")).strip():
            rejected["missing_training_text"] += 1
            continue
        if not valid_agent_response(row):
            rejected["invalid_agent_protocol"] += 1
            continue
        pools[task_key(row)][execution_sign].append(row)

    selected: list[dict[str, Any]] = []
    selection_by_language: dict[str, Any] = {}
    selected_pair_ids: set[str] = set()
    for language in LANGUAGES:
        candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
        mixed_tasks = 0
        for key, signs in pools.items():
            if key[1] != language or not signs[1] or not signs[-1]:
                continue
            mixed_tasks += 1
            positive = max(
                signs[1], key=lambda row: (float(row["reward"]), stable_rank(args.seed, row))
            )
            negative = min(
                signs[-1], key=lambda row: (float(row["reward"]), stable_rank(args.seed, row))
            )
            if int(positive.get("rollout_idx", -1)) == int(negative.get("rollout_idx", -1)):
                continue
            if str(positive.get("response", "")) == str(negative.get("response", "")):
                continue
            confidence = min(abs(float(positive["reward"])), abs(float(negative["reward"])))
            tie_break = stable_rank(args.seed + 1, positive) + stable_rank(args.seed + 2, negative)
            if confidence >= args.min_confidence:
                candidates.append((confidence, tie_break, positive, negative))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = candidates[: args.pairs_per_language]
        for confidence, _, positive, negative in chosen:
            pair_payload = "\0".join(
                (*task_key(positive), str(positive["rollout_idx"]), str(negative["rollout_idx"]), str(args.seed))
            )
            pair_id = hashlib.sha256(pair_payload.encode()).hexdigest()[:20]
            if pair_id in selected_pair_ids:
                raise AssertionError(f"duplicate pair_id: {pair_id}")
            selected_pair_ids.add(pair_id)
            for sign, source in ((1, positive), (-1, negative)):
                output = dict(source)
                output.update(
                    {
                        "train_weight": round(sign * confidence, 4),
                        "pair_confidence": round(confidence, 4),
                        "pair_id": pair_id,
                        "pair_role": "positive" if sign > 0 else "negative",
                        "selection_policy": POLICY_VERSION,
                        "selection_seed": args.seed,
                    }
                )
                selected.append(output)
        selection_by_language[language] = {
            "mixed_sign_tasks": mixed_tasks,
            "eligible_pairs_at_min_confidence": len(candidates),
            "selected_pairs": len(chosen),
            "selected_rows": 2 * len(chosen),
        }

    selected.sort(key=lambda row: (str(row["pair_id"]), str(row["pair_role"])))
    conflicts.sort(key=lambda row: stable_rank(args.seed + 3, row))
    write_jsonl(args.output, selected)
    write_jsonl(args.conflict_output, conflicts)

    weights = [float(row["train_weight"]) for row in selected]
    eligible_pool_rows = sum(
        len(sign_rows) for signs in pools.values() for sign_rows in signs.values()
    )
    classified_before_pair_selection = sum(rejected.values()) + eligible_pool_rows
    if classified_before_pair_selection != len(rows):
        raise AssertionError(
            "input classification does not conserve rows: "
            f"{classified_before_pair_selection} != {len(rows)}"
        )
    stats = {
        "policy_version": POLICY_VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "input_rows": len(rows),
        "seed": args.seed,
        "agent": args.agent,
        "turn": args.turn,
        "reward_source": args.reward_source,
        "min_confidence": args.min_confidence,
        "pairs_per_language_cap": args.pairs_per_language,
        "output": str(args.output.resolve()),
        "conflict_output": str(args.conflict_output.resolve()),
        "selected_pairs": len(selected) // 2,
        "selected_rows": len(selected),
        "train_weight_positive": sum(value > 0 for value in weights),
        "train_weight_negative": sum(value < 0 for value in weights),
        "train_weight_mean": sum(weights) / len(weights) if weights else None,
        "conflict_reserve_rows": len(conflicts),
        "eligible_pool_rows": eligible_pool_rows,
        "eligible_pool_tasks": len(pools),
        "unselected_eligible_pool_rows": eligible_pool_rows - len(selected),
        "classified_input_rows": classified_before_pair_selection,
        "input_classification_conserved": classified_before_pair_selection == len(rows),
        "selection_by_language": selection_by_language,
        "rejected_before_pair_selection": dict(sorted(rejected.items())),
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
