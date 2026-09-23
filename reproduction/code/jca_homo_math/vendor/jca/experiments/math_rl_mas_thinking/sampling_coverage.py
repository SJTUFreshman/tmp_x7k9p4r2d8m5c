from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from jca.src.math_eval import load_math_problems


def _load_state(state_path: Path) -> dict[str, Any]:
    with Path(state_path).open(encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict) or not isinstance(state.get("items"), list):
        raise ValueError("sample state has invalid root/config/items")
    identities = []
    for item in state["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("problem"), dict):
            raise ValueError("sample state has an invalid item")
        if item.get("status") not in {"done", "failed"}:
            raise ValueError("sample state contains unfinished trajectories")
        if (
            type(item.get("problem_index")) is not int
            or type(item.get("rollout_idx")) is not int
            or not isinstance(item.get("trajectory"), dict)
            or not isinstance(item["trajectory"].get("steps"), list)
        ):
            raise ValueError("sample state has invalid identity or trajectory fields")
        identities.append((item.get("problem_index"), item.get("rollout_idx")))
    if len(identities) != len(set(identities)):
        raise ValueError("sample state contains duplicate identities")
    return state


def validate_sampling_coverage(
    source_groups: Mapping[tuple[str, int], Any],
    *,
    expected_rollouts: int,
    data_root: Path,
    failure_ledger: Path | None = None,
    sample_state: Path | None = None,
) -> dict[str, int]:
    if (failure_ledger is None) != (sample_state is None):
        raise ValueError("failure ledger and sample state must be provided together")
    problems = load_math_problems(Path(data_root), split="train")
    expected = {
        (problem.problem_id, rollout_idx)
        for problem in problems
        for rollout_idx in range(expected_rollouts)
    }
    turn_keys = set(source_groups)
    if failure_ledger is None:
        counts = Counter(problem_id for problem_id, _ in turn_keys)
        if any(count != expected_rollouts for count in counts.values()):
            raise ValueError("source does not contain configured rollout count per problem")
        return {
            "groups_with_turns": len(turn_keys),
            "zero_step_failures": 0,
            "total_groups": len(turn_keys),
            "problems": len(counts),
        }
    state = _load_state(Path(sample_state))
    runner = importlib.import_module("jca.experiments.math_rl_mas_thinking.math_role_batched")
    config = state["config"]
    if (
        config.get("split") != "train"
        or config.get("start") != 0
        or config.get("limit") != len(problems)
        or config.get("num_rollouts") != expected_rollouts
        or config.get("subjects")
        or config.get("output_mode") != "turns"
    ):
        raise ValueError("sample state is not the complete configured training split")
    canonical = {
        problem.problem_id: {
            field: getattr(problem, field)
            for field in ("problem_id", "subject", "level", "prompt", "solution", "gold_answer")
        }
        for problem in problems
    }
    state_turn_keys = set()
    state_keys = set()
    for item in state["items"]:
        problem = item["problem"]
        if problem != canonical.get(problem["problem_id"]):
            raise ValueError("sample state problem disagrees with canonical data")
        if item["status"] not in {"done", "failed"}:
            raise ValueError("sample state contains unfinished trajectories")
        state_key = (problem["problem_id"], item["rollout_idx"])
        if state_key in state_keys:
            raise ValueError("sample state contains duplicate problem/rollout identities")
        state_keys.add(state_key)
        if item["trajectory"]["steps"]:
            if item["status"] != "done":
                raise ValueError("failed trajectory with valid turns is not quarantinable")
            state_turn_keys.add((problem["problem_id"], item["rollout_idx"]))
    if state_keys != expected:
        raise ValueError("sample state identities do not cover the canonical full training split")
    if state_turn_keys != turn_keys:
        raise ValueError("turn output identities disagree with the sample state")
    ledger = runner.validate_failure_ledger(Path(failure_ledger), state, require_present=True)
    failure_keys = set()
    for row in ledger:
        if (
            row.get("n_steps") != 0
            or row.get("terminated_by") == "stop"
            or row.get("status") not in {"done", "failed"}
        ):
            raise ValueError("failure ledger contains a usable or stop trajectory")
        key = (row.get("problem_id"), int(row.get("rollout_idx", -1)))
        if key in failure_keys:
            raise ValueError(f"duplicate failure identity: {key}")
        failure_keys.add(key)
    if turn_keys & failure_keys:
        raise ValueError("turn output and failure ledger overlap")
    if turn_keys | failure_keys != expected:
        missing = sorted(expected - (turn_keys | failure_keys))[:5]
        extra = sorted((turn_keys | failure_keys) - expected)[:5]
        raise ValueError(f"sampling coverage mismatch missing={missing} extra={extra}")
    turn_counts = Counter(problem_id for problem_id, _ in turn_keys)
    failure_counts = Counter(problem_id for problem_id, _ in failure_keys)
    if any(turn_counts[p.problem_id] + failure_counts[p.problem_id] != expected_rollouts for p in problems):
        raise ValueError("sampling coverage is not exactly expected per problem")
    if len(ledger) != len(failure_keys):
        raise ValueError("failure ledger identity count mismatch")
    return {
        "groups_with_turns": len(turn_keys),
        "zero_step_failures": len(failure_keys),
        "total_groups": len(expected),
        "problems": len(problems),
    }
