#!/usr/bin/env python3
"""Build role-aligned MATH SFT data from controlled model rollouts."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


BUNDLED_PACKAGE_PARENT = Path(__file__).resolve().parent / "vendor"
if str(BUNDLED_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(BUNDLED_PACKAGE_PARENT))

from jca.gsm.scripts import run_mas as protocol  # noqa: E402
from jca.src.math_eval import math_answers_equivalent, render_math_mas_system_prompt  # noqa: E402

from collect_math_correction_sft_rollouts import (  # noqa: E402
    COLLECTOR_VERSION,
    CORRECTION_TERMS,
    ERROR_TYPES,
    ROUTES,
)


BUILDER_VERSION = "math_specific_fixed_a1_rollout_sft_v2"
AGENTS = ("A1", "A2", "A3")
ROUTE_SIGNATURES = tuple(">".join(route) for route in ROUTES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-agent", type=int, default=2200)
    parser.add_argument("--correction-fraction", type=float, default=0.30)
    parser.add_argument("--balance-seed", type=int, default=42)
    parser.add_argument("--max-verifier-reasoning-similarity", type=float, default=0.85)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if args.samples_per_agent <= 0 or args.samples_per_agent % 40:
        parser.error("--samples-per-agent must be a positive multiple of 40")
    if not math.isclose(args.correction_fraction, 0.30, abs_tol=1e-12):
        parser.error("the fixed-A1 role matrix requires --correction-fraction=0.30")
    if not 0.0 <= args.max_verifier_reasoning_similarity <= 1.0:
        parser.error("--max-verifier-reasoning-similarity must be in [0, 1]")
    return args


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


def iter_records(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank rollout line at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object rollout at {path}:{line_number}")
                yield path, line_number, value


def clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def parse_raw_object(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalized_similarity(left: str, right: str) -> float:
    left_normalized = " ".join(left.lower().split())
    right_normalized = " ".join(right.lower().split())
    if not left_normalized or not right_normalized:
        return 0.0
    return difflib.SequenceMatcher(None, left_normalized, right_normalized).ratio()


def label_from_step(step: dict[str, Any]) -> str:
    action = str(step["action"])
    payload = {
        "reasoning": str(step["reasoning"]).strip(),
        "tentative_answer": str(step["tentative_answer"]).strip(),
        "action": action,
        "handoff_target": clean_optional(step.get("handoff_target")),
        "handoff_note": clean_optional(step.get("handoff_note")),
        "confirmed_answer": (
            clean_optional(step.get("confirmed_answer"))
            if action == "confirm_stop"
            else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def explicit_correction(reasoning: str, error_type: str) -> bool:
    normalized = " ".join(reasoning.lower().split())
    generic = ("error", "mistake", "incorrect", "wrong", "corrected")
    return any(term in normalized for term in generic) and any(
        term in normalized for term in CORRECTION_TERMS[error_type]
    )


def parse_candidate(
    source_path: Path,
    line_number: int,
    record: dict[str, Any],
    *,
    max_similarity: float,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if record.get("collector_version") != COLLECTOR_VERSION:
        return None, "stale_collector"
    if record.get("dataset") != "MATH" or record.get("source_split") != "train":
        return None, "wrong_dataset"
    if record.get("terminated_by") != "stop" or float(record.get("em", 0.0)) != 1.0:
        return None, "not_accepted"
    problem = record.get("problem") or {}
    trajectory = record.get("trajectory") or {}
    kind = str(trajectory.get("trajectory_kind") or "")
    if kind not in {"regular", "correction"}:
        return None, "bad_kind"
    error_type = trajectory.get("error_type")
    if kind == "regular" and error_type is not None:
        return None, "regular_has_error"
    if kind == "correction" and error_type not in ERROR_TYPES:
        return None, "bad_error_type"
    if kind == "correction" and trajectory.get("error_problem_eligible") is not True:
        return None, "ineligible_error_problem"
    route = tuple(trajectory.get("planned_route") or ())
    if route not in ROUTES or trajectory.get("route_signature") != ">".join(route):
        return None, "bad_route"
    steps = trajectory.get("steps") or []
    if len(steps) != 3:
        return None, "wrong_turn_count"
    if [step.get("turn") for step in steps] != [0, 1, 2]:
        return None, "noncontiguous_turns"
    if [step.get("active_agent") for step in steps] != list(route):
        return None, "route_mismatch"
    if [step.get("action") for step in steps] != ["handoff", "handoff", "confirm_stop"]:
        return None, "action_mismatch"
    if [step.get("handoff_target") for step in steps[:2]] != list(route[1:]):
        return None, "handoff_target_mismatch"
    if steps[2].get("handoff_target") is not None:
        return None, "final_handoff_target"
    gold = clean_optional(problem.get("answer"))
    question = clean_optional(problem.get("question"))
    problem_id = clean_optional(problem.get("id"))
    if not gold or not question or not problem_id:
        return None, "missing_problem_fields"
    if not math_answers_equivalent(record.get("final_answer") or "", gold):
        return None, "wrong_final_answer"

    prior_reasonings: list[str] = []
    conversation: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "# Mathematics Problem\n"
                f"{question}\n\n"
                "Solve carefully. Your final answer must be a short mathematical expression "
                "or value, not a sentence."
            ),
        }
    ]
    samples: dict[int, dict[str, Any]] = {}
    for position, step in enumerate(steps):
        if parse_raw_object(step.get("raw_output")) is None:
            return None, "invalid_raw_json"
        reasoning = clean_optional(step.get("reasoning"))
        answer = clean_optional(step.get("tentative_answer"))
        if not reasoning or not answer:
            return None, "empty_step"
        is_wrong_context = kind == "correction" and position == 0
        if is_wrong_context and math_answers_equivalent(answer, gold):
            return None, "correction_context_is_correct"
        if not is_wrong_context and not math_answers_equivalent(answer, gold):
            return None, "wrong_supervised_answer"
        if position == 2 and not math_answers_equivalent(
            clean_optional(step.get("confirmed_answer")) or "", gold
        ):
            return None, "wrong_confirmed_answer"
        if position > 0 and any(
            normalized_similarity(reasoning, previous) >= max_similarity
            for previous in prior_reasonings
        ):
            return None, "nonindependent_reasoning"
        if kind == "correction" and position == 1 and not explicit_correction(
            reasoning, str(error_type)
        ):
            return None, "implicit_correction"
        if not is_wrong_context:
            label = label_from_step(step)
            samples[position] = {
                "builder_version": BUILDER_VERSION,
                "supervision_source": "controlled_model_rollout",
                "dataset": "MATH",
                "source_split": "train",
                "agent_id": step["active_agent"],
                "problem_id": problem_id,
                "subject": str(problem.get("subject") or ""),
                "level": str(problem.get("level") or ""),
                "trajectory_id": f"{source_path.resolve()}:{line_number}",
                "plan_index": int(record.get("plan_index", -1)),
                "trajectory_kind": kind,
                "error_type": error_type,
                "route_signature": ">".join(route),
                "turn": position,
                "route_position": position,
                "protocol_mode": "fixed_a1_3_turn",
                "min_agents_before_stop": 3,
                "sample_type": f"rollout_{step['action']}",
                "gold_answer": gold,
                "messages": [
                    {
                        "role": "system",
                        "content": render_math_mas_system_prompt(
                            step["active_agent"], min_agents_before_stop=3
                        ),
                    },
                    *[dict(message) for message in conversation],
                ],
                "label": label,
            }
        conversation.append(
            {
                "role": "assistant",
                "content": protocol.render_assistant_message(
                    protocol.GSMStep(
                        turn=position,
                        active_agent=str(step["active_agent"]),
                        reasoning=reasoning,
                        tentative_answer=answer,
                        action=str(step["action"]),
                        handoff_target=clean_optional(step.get("handoff_target")),
                        handoff_note=clean_optional(step.get("handoff_note")),
                        confirmed_answer=clean_optional(step.get("confirmed_answer")),
                        raw_output=str(step.get("raw_output") or ""),
                    )
                ),
            }
        )
        prior_reasonings.append(reasoning)
    expected_positions = {1, 2} if kind == "correction" else {0, 1, 2}
    if set(samples) != expected_positions:
        return None, "incomplete_supervision"
    return {
        "kind": kind,
        "error_type": error_type,
        "route": ">".join(route),
        "samples": samples,
        "sort_key": f"{int(record.get('plan_index', -1)):09d}:{problem_id}",
    }, None


def shuffled(values: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = sorted(values, key=lambda value: value["sort_key"])
    random.Random(seed).shuffle(result)
    return result


def select_rows(
    candidates: list[dict[str, Any]],
    *,
    samples_per_agent: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    regular: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corrections: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["kind"] == "regular":
            regular[candidate["route"]].append(candidate)
        else:
            corrections[(candidate["route"], candidate["error_type"])].append(candidate)
    for index, route in enumerate(ROUTE_SIGNATURES):
        regular[route] = shuffled(regular[route], seed + index)
        for error_index, error_type in enumerate(ERROR_TYPES):
            corrections[(route, error_type)] = shuffled(
                corrections[(route, error_type)], seed + 10 + index * 10 + error_index
            )

    correction_per_agent = round(samples_per_agent * 0.30)
    regular_per_verifier = samples_per_agent - correction_per_agent
    a1_per_route = samples_per_agent // 2
    verifier_regular_per_route = regular_per_verifier // 2
    correction_per_cell = correction_per_agent // len(ROUTES) // len(ERROR_TYPES)
    deficits: dict[str, dict[str, int]] = {}
    for route in ROUTE_SIGNATURES:
        if len(regular[route]) < a1_per_route:
            deficits[f"regular:{route}"] = {
                "available": len(regular[route]),
                "required": a1_per_route,
            }
        for error_type in ERROR_TYPES:
            available = len(corrections[(route, error_type)])
            if available < correction_per_cell:
                deficits[f"correction:{route}:{error_type}"] = {
                    "available": available,
                    "required": correction_per_cell,
                }
    if deficits:
        raise ValueError(f"insufficient accepted rollout cells: {deficits}")

    a1_regular: list[dict[str, Any]] = []
    verifier_regular: list[dict[str, Any]] = []
    selected_corrections: list[dict[str, Any]] = []
    for route in ROUTE_SIGNATURES:
        chosen = regular[route][:a1_per_route]
        a1_regular.extend(chosen)
        verifier_regular.extend(chosen[:verifier_regular_per_route])
        for error_type in ERROR_TYPES:
            selected_corrections.extend(
                corrections[(route, error_type)][:correction_per_cell]
            )

    per_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENTS}
    per_agent["A1"] = [candidate["samples"][0] for candidate in a1_regular]
    for candidate in [*verifier_regular, *selected_corrections]:
        for position in (1, 2):
            sample = candidate["samples"][position]
            per_agent[sample["agent_id"]].append(sample)
    for index, agent in enumerate(AGENTS):
        random.Random(seed + 100 + index).shuffle(per_agent[agent])

    selection = {
        "a1_regular_trajectories": len(a1_regular),
        "verifier_regular_trajectories": len(verifier_regular),
        "correction_trajectories": len(selected_corrections),
        "correction_per_route_error_cell": correction_per_cell,
    }
    return per_agent, selection


def quality_gate(
    per_agent: dict[str, list[dict[str, Any]]], samples_per_agent: int
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(passed), **details}

    counts = {agent: len(per_agent[agent]) for agent in AGENTS}
    add(
        "exact_samples_per_agent",
        all(count == samples_per_agent for count in counts.values()),
        counts=counts,
    )
    a1 = per_agent["A1"]
    add(
        "A1_correct_first_handoff_only",
        all(
            row["trajectory_kind"] == "regular"
            and row["turn"] == 0
            and json.loads(row["label"])["action"] == "handoff"
            for row in a1
        ),
    )
    add(
        "A1_route_balance",
        Counter(row["route_signature"] for row in a1)
        == Counter({route: samples_per_agent // 2 for route in ROUTE_SIGNATURES}),
    )
    correction_target = round(samples_per_agent * 0.30)
    for agent in ("A2", "A3"):
        rows = per_agent[agent]
        corrections = [row for row in rows if row["trajectory_kind"] == "correction"]
        add(
            f"{agent}_correction_fraction",
            len(corrections) == correction_target,
            count=len(corrections),
            target=correction_target,
        )
        add(
            f"{agent}_route_balance",
            Counter(row["route_signature"] for row in rows)
            == Counter({route: samples_per_agent // 2 for route in ROUTE_SIGNATURES}),
        )
        add(
            f"{agent}_action_balance",
            Counter(json.loads(row["label"])["action"] for row in rows)
            == Counter(
                {"handoff": samples_per_agent // 2, "confirm_stop": samples_per_agent // 2}
            ),
        )
        add(
            f"{agent}_error_balance",
            Counter(row["error_type"] for row in corrections)
            == Counter(
                {error_type: correction_target // len(ERROR_TYPES) for error_type in ERROR_TYPES}
            ),
        )
    add(
        "controlled_rollout_only",
        all(
            row.get("supervision_source") == "controlled_model_rollout"
            and not row["sample_type"].startswith("official_solution_")
            for rows in per_agent.values()
            for row in rows
        ),
    )
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def output_payloads(
    per_agent: dict[str, list[dict[str, Any]]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    all_rows = [row for agent in AGENTS for row in per_agent[agent]]
    random.Random(seed).shuffle(all_rows)
    return {"all_agents.jsonl": all_rows, **{f"{a}.jsonl": per_agent[a] for a in AGENTS}}


def encoded_rows(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    for path in args.input:
        if not path.is_file():
            raise SystemExit(f"rollout input missing: {path}")
    parse_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_plans: set[int] = set()
    for path, line_number, record in iter_records(args.input):
        parse_counts["input_trajectories"] += 1
        plan_index = int(record.get("plan_index", -1))
        if plan_index < 0 or plan_index in seen_plans:
            raise ValueError(f"invalid or duplicate plan_index at {path}:{line_number}")
        seen_plans.add(plan_index)
        candidate, reason = parse_candidate(
            path,
            line_number,
            record,
            max_similarity=args.max_verifier_reasoning_similarity,
        )
        if candidate is None:
            parse_counts[f"drop_{reason}"] += 1
        else:
            candidates.append(candidate)
            parse_counts[f"valid_{candidate['kind']}"] += 1
    per_agent, selection = select_rows(
        candidates,
        samples_per_agent=args.samples_per_agent,
        seed=args.balance_seed,
    )
    gate = quality_gate(per_agent, args.samples_per_agent)
    if not gate["passed"]:
        failed = [name for name, check in gate["checks"].items() if not check["passed"]]
        raise SystemExit("MATH rollout SFT quality gate failed: " + ", ".join(failed))
    payloads = output_payloads(per_agent, args.balance_seed)
    summary = {
        "builder_version": BUILDER_VERSION,
        "supervision_source": "controlled_model_rollout",
        "dataset": "MATH",
        "source_split": "train",
        "inputs": [file_meta(path) for path in args.input],
        "counts_by_agent": {agent: len(per_agent[agent]) for agent in AGENTS},
        "selection": selection,
        "parse_counts": dict(parse_counts),
        "quality_gate": gate,
        "targets": {
            "samples_per_agent": args.samples_per_agent,
            "correction_fraction_A2_A3": 0.30,
            "fixed_routes": list(ROUTE_SIGNATURES),
            "route_fraction": 0.50,
            "confirm_fraction_A2_A3": 0.50,
            "error_types": list(ERROR_TYPES),
        },
        "wrong_A1_context_policy": "context_only_never_label",
        "max_verifier_reasoning_similarity": args.max_verifier_reasoning_similarity,
        "output_sha256": {
            filename: hashlib.sha256(encoded_rows(rows)).hexdigest()
            for filename, rows in payloads.items()
        },
    }
    stats_payload = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.validate_existing:
        for filename, rows in payloads.items():
            path = args.out_dir / filename
            if not path.is_file() or sha256(path) != hashlib.sha256(encoded_rows(rows)).hexdigest():
                raise SystemExit(f"existing SFT output mismatch: {path}")
        stats_path = args.out_dir / "stats.json"
        if not stats_path.is_file() or stats_path.read_bytes() != stats_payload:
            raise SystemExit("existing SFT stats mismatch")
        print(json.dumps({"status": "valid", "counts": summary["counts_by_agent"]}))
        return
    outputs = [args.out_dir / filename for filename in payloads]
    outputs.append(args.out_dir / "stats.json")
    if any(path.exists() for path in outputs):
        raise SystemExit("SFT output exists; validate it or choose a new TAG")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in payloads.items():
        atomic_write(args.out_dir / filename, encoded_rows(rows))
    atomic_write(args.out_dir / "stats.json", stats_payload)
    print(
        json.dumps(
            {
                "status": "built",
                "valid_trajectories": len(candidates),
                "counts": summary["counts_by_agent"],
                "quality_gate": "PASS",
            }
        )
    )


if __name__ == "__main__":
    main()
