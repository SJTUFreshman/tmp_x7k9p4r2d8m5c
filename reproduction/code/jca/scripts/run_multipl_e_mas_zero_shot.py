#!/usr/bin/env python3
"""Run zero-shot handoff MAS on MultiPL-E code-completion problems.

Each agent uses the v4 code-continuation contract, but returns a structured MAS
action. Raw candidates are shared between agents, and only the confirmed final
continuation is normalized with the same adapter used by the v4 SAS benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import gzip
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MULTIPL_E_ROOT = REPO_ROOT / "Code" / "MultiPL-E"
sys.path.insert(0, str(REPO_ROOT.parent))
sys.path.insert(0, str(MULTIPL_E_ROOT / "scripts"))

from multipl_e_completion_adapter import (  # noqa: E402
    PROMPT_PROTOCOL,
    build_chat_user_prompt,
    normalize_completion,
)


AGENT_IDS = ("A1", "A2", "A3")
MAS_SCHEMA_VERSION = 3
MAS_PROMPT_PROTOCOL = "multihop_verification_v4_code_continuation_v12_note_progress"
RETRY_POLICY = "identity_aware_once"
FINALIZATION_POLICY = "confirmed_only"
NORMALIZATION_POLICY = "final_once"
ACTION_UNION_NORMALIZATION = "handoff_ignores_confirmed_completion_v1"


@dataclass
class Step:
    turn: int
    active_agent: str
    reasoning: str
    tentative_completion: str
    action: str
    handoff_target: str | None
    handoff_note: str | None
    confirmed_completion: str | None
    raw_output: str
    raw_output_initial: str
    raw_output_retry: str | None
    retry_used: bool
    initial_parse_error: str | None
    retry_parse_error: str | None
    inactive_field_ignored: bool
    ignored_inactive_fields: list[str]


@dataclass
class Trajectory:
    problem_id: str
    steps: list[Step] = field(default_factory=list)
    raw_final_completion: str | None = None
    final_completion: str | None = None
    latest_candidate: str | None = None
    terminated_by: str = "truncated"
    error: str | None = None

    @property
    def active_agents(self) -> list[str]:
        return list(dict.fromkeys(step.active_agent for step in self.steps))

    @property
    def n_handoffs(self) -> int:
        return sum(step.action == "handoff" for step in self.steps)

    @property
    def n_inactive_field_normalizations(self) -> int:
        return sum(step.inactive_field_ignored for step in self.steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot handoff MAS on MultiPL-E test manifests."
    )
    parser.add_argument("--split-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--timings-file", type=Path)
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--max-problems-per-language", type=int)
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--start-agent", choices=AGENT_IDS, default="A1")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8301/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8302/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8303/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument(
        "--json-transport",
        choices=("json_object", "none"),
        default="json_object",
        help="JSON serialization mode; this does not enforce MAS action semantics.",
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--log-raw-chars", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
    return rows


def load_tasks(args: argparse.Namespace) -> list[tuple[str, str, dict[str, Any]]]:
    tasks = []
    seen = set()
    for manifest_arg in args.split_manifest:
        manifest_path = manifest_arg.resolve()
        manifest = load_json(manifest_path)
        root_dataset = manifest["root_dataset"]
        split_root = manifest_path.parent
        for language, details in sorted(manifest["languages"].items()):
            if args.languages and language not in args.languages:
                continue
            rows = read_jsonl((split_root / details["test_file"]).resolve())
            expected = {item["problem_id"] for item in details["test"]}
            if {row.get("name") for row in rows} != expected:
                raise ValueError(f"Manifest mismatch for {root_dataset}/{language}")
            if args.max_problems_per_language:
                rows = rows[: args.max_problems_per_language]
            for row in rows:
                key = (root_dataset, language, row["name"])
                if key in seen:
                    raise ValueError(f"Duplicate task: {key}")
                seen.add(key)
                tasks.append((root_dataset, language, row))
    return tasks


def format_mas_system_prompt(agent_id: str, turn: int, t_max: int) -> str:
    others = [agent for agent in AGENT_IDS if agent != agent_id]
    return f"""You are agent {agent_id}. You collaborate with two other agents: {others[0]} and {others[1]}.

# Task and Shared Conversation
Solve the MultiPL-E code-completion problem in the user message. It contains
the target language, complete source prefix, and evaluator boundary. Previous
assistant messages show other agents' reasoning, candidate code, and handoffs.

# Code-Continuation Rules
- Code fields contain only text appended at the cursor. Do not repeat the
  prefix or include explanations, Markdown, fences, imports, tests, or <think>.
- Continue the target function already started by the source prefix. Do not
  repeat its declaration.
- Complete the function, including returns and every nested delimiter you open.
- Obey the evaluator boundary. Do not emit the final function delimiter when
  the user says the evaluator supplies it.

# Collaboration Rules
- The latest non-empty tentative_completion is the current candidate.
- On the first turn, provide a non-empty candidate and hand off to {others[0]}
  or {others[1]}. Do not confirm_stop.
- When a handoff_note is present, address it. Reason independently about the
  original task and candidate, including behavior, types, edge cases, syntax,
  delimiters, and evaluator boundary.
- You may use confirm_stop only when confirmed_completion is character-for-
  character identical to the latest non-empty tentative_completion proposed by
  a previous agent. If you change any character, use handoff instead.
- If the candidate is wrong, put the complete corrected code in
  tentative_completion and hand off for another audit. Set
  confirmed_completion to null.
- Confirm only a candidate from a previous, different agent. Never hand off to
  yourself. Keep reasoning concise and make each handoff_note problem-specific.

# Output Format
Return exactly one JSON object with these six keys and no surrounding prose or
Markdown:
- reasoning: a brief string.
- tentative_completion: a code string, or "" when confirming.
- action: exactly "handoff" or "confirm_stop".
- handoff_target: {others[0]} or {others[1]} for handoff; otherwise null.
- handoff_note: a specific verification request for handoff; otherwise null.
- confirmed_completion: null for handoff; for confirm_stop it MUST exactly
  equal the latest non-empty tentative_completion from a previous agent.

Collaboration turn: {turn + 1} of {t_max}.
Turns remaining after this response: {max(0, t_max - turn - 1)}.
""".strip()


def build_user_prompt(language: str, row: dict[str, Any]) -> str:
    base = build_chat_user_prompt(
        language, row["prompt"], row.get("tests", ""), row.get("stop_tokens") or []
    )
    return (
        "This is a shared MAS turn. Inspect prior agents' candidate code in the "
        "conversation, then return the required protocol JSON.\n\n" + base
    )


def load_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n|\n```$", "", text.strip(), flags=re.DOTALL)
    try:
        value, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_action(
    raw: str, active_agent: str, prior_tentative: bool
) -> tuple[dict[str, Any] | None, str | None]:
    payload = load_json_object(raw)
    if payload is None:
        return None, "malformed_or_missing_json_object"
    reasoning = payload.get("reasoning", "")
    tentative = payload.get("tentative_completion", "")
    action = payload.get("action")
    target = payload.get("handoff_target")
    note = payload.get("handoff_note")
    confirmed = payload.get("confirmed_completion")
    if not all(isinstance(value, str) for value in (reasoning, tentative)):
        return None, "reasoning_or_tentative_completion_must_be_strings"
    reasoning = reasoning.strip()
    if not tentative.strip():
        tentative = ""
    if action == "handoff":
        if target not in AGENT_IDS:
            return None, "handoff_target_missing_or_invalid"
        if target == active_agent:
            return None, "handoff_target_is_active_agent"
        if note is not None and not isinstance(note, str):
            return None, "handoff_note_must_be_string_or_null"
        ignored = ["confirmed_completion"] if confirmed not in (None, "") else []
        return ({"reasoning": reasoning, "tentative": tentative, "action": action,
                 "target": target, "note": str(note or "").strip() or None,
                 "confirmed": None, "ignored_inactive_fields": ignored}, None)
    if action == "confirm_stop":
        if target not in (None, ""):
            return None, "confirm_stop_requires_null_handoff_target"
        if note not in (None, ""):
            return None, "confirm_stop_requires_null_handoff_note"
        if not isinstance(confirmed, str) or not confirmed.strip():
            return None, "confirm_stop_requires_nonempty_confirmed_completion"
        return ({"reasoning": reasoning, "tentative": tentative, "action": action,
                 "target": None, "note": None, "confirmed": confirmed,
                 "ignored_inactive_fields": []}, None)
    return None, "action_missing_or_invalid"


def parse_action(
    raw: str, active_agent: str, prior_tentative: bool
) -> dict[str, Any] | None:
    parsed, _ = validate_action(raw, active_agent, prior_tentative)
    return parsed


def retry_instruction(active_agent: str) -> str:
    others = [agent for agent in AGENT_IDS if agent != active_agent]
    return (
        "Your previous output was invalid. Return only the MAS JSON object with "
        '"reasoning", "tentative_completion", "action", "handoff_target", '
        '"handoff_note", and "confirmed_completion". You are agent '
        f'{active_agent}. For handoff, handoff_target must be "{others[0]}" or '
        f'"{others[1]}", never "{active_agent}". Use action="handoff" unless '
        "you are confirming a previous tentative_completion."
    )


def render_shared_message(agent: str, parsed: dict[str, Any]) -> str:
    parts = [f"[{agent}]", parsed["reasoning"]]
    if parsed["tentative"]:
        parts.append("tentative_completion:\n" + parsed["tentative"])
    if parsed["action"] == "handoff":
        line = f"→ handoff to {parsed['target']}"
        if parsed["note"]:
            line += f": {parsed['note']}"
        parts.append(line)
    else:
        parts.append("→ confirm_stop")
    return "\n".join(part for part in parts if part)


def run_trajectory(task: tuple[str, str, dict[str, Any]], callers: dict[str, Any], args: argparse.Namespace) -> tuple[Trajectory, dict[str, Any]]:
    root_dataset, language, row = task
    trajectory = Trajectory(problem_id=row["name"])
    messages = [
        {
            "role": "system",
            "content": format_mas_system_prompt(args.start_agent, 0, args.t_max),
        },
        {"role": "user", "content": build_user_prompt(language, row)},
    ]
    current = args.start_agent
    prior_tentative = False
    try:
        for turn in range(args.t_max):
            raw_initial = callers[current](list(messages))
            parsed, initial_parse_error = validate_action(
                raw_initial, current, prior_tentative
            )
            raw_retry = None
            retry_parse_error = None
            if parsed is None:
                retry_messages = list(messages) + [
                    {"role": "user", "content": retry_instruction(current)}
                ]
                raw_retry = callers[current](retry_messages)
                parsed, retry_parse_error = validate_action(
                    raw_retry, current, prior_tentative
                )
            raw = raw_retry if raw_retry is not None else raw_initial
            if parsed is None:
                trajectory.terminated_by = "exception"
                trajectory.error = (
                    f"Invalid MAS action from {current}: {retry_parse_error or initial_parse_error}"
                )
                trajectory.steps.append(
                    Step(
                        turn, current, "", "", "invalid", None, None, None,
                        raw, raw_initial, raw_retry, raw_retry is not None,
                        initial_parse_error, retry_parse_error, False, [],
                    )
                )
                break

            if parsed["tentative"]:
                trajectory.latest_candidate = parsed["tentative"]
            target = parsed["target"] if parsed["action"] == "handoff" else None
            step = Step(
                turn, current, parsed["reasoning"], parsed["tentative"],
                parsed["action"], target, parsed["note"] if target else None,
                parsed["confirmed"], raw, raw_initial, raw_retry,
                raw_retry is not None, initial_parse_error, retry_parse_error,
                bool(parsed["ignored_inactive_fields"]),
                list(parsed["ignored_inactive_fields"]),
            )
            trajectory.steps.append(step)
            if step.action == "confirm_stop":
                trajectory.raw_final_completion = step.confirmed_completion
                trajectory.latest_candidate = step.confirmed_completion
                trajectory.terminated_by = "stop"
                break
            prior_tentative = prior_tentative or bool(parsed["tentative"])
            messages.append(
                {"role": "assistant", "content": render_shared_message(current, parsed)}
            )
            current = target
            if turn + 1 < args.t_max:
                messages[0] = {
                    "role": "system",
                    "content": format_mas_system_prompt(
                        current, turn + 1, args.t_max
                    ),
                }
        else:
            trajectory.terminated_by = "truncated"
    except Exception as exc:  # preserve failed trajectories for diagnosis
        trajectory.terminated_by = "exception"
        trajectory.error = str(exc)

    if trajectory.terminated_by == "stop" and trajectory.raw_final_completion:
        try:
            trajectory.final_completion = normalize_completion(
                trajectory.raw_final_completion,
                row["prompt"],
                row.get("tests", ""),
                row.get("stop_tokens") or [],
                language,
            )
        except Exception as exc:
            trajectory.terminated_by = "exception"
            trajectory.error = f"Final completion normalization failed: {exc}"
            trajectory.final_completion = None

    completion = trajectory.final_completion
    data = dict(row)
    data.update({
        "completions": [completion or ""],
        "mas_protocol": "v4_code_continuation_handoff",
        "mas_schema_version": MAS_SCHEMA_VERSION,
        "mas_prompt_protocol": MAS_PROMPT_PROTOCOL,
        "prompt_protocol": PROMPT_PROTOCOL,
        "completion_adapter": "qwen_code_continuation_v4",
        "json_transport": args.json_transport,
        "semantic_schema_enforcement": "disabled",
        "action_union_normalization": ACTION_UNION_NORMALIZATION,
        "retry_policy": RETRY_POLICY,
        "finalization_policy": FINALIZATION_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "agents": list(AGENT_IDS),
        "terminated_by": trajectory.terminated_by,
    })
    return trajectory, {"root_dataset": root_dataset, "language": language, "data": data}


def write_gzip(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    temporary.replace(path)


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def one_line(value: str) -> str:
    return " ".join(value.split())


def print_trajectory(trajectory: Trajectory, raw_chars: int) -> None:
    for step in trajectory.steps:
        print(f"  turn={step.turn} agent={step.active_agent} action={step.action}")
        if step.reasoning:
            print(f"    reasoning={one_line(step.reasoning)}")
        if step.tentative_completion:
            print(f"    tentative_completion={one_line(step.tentative_completion)}")
        if step.handoff_target:
            note = f" note={one_line(step.handoff_note or '')}" if step.handoff_note else ""
            print(f"    handoff={step.handoff_target}{note}")
        if step.confirmed_completion:
            print(f"    confirmed_completion={one_line(step.confirmed_completion)}")
        if step.retry_used:
            print(
                "    retry_used=true "
                f"initial_parse_error={step.initial_parse_error} "
                f"retry_parse_error={step.retry_parse_error}"
            )
        if step.inactive_field_ignored:
            print(
                "    inactive_field_ignored=true fields="
                + ",".join(step.ignored_inactive_fields)
            )
        if raw_chars > 0 and step.raw_output_initial:
            print(f"    raw_initial={one_line(step.raw_output_initial[:raw_chars])}")
        if raw_chars > 0 and step.raw_output_retry:
            print(f"    raw_retry={one_line(step.raw_output_retry[:raw_chars])}")


def main() -> int:
    args = parse_args()
    if args.t_max <= 0 or args.max_new_tokens <= 0 or args.max_concurrency <= 0:
        raise SystemExit("--t-max, --max-new-tokens, and --max-concurrency must be positive")
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No MultiPL-E tasks selected")
    print(f"MAS tasks={len(tasks)} languages={len(set((r, l) for r, l, _ in tasks))}")
    print(
        f"mas_prompt_protocol={MAS_PROMPT_PROTOCOL} "
        f"completion_prompt_protocol={PROMPT_PROTOCOL} "
        f"json_transport={args.json_transport} "
        f"action_union_normalization={ACTION_UNION_NORMALIZATION} "
        f"t_max={args.t_max} temperature={args.temperature}"
    )
    print(f"trajectory_output={args.trajectory_output}")
    if args.dry_run:
        root, language, row = tasks[0]
        print(f"dry-run first={root}/{language}/{row['name']}")
        return 0

    from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller

    generation = GenerationOptions(args.max_new_tokens, args.temperature, args.top_p, args.enable_thinking)
    response_format = (
        {"type": "json_object"} if args.json_transport == "json_object" else None
    )
    callers = {
        agent: OpenAIChatLLMCaller(
            url,
            model,
            generation=generation,
            timeout=args.api_timeout,
            response_format=response_format,
        )
        for agent, url, model in (
            ("A1", args.api_base_a1, args.api_model_a1),
            ("A2", args.api_base_a2, args.api_model_a2),
            ("A3", args.api_base_a3, args.api_model_a3),
        )
    }
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.timings_file is not None:
        args.timings_file = args.timings_file.resolve()
    started = time.monotonic()
    with args.trajectory_output.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(tasks))) as pool:
            futures = {pool.submit(run_trajectory, task, callers, args): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                trajectory, result = future.result()
                root, language, row = task
                output = args.output_dir / root / language / f"{row['name']}.json.gz"
                write_gzip(output, result["data"])
                completed_at = time.time()
                elapsed = time.monotonic() - started
                append_jsonl(args.timings_file, {
                    "event": "generation_completed",
                    "root_dataset": root,
                    "language": language,
                    "problem_id": row["name"],
                    "completion_file": str(output),
                    "completed_at_unix": round(completed_at, 3),
                    "total_elapsed_seconds": round(elapsed, 3),
                    "status": trajectory.terminated_by,
                    "active_agents": trajectory.active_agents,
                    "n_handoffs": trajectory.n_handoffs,
                    "n_inactive_field_normalizations": (
                        trajectory.n_inactive_field_normalizations
                    ),
                })
                record = {
                    "schema_version": MAS_SCHEMA_VERSION,
                    "root_dataset": root,
                    "language": language,
                    "problem_id": row["name"],
                    "trajectory": asdict(trajectory),
                    "final_completion": trajectory.final_completion,
                    "terminated_by": trajectory.terminated_by,
                "n_handoffs": trajectory.n_handoffs,
                "n_inactive_field_normalizations": (
                    trajectory.n_inactive_field_normalizations
                ),
                    "active_agents": trajectory.active_agents,
                    "sampling": {
                        "mas_prompt_protocol": MAS_PROMPT_PROTOCOL,
                        "prompt_protocol": PROMPT_PROTOCOL,
                        "json_transport": args.json_transport,
                        "semantic_schema_enforcement": "disabled",
                        "action_union_normalization": ACTION_UNION_NORMALIZATION,
                        "retry_policy": RETRY_POLICY,
                        "finalization_policy": FINALIZATION_POLICY,
                        "normalization_policy": NORMALIZATION_POLICY,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "t_max": args.t_max,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(tasks)}] {root}/{language}/{row['name']} terminated={trajectory.terminated_by} agents={trajectory.active_agents} handoffs={trajectory.n_handoffs}", flush=True)
                if trajectory.error:
                    print(f"  error={trajectory.error}")
                print_trajectory(trajectory, args.log_raw_chars)
    print(f"MAS generation finished in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
