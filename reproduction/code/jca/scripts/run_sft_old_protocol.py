"""Run SFT adapters with the old prompt/schema used to build v1 SFT data.

This runner is intentionally separate from scripts/run_zero_shot.py because the
SFT v1 adapters were trained on a different action schema:

  {
    "reasoning": "...",
    "tentative_answer": "...",
    "action": "handoff" | "confirm_stop",
    "handoff_target": "A2" | null,
    "handoff_note": "..." | null,
    "confirmed_answer": "\\boxed{...}" | null
  }

Use it against vLLM OpenAI-compatible LoRA servers exposing model names A1/A2/A3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.src.agents import AGENT_IDS  # noqa: E402
from jca.src.data import MuSiQueProblem, format_problem_as_prompt, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1, is_correct  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402


OLD_SFT_SYSTEM_PROMPT_TEMPLATE = """You are agent {AGENT_ID}. You collaborate with two other agents: {OTHER_AGENTS}.

# The Task
You are answering a multi-hop question. The question requires combining
information from multiple paragraphs. All paragraphs and the question are
shown in the user message.

# Shared Conversation
You and the other agents share this conversation. Each previous assistant
message is prefixed with the agent ID who wrote it (e.g. "[A1] ..."). You
can read what previous agents have done.

# Collaborative Verification Protocol
This system uses a verification protocol: at least two agents must agree on
the answer before it can be finalized. Concretely:

  1. The first agent to address the question contributes a "tentative_answer"
     and HANDS OFF to another agent for verification. It cannot finalize.
  2. The second (or later) agent reads the tentative answer, verifies it
     against the paragraphs, and either:
       - Confirms it (action = "confirm_stop") with the same answer, or
       - Disagrees: provides a corrected tentative_answer and hands off again.
  3. confirm_stop is only valid once at least one previous agent has
     produced a tentative_answer in the conversation.

# Your Action Each Turn
At each turn, you must produce exactly this JSON object:

  {{
    "reasoning": "<your reasoning step, citing paragraph numbers>",
    "tentative_answer": "<your best answer string, or empty string if you have none yet>",
    "action": "handoff" | "confirm_stop",
    "handoff_target": "<one of {OTHER_AGENTS}>" | null,
    "handoff_note": "<concise question/instruction for the next agent>" | null,
    "confirmed_answer": "<final answer wrapped in \\\\boxed{{...}}>" | null
  }}

# Field Rules

  - "reasoning" should be 1-4 sentences citing the paragraph numbers you used.
  - "tentative_answer" is your best guess at the answer in plain text (no
    \\\\boxed{{}}). May be empty string if you genuinely cannot guess.
  - "action" determines what happens after this turn:

      action = "handoff":
        - handoff_target must be one of {OTHER_AGENTS}, not yourself.
        - handoff_note must be a concise instruction (<= 30 words),
          e.g. "Please verify that the founder of Orion is Mike Medavoy
          using paragraph 7."
        - confirmed_answer must be null.

      action = "confirm_stop":
        - confirmed_answer must be a non-empty string wrapping the final
          answer in \\\\boxed{{...}}, e.g. "The answer is \\\\boxed{{1995}}."
        - handoff_target and handoff_note must be null.
        - LEGAL ONLY IF at least one previous agent in this conversation
          has produced a non-empty tentative_answer. If no prior tentative
          answer exists, you MUST use action = "handoff" instead.

# Strategy

  - When you are the FIRST to engage:
      Read the paragraphs, write your reasoning, set tentative_answer to
      your best guess (even if uncertain), and HAND OFF with a clear
      verification request. You are forbidden from confirm_stop on this turn.

  - When a previous agent gave a tentative_answer:
      Verify it. If you agree: action = "confirm_stop", confirmed_answer =
      "\\\\boxed{{their answer}}". If you disagree: provide your own
      tentative_answer + reasoning, hand off again.

  - When the conversation is long and getting circular:
      Pick the most-supported tentative_answer from history and confirm it
      rather than starting another verification round.

  - "Final answer" = a concise number, entity name, or short phrase.

You should NOT:
  - Repeat verbatim what a previous agent has already concluded.
  - Hand off to yourself.
  - Use confirm_stop on the first turn (no prior tentative_answer to confirm).
  - Output the answer in tentative_answer wrapped in \\\\boxed{{}} (only
    confirmed_answer uses \\\\boxed{{}}).

Output ONLY the JSON object. No prose around it."""


@dataclass
class OldParsedAction:
    reasoning: str = ""
    tentative_answer: str = ""
    action: Optional[str] = None
    handoff_target: Optional[str] = None
    handoff_note: Optional[str] = None
    final_answer: Optional[str] = None


@dataclass
class OldTrajectoryStep:
    turn: int
    active_agent: str
    reasoning: str
    tentative_answer: str
    action: Optional[str]
    handoff_target: Optional[str]
    handoff_note: Optional[str]
    final_answer: Optional[str]
    raw_output: str
    raw_outputs: List[str] = field(default_factory=list)


@dataclass
class OldTrajectory:
    problem_id: str
    steps: List[OldTrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated_by: str = "running"
    error: Optional[str] = None

    @property
    def active_agents(self) -> List[str]:
        agents: List[str] = []
        for step in self.steps:
            if step.active_agent not in agents:
                agents.append(step.active_agent)
        return agents

    @property
    def n_handoffs(self) -> int:
        return sum(1 for step in self.steps if step.handoff_target is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v1 SFT adapters with the old SFT action protocol."
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation"])
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument("--start-agent", default="A1", choices=AGENT_IDS)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-raw-chars", type=int, default=1200)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of independent MAS trajectories evaluated concurrently.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.max_concurrency <= 0:
        raise SystemExit("--limit and --max-concurrency must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")

    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    output_path = args.output or (
        Path("outputs")
        / "sft_old_protocol"
        / f"{args.split}_start{args.start}_n{len(selected)}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    callers = {
        "A1": OpenAIChatLLMCaller(
            args.api_base_a1,
            args.api_model_a1,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
        "A2": OpenAIChatLLMCaller(
            args.api_base_a2,
            args.api_model_a2,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
        "A3": OpenAIChatLLMCaller(
            args.api_base_a3,
            args.api_model_a3,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
    }

    print("SFT old-protocol MuSiQue inference")
    print(f"  split: {args.split}")
    print(f"  selected: start={args.start}, n={len(selected)}")
    print(f"  t_max: {args.t_max}")
    print(f"  start_agent: {args.start_agent}")
    print(f"  max_concurrency: {args.max_concurrency}")
    print(f"  output: {output_path}")
    for agent_id, caller in callers.items():
        print(f"  {agent_id} api: {caller.base_url} model={caller.model_name}")

    if args.dry_run:
        first = selected[0]
        print("\nDry run only. First selected problem:")
        print(f"  id: {first.id}")
        print(f"  question: {first.question}")
        print(f"  answer: {first.answer}")
        return

    def evaluate_problem(problem: MuSiQueProblem):
        item_started = time.monotonic()
        traj = run_old_protocol_trajectory(
            problem,
            callers,
            t_max=args.t_max,
            start_agent=args.start_agent,
        )
        em, f1 = compute_em_f1(traj.final_answer or "", problem)
        record = {
            "problem": {
                "id": problem.id,
                "question": problem.question,
                "answer": problem.answer,
                "answer_aliases": problem.answer_aliases,
                "hop": problem.hop,
            },
            "trajectory": old_trajectory_to_dict(traj),
            "final_answer": traj.final_answer,
            "terminated_by": traj.terminated_by,
            "correct": is_correct(traj.final_answer or "", problem),
            "em": em,
            "f1": f1,
        }
        return problem, traj, record, time.monotonic() - item_started

    records: List[Dict[str, Any]] = []
    run_started = time.monotonic()
    worker_count = min(args.max_concurrency, len(selected))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        evaluated = executor.map(evaluate_problem, selected)
        with output_path.open("w", encoding="utf-8") as handle:
            for idx, (problem, traj, record, item_elapsed) in enumerate(evaluated, start=1):
                print(f"\n{format_progress_prefix(idx, len(selected), run_started)} {problem.id}")
                print(f"  question={problem.question}")
                print(f"  gold={problem.answer}")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(
                    "  "
                    f"terminated={traj.terminated_by} "
                    f"agents={traj.active_agents} "
                    f"handoffs={traj.n_handoffs} "
                    f"em={record['em']:.3f} f1={record['f1']:.3f}"
                )
                if traj.error:
                    print(f"  error={traj.error}")
                print(f"  final={traj.final_answer}")
                print_old_trajectory_log(traj, raw_chars=args.log_raw_chars)
                print(
                    "  progress="
                    f"{format_progress_line(idx, len(selected), run_started)} "
                    f"last={format_duration(item_elapsed)}"
                )

    print("\nDone.")
    print(f"  output: {output_path}")
    print(f"  accuracy: {mean(r['em'] for r in records):.3f}")
    print(f"  avg_f1: {mean(r['f1'] for r in records):.3f}")
    print(f"  total_time: {format_duration(time.monotonic() - run_started)}")


def run_old_protocol_trajectory(
    problem: MuSiQueProblem,
    callers: Dict[str, OpenAIChatLLMCaller],
    *,
    t_max: int,
    start_agent: str,
) -> OldTrajectory:
    traj = OldTrajectory(problem_id=problem.id)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": render_old_system_prompt(start_agent)},
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]
    current_agent = start_agent
    prior_tentative = False

    try:
        for turn in range(t_max):
            raw_output = callers[current_agent]([dict(message) for message in messages])
            raw_outputs = [raw_output]
            parsed = parse_old_action(
                raw_output,
                active_agent=current_agent,
                prior_tentative=prior_tentative,
            )
            if is_empty_old_action(parsed):
                retry_messages = [dict(message) for message in messages]
                retry_messages.append({"role": "user", "content": old_retry_instruction()})
                raw_output = callers[current_agent](retry_messages)
                raw_outputs.append(raw_output)
                parsed = parse_old_action(
                    raw_output,
                    active_agent=current_agent,
                    prior_tentative=prior_tentative,
                )

            step = OldTrajectoryStep(
                turn=turn,
                active_agent=current_agent,
                reasoning=parsed.reasoning,
                tentative_answer=parsed.tentative_answer,
                action=parsed.action,
                handoff_target=parsed.handoff_target,
                handoff_note=parsed.handoff_note,
                final_answer=parsed.final_answer,
                raw_output=raw_output,
                raw_outputs=raw_outputs,
            )
            traj.steps.append(step)

            if is_empty_old_action(parsed):
                messages.append(
                    {"role": "assistant", "content": render_old_assistant_message(current_agent, parsed)}
                )
                traj.terminated_by = "exception"
                traj.error = f"Invalid old-protocol action from {current_agent}"
                return traj

            if parsed.tentative_answer:
                prior_tentative = True

            messages.append(
                {"role": "assistant", "content": render_old_assistant_message(current_agent, parsed)}
            )

            if parsed.final_answer is not None:
                traj.final_answer = parsed.final_answer
                traj.terminated_by = "stop"
                return traj

            if parsed.handoff_target is not None:
                current_agent = parsed.handoff_target
                messages[0] = {
                    "role": "system",
                    "content": render_old_system_prompt(current_agent),
                }

        traj.terminated_by = "truncated"
        return traj
    except Exception as exc:
        traj.terminated_by = "exception"
        traj.error = str(exc)
        return traj


def render_old_system_prompt(agent_id: str) -> str:
    return OLD_SFT_SYSTEM_PROMPT_TEMPLATE.format(
        AGENT_ID=agent_id,
        OTHER_AGENTS=format_other_agents(agent_id),
    )


def format_other_agents(agent_id: str) -> str:
    others = [other for other in AGENT_IDS if other != agent_id]
    if len(others) == 2:
        return f"{others[0]} and {others[1]}"
    return ", ".join(others)


def parse_old_action(
    raw_output: str,
    *,
    active_agent: str,
    prior_tentative: bool,
) -> OldParsedAction:
    try:
        payload = load_json_object(raw_output)
    except (json.JSONDecodeError, TypeError, ValueError):
        return OldParsedAction()
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return OldParsedAction()

    reasoning = payload.get("reasoning") or ""
    tentative_answer = payload.get("tentative_answer") or ""
    action = payload.get("action")
    handoff_target = payload.get("handoff_target")
    handoff_note = payload.get("handoff_note")
    confirmed_answer = payload.get("confirmed_answer")
    if handoff_target == "":
        handoff_target = None
    if handoff_note == "":
        handoff_note = None
    if confirmed_answer == "":
        confirmed_answer = None

    if not isinstance(reasoning, str) or not isinstance(tentative_answer, str):
        return OldParsedAction()
    reasoning = reasoning.strip()
    tentative_answer = tentative_answer.strip()

    if action == "handoff":
        if confirmed_answer is not None:
            return OldParsedAction()
        if not isinstance(handoff_target, str) or handoff_target not in AGENT_IDS:
            return OldParsedAction()
        if handoff_target == active_agent:
            return OldParsedAction()
        if handoff_note is not None and not isinstance(handoff_note, str):
            return OldParsedAction()
        note = handoff_note.strip() if isinstance(handoff_note, str) else None
        return OldParsedAction(
            reasoning=reasoning,
            tentative_answer=tentative_answer,
            action=action,
            handoff_target=handoff_target,
            handoff_note=note or None,
        )

    if action == "confirm_stop":
        if not prior_tentative:
            return OldParsedAction()
        if handoff_target is not None or handoff_note is not None:
            return OldParsedAction()
        if not isinstance(confirmed_answer, str) or not confirmed_answer.strip():
            return OldParsedAction()
        return OldParsedAction(
            reasoning=reasoning,
            tentative_answer=tentative_answer,
            action=action,
            final_answer=confirmed_answer.strip(),
        )

    return OldParsedAction()


def load_json_object(raw_output: str) -> object:
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
        # If model outputs a JSON array instead of object, take the first element
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return obj[0]
        return obj
    except json.JSONDecodeError:
        # Fallback 1: find first { and try raw_decode (handles [{ ... }] prefix)
        start = text.find("{")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Fallback 2: regex extraction for missing-comma format errors
        # e.g. {"reasoning": "..." \n "action": "handoff"  (comma missing between fields)
        import re
        result = {}
        for field, pattern in [
            ("reasoning",       r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"'),
            ("tentative_answer",r'"tentative_answer"\s*:\s*"((?:[^"\\]|\\.)*)"'),
            ("action",          r'"action"\s*:\s*"([^"]*)"'),
            ("handoff_target",  r'"handoff_target"\s*:\s*"([^"]*)"'),
            ("handoff_note",    r'"handoff_note"\s*:\s*"((?:[^"\\]|\\.)*)"'),
            ("confirmed_answer",r'"confirmed_answer"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                result[field] = m.group(1)
        # handle null values
        for field in ("handoff_target", "handoff_note", "confirmed_answer"):
            if field not in result:
                m = re.search(rf'"{field}"\s*:\s*null', text)
                if m:
                    result[field] = None
        if "action" in result:
            return result
        raise


def render_old_assistant_message(agent_id: str, parsed: OldParsedAction) -> str:
    parts = [f"[{agent_id}]"]
    if parsed.reasoning:
        parts.append(parsed.reasoning)
    if parsed.tentative_answer:
        parts.append(f"tentative_answer: {parsed.tentative_answer}")
    if parsed.handoff_target:
        line = f"-> handoff to {parsed.handoff_target}"
        if parsed.handoff_note:
            line += f": {parsed.handoff_note}"
        parts.append(line)
    if parsed.final_answer:
        parts.append(f"-> answer: {parsed.final_answer}")
    return "\n".join(parts)


def old_retry_instruction() -> str:
    return (
        "Your previous output was invalid. Return only the old SFT JSON object "
        'with "reasoning", "tentative_answer", "action", "handoff_target", '
        '"handoff_note", and "confirmed_answer". Use action="handoff" unless '
        "you are confirming a previous tentative_answer."
    )


def is_empty_old_action(parsed: OldParsedAction) -> bool:
    return parsed.action is None and not parsed.reasoning and parsed.final_answer is None


def old_trajectory_to_dict(traj: OldTrajectory) -> Dict[str, Any]:
    return {
        "problem_id": traj.problem_id,
        "steps": [asdict(step) for step in traj.steps],
        "final_answer": traj.final_answer,
        "terminated_by": traj.terminated_by,
        "error": traj.error,
        "active_agents": traj.active_agents,
        "n_handoffs": traj.n_handoffs,
    }


def print_old_trajectory_log(traj: OldTrajectory, *, raw_chars: int) -> None:
    for step in traj.steps:
        print(f"  turn={step.turn} agent={step.active_agent} action={step.action}")
        if step.reasoning:
            print(f"    reasoning={one_line(step.reasoning)}")
        if step.tentative_answer:
            print(f"    tentative={one_line(step.tentative_answer)}")
        if step.handoff_target:
            note = f" note={one_line(step.handoff_note or '')}" if step.handoff_note else ""
            print(f"    handoff={step.handoff_target}{note}")
        if step.final_answer:
            print(f"    confirm={one_line(step.final_answer)}")
        if raw_chars > 0:
            raw = one_line(step.raw_output)
            if len(raw) > raw_chars:
                raw = raw[:raw_chars] + "...<truncated>"
            print(f"    raw={raw}")


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def format_progress_prefix(idx: int, total: int, started_at: float) -> str:
    percent = 100.0 * (idx - 1) / total if total else 100.0
    return (
        f"[{idx}/{total} {percent:5.1f}% "
        f"elapsed={format_duration(time.monotonic() - started_at)}]"
    )


def format_progress_line(idx: int, total: int, started_at: float) -> str:
    elapsed = time.monotonic() - started_at
    avg = elapsed / idx if idx > 0 else 0.0
    eta = avg * max(total - idx, 0)
    percent = 100.0 * idx / total if total else 100.0
    return (
        f"{idx}/{total} {percent:5.1f}% "
        f"elapsed={format_duration(elapsed)} "
        f"avg={format_duration(avg)}/item "
        f"eta={format_duration(eta)}"
    )


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


if __name__ == "__main__":
    main()
