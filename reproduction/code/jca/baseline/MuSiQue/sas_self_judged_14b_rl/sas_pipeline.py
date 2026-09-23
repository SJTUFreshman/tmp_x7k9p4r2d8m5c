"""One-shot SAS rollout and Qwen14B self-judged RL data builder.

This intentionally keeps SAS separate from the MAS protocol. Each problem is
sampled once per rollout, with one retry only for an invalid JSON response.
Successful rows use the same per-turn RL record shape consumed by rl_train.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_PARENT = REPO_ROOT.parent
for path in (PACKAGE_PARENT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jca.src.data import MuSiQueProblem, format_problem_as_prompt, load_musique  # noqa: E402
from jca.src.grader import compute_em_f1  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402


SAS_SYSTEM_PROMPT = """You are a single-agent multi-hop question answering system.

Read the question and all paragraphs carefully. Produce exactly one JSON object:

{
  \"reasoning\": \"a concise reasoning chain citing paragraph numbers\",
  \"final_answer\": \"the one final answer, preferably wrapped in \\\\boxed{...}\"
}

There is no handoff and no second agent. Give one non-empty final_answer. Output
only the JSON object and no markdown or additional commentary."""

RETRY_PROMPT = (
    "Your previous response was invalid. Return only one JSON object with the "
    "string fields `reasoning` and non-empty `final_answer`. Do not add markdown."
)

JUDGE_SYSTEM_PROMPT = """You are a strict judge for a one-shot single-agent QA system.

The policy has only one turn and no handoff. Score PROCESS QUALITY, not answer
correctness itself (correctness is supplied separately by F1).

Score two dimensions in [-1, 1].

reasoning_score:
  +1.0 correct paragraph citations, complete multi-hop logic
  +0.7 mostly correct with a minor gap or imprecise citation
  +0.5 partially correct, one hop missing or citation vague
   0.0 restates the question or has no useful citations
  -0.5 wrong paragraph or partially misleading reasoning
  -1.0 completely wrong and misleading

finalization_score:
  MUST be exactly one of -1, 0, or 1
  +1 clear, unique, stably extractable final answer
   0 non-empty answer with ambiguity, hesitation, or formatting issues, but the
     intended submitted answer is still identifiable
  -1 conflicting answers or no stable usable submitted answer

Finalization is STRUCTURE-ONLY. A clear, unique, stably extractable answer MUST
receive +1 even when it is factually wrong, contradicts the gold answer, or is
unsupported by the reasoning. Never use correctness, the gold answer, F1, or
reasoning quality to lower finalization_score. Use -1 only when the submitted
text itself contains conflicting candidate answers or does not identify one
stable answer. Return only one JSON object with numeric reasoning_score,
numeric finalization_score, and a concise comment."""


@dataclass
class ParsedSAS:
    reasoning: str
    final_answer: str


def _load_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    return obj if isinstance(obj, dict) else None


def parse_sas_response(raw: str) -> Optional[ParsedSAS]:
    payload = _load_json_object(raw)
    if payload is None:
        return None
    reasoning = payload.get("reasoning") or ""
    final_answer = payload.get("final_answer")
    if not isinstance(reasoning, str) or not isinstance(final_answer, str):
        return None
    final_answer = final_answer.strip()
    if not final_answer:
        return None
    return ParsedSAS(reasoning=reasoning.strip(), final_answer=final_answer)


def run_sas_rollout(
    problem: MuSiQueProblem,
    caller: OpenAIChatLLMCaller,
) -> Tuple[Optional[Dict[str, Any]], str]:
    messages = [
        {"role": "system", "content": SAS_SYSTEM_PROMPT},
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]
    raw = caller([dict(m) for m in messages])
    parsed = parse_sas_response(raw)
    if parsed is None:
        retry_messages = [*messages, {"role": "user", "content": RETRY_PROMPT}]
        raw = caller([dict(m) for m in retry_messages])
        parsed = parse_sas_response(raw)
    if parsed is None:
        return None, "exception"
    return {
        "problem_id": problem.id,
        "turn": 0,
        "agent_id": "SAS",
        "messages": messages,
        "response": str(raw),
        "reasoning": parsed.reasoning,
        "final_answer": parsed.final_answer,
        "terminated_by": "stop",
    }, "stop"


def _judge_request(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"judge HTTP {exc.code}: {body}") from exc
    return str(data["choices"][0]["message"].get("content", ""))


def judge_one(
    problem: MuSiQueProblem,
    record: Dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    retries: int = 2,
) -> Optional[Dict[str, float | str]]:
    em, f1 = compute_em_f1(record["final_answer"], problem)
    user = (
        f"# Question\n{problem.question}\n\n"
        f"# Paragraphs\n{format_problem_as_prompt(problem)}\n\n"
        f"# Gold Answer\n{problem.answer}\n\n"
        f"# Reasoning\n{record['reasoning']}\n\n"
        f"# Final Answer\n{record['final_answer']}\n\n"
        f"# Correctness\n{'YES (F1 > 0)' if f1 > 0 else 'NO'}\n\n"
        "Score this one turn."
    )
    raw = ""
    for attempt in range(max(1, retries + 1)):
        try:
            raw = _judge_request(
                api_base, api_key, model,
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            )
            payload = _load_json_object(raw)
            if payload is not None:
                reasoning = float(payload.get("reasoning_score"))
                finalization = float(payload.get("finalization_score"))
                reasoning = max(-1.0, min(1.0, reasoning))
                if finalization not in (-1.0, 0.0, 1.0):
                    continue
                return {
                    "reasoning_score": reasoning,
                    "finalization_score": finalization,
                    "judge_score": 0.5 * (reasoning + finalization),
                    "comment": str(payload.get("comment", "")),
                    "em": em,
                    "f1": f1,
                }
        except (Exception, TypeError, ValueError):
            pass
    return None


def build_record(
    record: Dict[str, Any],
    judged: Dict[str, Any],
    alpha: float,
    judge_model: str,
) -> Dict[str, Any]:
    task_reward = 2.0 * float(judged["f1"]) - 1.0
    judge_score = float(judged["judge_score"])
    return {
        **record,
        "reward": round(alpha * task_reward + (1.0 - alpha) * judge_score, 4),
        "task_reward": round(task_reward, 4),
        "judge_score": round(judge_score, 4),
        "turn_scores": {
            "reasoning_score": judged["reasoning_score"],
            "finalization_score": judged["finalization_score"],
            "judge_score": judged["judge_score"],
            "comment": judged["comment"],
        },
        "judge_failed": False,
        "judge_model": judge_model,
        "reward_source": "qwen14b_sas_judge",
        "em": judged["em"],
        "f1": judged["f1"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one-shot SAS Qwen14B-judged RL records.")
    p.add_argument("--phase", choices=("rollout", "judge", "all"), default="all")
    p.add_argument("--split", default="train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None, help="Raw rollout JSONL for --phase judge")
    p.add_argument("--api-base", default="http://127.0.0.1:8200/v1")
    p.add_argument("--api-model", default="sas_policy")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--judge-api-base", default="http://127.0.0.1:8300/v1")
    p.add_argument("--judge-model", default="qwen14b_judge")
    p.add_argument("--judge-api-key", default="EMPTY")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--num-rollouts", type=int, default=2)
    p.add_argument("--rollout-concurrency", type=int, default=1)
    p.add_argument("--judge-timeout", type=float, default=300.0)
    p.add_argument("--judge-retries", type=int, default=2)
    p.add_argument("--judge-concurrency", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problems = load_musique(args.split, data_dir=args.data_dir)
    selected = problems[args.start : args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    problem_by_id = {problem.id: problem for problem in selected}
    caller = None
    if args.phase in {"rollout", "all"}:
        generation = GenerationOptions(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=False,
        )
        caller = OpenAIChatLLMCaller(
            args.api_base, args.api_model,
            generation=generation, api_key=args.api_key, timeout=900,
        )
    written = 0

    if args.phase == "judge":
        if args.input is None:
            raise SystemExit("--input is required for --phase judge")
        raw_rows = []
        with args.input.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                problem = problem_by_id.get(str(row.get("problem_id", "")))
                if problem is None:
                    continue
                raw_rows.append((problem, row))
        def judge_raw(item: Tuple[MuSiQueProblem, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            problem, row = item
            judged = judge_one(
                problem, row, api_base=args.judge_api_base,
                api_key=args.judge_api_key, model=args.judge_model,
                timeout=args.judge_timeout, retries=args.judge_retries,
            )
            if judged is not None:
                return build_record(row, judged, args.alpha, args.judge_model)
            return None

        if args.judge_concurrency <= 1:
            judged_results = [judge_raw(item) for item in raw_rows]
        else:
            workers = min(args.judge_concurrency, len(raw_rows))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                judged_results = list(pool.map(judge_raw, raw_rows))
        judged_rows = [row for row in judged_results if row is not None]
        with args.output.open("w", encoding="utf-8") as handle:
            for row in judged_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"raw={len(raw_rows)} judged={len(judged_rows)} "
            f"failed_judge={len(raw_rows) - len(judged_rows)} output={args.output}"
        )
        return

    def process(item: Tuple[int, MuSiQueProblem]) -> List[Dict[str, Any]]:
        idx, problem = item
        rows: List[Dict[str, Any]] = []
        assert caller is not None
        for rollout_idx in range(args.num_rollouts):
            raw_record, status = run_sas_rollout(problem, caller)
            if raw_record is None:
                continue
            raw_record["rollout_idx"] = rollout_idx
            em, f1 = compute_em_f1(raw_record["final_answer"], problem)
            raw_record["em"] = em
            raw_record["f1"] = f1
            if args.phase == "rollout":
                rows.append(raw_record)
            else:
                judged = judge_one(
                    problem, raw_record, api_base=args.judge_api_base,
                    api_key=args.judge_api_key, model=args.judge_model,
                    timeout=args.judge_timeout, retries=args.judge_retries,
                )
                if judged is not None:
                    rows.append(build_record(raw_record, judged, args.alpha, args.judge_model))
        return rows

    items = list(enumerate(selected, start=args.start))
    if args.rollout_concurrency <= 1:
        results = [process(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=args.rollout_concurrency) as pool:
            results = [future.result() for future in as_completed([pool.submit(process, item) for item in items])]
    with args.output.open("w", encoding="utf-8") as handle:
        for rows in results:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    expected = len(selected) * args.num_rollouts
    print(
        f"phase={args.phase} expected={expected} wrote={written} "
        f"dropped={expected - written} output={args.output}"
    )


if __name__ == "__main__":
    main()
