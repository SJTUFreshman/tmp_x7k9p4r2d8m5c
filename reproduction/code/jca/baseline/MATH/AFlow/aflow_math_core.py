#!/usr/bin/env python3
"""MATH-specific AFlow operators, workflow execution, and search utilities."""

from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jca.baseline.MATH.MAD import math_mad_role_batched as shared
from jca.src.math_eval import compute_math_em, extract_last_boxed, format_math_problem_as_prompt, normalize_math_answer
from jca.src.relaxed_json import find_json_object


THIS_DIR = Path(__file__).resolve().parent
MUSIQUE_AFLOW = THIS_DIR.parents[1] / "MuSiQue" / "AFlow"
if str(MUSIQUE_AFLOW) not in sys.path:
    sys.path.insert(0, str(MUSIQUE_AFLOW))

import workflow as workflow_core  # noqa: E402


AGENTS = ("A1", "A2", "A3")
MODEL_LABELS = {"A1": "Qwen3-1.7B", "A2": "Qwen3-4B", "A3": "Qwen3-8B", "OPT": "Qwen3-14B"}


@dataclass(frozen=True)
class Endpoint:
    agent: str
    api_base: str
    api_model: str
    model_path: str
    api_key: str = "EMPTY"
    timeout: float = 900.0


@dataclass(frozen=True)
class MathWorkflowProblem:
    id: str
    subject: str
    level: str
    rendered_text: str


def problem_view(problem: Any) -> MathWorkflowProblem:
    return MathWorkflowProblem(
        id=problem.problem_id,
        subject=problem.subject,
        level=problem.level,
        rendered_text=format_math_problem_as_prompt(problem),
    )


def request_seed(problem_id: str, workflow_round: int, call_index: int, retry_index: int, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{problem_id}:{workflow_round}:{call_index}:{retry_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_ensemble(visible: str, candidate_count: int) -> tuple[dict[str, Any] | None, str | None]:
    if "{" not in visible:
        return None, "no JSON object"
    payload = find_json_object(visible)
    if payload is None:
        return None, "invalid JSON"
    index = payload.get("chosen_index")
    reason = payload.get("reason")
    if isinstance(index, bool) or not isinstance(index, (int, float)) or int(index) != index:
        return None, "chosen_index is not an integer"
    if not 0 <= int(index) < candidate_count:
        return None, "chosen_index is out of range"
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason is empty"
    return {"chosen_index": int(index), "reason": reason.strip()}, None


def parse_answer(visible: str) -> tuple[dict[str, str] | None, str | None]:
    if "{" not in visible:
        return None, "no JSON object"
    payload = find_json_object(visible)
    if payload is None:
        return None, "invalid JSON"
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None, "answer is empty"
    return {"answer": answer.strip()}, None


class MathOps:
    def __init__(
        self, problem: MathWorkflowProblem, workflow_round: int,
        endpoints: dict[str, Endpoint], config: dict[str, Any], prompt_dir: Path,
    ) -> None:
        self.problem = problem
        self.workflow_round = workflow_round
        self.endpoints = endpoints
        self.config = config
        self.prompt_dir = prompt_dir
        self.solve_counter = 0
        self.call_counter = 0
        self.calls: list[dict[str, Any]] = []

    def _next_call(self) -> int:
        if self.call_counter >= 12:
            raise RuntimeError("workflow exceeded the 12-operator runtime budget")
        index = self.call_counter
        self.call_counter += 1
        return index

    async def _request(
        self, *, op: str, endpoint: Endpoint, messages: list[dict[str, str]], parser: Any,
    ) -> tuple[Any, dict[str, Any]]:
        call_index = self._next_call()
        started = time.monotonic()
        attempts = []
        accepted = None
        failure_reason = "no request made"
        enable_thinking = bool(self.config.get("enable_thinking", False))
        require_thinking = bool(self.config.get("require_thinking", False))
        for retry_index in range(int(self.config["request_retries"]) + 1):
            attempt_messages = copy.deepcopy(messages)
            if retry_index:
                attempt_messages.append({
                    "role": "user",
                    "content": f"The previous {op} output failed validation: {failure_reason}. Recompute and return exactly the required JSON.",
                })
            seed = request_seed(
                self.problem.id, self.workflow_round, call_index, retry_index,
                int(self.config["generation_seed"]),
            )
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                attempt = await loop.run_in_executor(
                    executor,
                    lambda: shared.request_chat(
                        endpoint.api_base, endpoint.api_model, endpoint.api_key, attempt_messages,
                        int(self.config["max_new_tokens_executor"]),
                        float(self.config["temperature_executor"]), float(self.config["top_p"]),
                        seed, endpoint.timeout, enable_thinking,
                    ),
                )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            attempt.update({"retry_index": retry_index, "messages": attempt_messages, "seed": seed})
            if not attempt["request_ok"]:
                failure_reason = attempt["exception"]
            elif not enable_thinking and attempt["thinking"].strip():
                failure_reason = "unexpected thinking output while thinking is disabled"
            elif require_thinking and not attempt["thinking"].strip():
                failure_reason = "thinking is empty"
            else:
                accepted, parse_error = parser(attempt["visible_output"])
                failure_reason = parse_error or ""
            attempt["parse_ok"] = accepted is not None
            attempt["validation_error"] = None if accepted is not None else failure_reason
            attempts.append(attempt)
            if accepted is not None:
                break
        record = {
            "call_index": call_index, "op": op, "agent": endpoint.agent,
            "api_model": endpoint.api_model, "model": endpoint.model_path,
            "messages": messages, "attempts": attempts, "success": accepted is not None,
            "failure_reason": None if accepted is not None else failure_reason,
            "wall_time_s": round(time.monotonic() - started, 6),
        }
        self.calls.append(record)
        return accepted, record

    async def solve(self, problem: Any, instruction: str = "") -> dict[str, str]:
        agent = AGENTS[self.solve_counter % len(AGENTS)]
        self.solve_counter += 1
        system = (self.prompt_dir / "op_solve.md").read_text(encoding="utf-8").strip()
        system = system.replace("{INSTRUCTION}", instruction or "(none)")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": self.problem.rendered_text}]
        accepted, _ = await self._request(
            op="solve", endpoint=self.endpoints[agent], messages=messages, parser=shared.parse_visible_json,
        )
        return accepted or {"reasoning": "", "answer": ""}

    async def ensemble(self, candidates: list[dict[str, str]], problem: Any) -> dict[str, Any]:
        if not isinstance(candidates, list) or len(candidates) < 2:
            self.calls.append({
                "call_index": self._next_call(), "op": "ensemble", "agent": "A3",
                "api_model": self.endpoints["A3"].api_model, "model": self.endpoints["A3"].model_path,
                "messages": [], "attempts": [], "success": False,
                "failure_reason": "ensemble requires at least two candidates", "wall_time_s": 0.0,
            })
            return {"chosen_index": 0, "reason": "invalid candidates"}
        blocks = []
        for index, candidate in enumerate(candidates):
            blocks.append(
                f"## Candidate {index}\nReasoning: {candidate.get('reasoning', '')}\nAnswer: {candidate.get('answer', '')}"
            )
        messages = [
            {"role": "system", "content": (self.prompt_dir / "op_ensemble.md").read_text(encoding="utf-8").strip()},
            {"role": "user", "content": f"{self.problem.rendered_text}\n\n# Candidates\n" + "\n\n".join(blocks)},
        ]
        accepted, _ = await self._request(
            op="ensemble", endpoint=self.endpoints["A3"], messages=messages,
            parser=lambda visible: parse_ensemble(visible, len(candidates)),
        )
        if accepted is not None:
            return accepted
        counts: dict[str, int] = {}
        for candidate in candidates:
            key = normalize_math_answer(candidate.get("answer", ""))
            if key:
                counts[key] = counts.get(key, 0) + 1
        if counts:
            winning = max(counts, key=counts.get)
            for index, candidate in enumerate(candidates):
                if normalize_math_answer(candidate.get("answer", "")) == winning:
                    return {"chosen_index": index, "reason": "deterministic normalized-majority fallback"}
        return {"chosen_index": 0, "reason": "deterministic first-candidate fallback"}

    async def answer_generate(self, text: Any) -> dict[str, str]:
        input_text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        messages = [
            {"role": "system", "content": (self.prompt_dir / "op_answer.md").read_text(encoding="utf-8").strip()},
            {"role": "user", "content": input_text},
        ]
        accepted, _ = await self._request(
            op="answer_generate", endpoint=self.endpoints["A3"], messages=messages, parser=parse_answer,
        )
        if accepted is not None:
            return accepted
        boxed = extract_last_boxed(input_text)
        if boxed:
            return {"answer": boxed}
        lines = [line.strip() for line in input_text.splitlines() if line.strip()]
        return {"answer": lines[-1] if lines else ""}


def validate_workflow_source(source: str) -> None:
    if len(source) > 8000:
        raise ValueError(f"source exceeds 8000 chars: {len(source)}")
    tree = ast.parse(source, mode="exec")
    allowed_ops = {"solve", "ensemble", "answer_generate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ops":
            if node.attr not in allowed_ops:
                raise ValueError(f"workflow may not access ops.{node.attr}")
    call_count = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ops"
        and node.func.attr in allowed_ops
        for node in ast.walk(tree)
    )
    if not 2 <= call_count <= 12:
        raise ValueError(f"workflow source must contain 2-12 operator calls, found {call_count}")
    workflow_core._check_ast(source)
    workflow_core.load_workflow_from_source(source, round_id=0, name="validation")


def load_workflow(path: Path, round_id: int = 0) -> Any:
    source = path.read_text(encoding="utf-8")
    validate_workflow_source(source)
    return workflow_core.load_workflow_from_source(
        source, round_id=round_id, name=path.stem, origin_path=path,
    )


async def _execute_async(
    workflow: Any, problem: Any, endpoints: dict[str, Endpoint], config: dict[str, Any], prompt_dir: Path,
) -> dict[str, Any]:
    view = problem_view(problem)
    ops = MathOps(view, workflow.round_id, endpoints, config, prompt_dir)
    started = time.monotonic()
    answer = ""
    error = None
    try:
        result = await workflow.fn(view, ops)
        answer = str(result) if result is not None else ""
        if len(ops.calls) < 2:
            answer = ""
            error = f"workflow executed only {len(ops.calls)} operator calls; minimum is 2"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    calls = sorted(ops.calls, key=lambda record: int(record["call_index"]))
    return {
        "problem_id": problem.problem_id, "subject": problem.subject, "level": problem.level,
        "prediction": answer.strip(), "gold_answer": problem.gold_answer,
        "em": compute_math_em(answer, problem.gold_answer), "correct": bool(compute_math_em(answer, problem.gold_answer)),
        "error": error, "n_ops": len(calls), "op_kinds": [call["op"] for call in calls],
        "op_records": calls, "wall_time_s": round(time.monotonic() - started, 6),
    }


def execute_workflow(
    workflow: Any, problem: Any, endpoints: dict[str, Endpoint], config: dict[str, Any], prompt_dir: Path,
) -> dict[str, Any]:
    return asyncio.run(_execute_async(workflow, problem, endpoints, config, prompt_dir))


def evaluate_workflow(
    workflow: Any, problems: list[Any], endpoints: dict[str, Endpoint], config: dict[str, Any],
    prompt_dir: Path, max_concurrency: int,
) -> tuple[float, list[dict[str, Any]]]:
    records = []
    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        futures = {
            pool.submit(execute_workflow, workflow, problem, endpoints, config, prompt_dir): problem
            for problem in problems
        }
        for future in as_completed(futures):
            problem = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append({
                    "problem_id": problem.problem_id, "subject": problem.subject, "level": problem.level,
                    "prediction": "", "gold_answer": problem.gold_answer, "em": 0.0, "correct": False,
                    "error": f"{type(exc).__name__}: {exc}", "n_ops": 0, "op_kinds": [],
                    "op_records": [], "wall_time_s": 0.0,
                })
    records.sort(key=lambda record: record["problem_id"])
    return sum(record["em"] for record in records) / len(problems), records


def ucb_parent(nodes: list[dict[str, Any]], round_id: int, seed: int) -> dict[str, Any]:
    valid = [node for node in nodes if node["parse_ok"]]
    total_visits = sum(int(node["visits"]) for node in valid)
    scored = []
    for node in valid:
        exploit = float(node["dev_em"])
        explore = 1.4 * math.sqrt(math.log(max(total_visits, 1) + 1) / max(int(node["visits"]), 1))
        scored.append((exploit + explore, node))
    best_score = max(score for score, _ in scored)
    winners = [node for score, node in scored if abs(score - best_score) < 1e-9]
    return random.Random(seed + round_id).choice(winners)


def failure_samples(records: list[dict[str, Any]], limit: int = 5) -> str:
    failed = [record for record in records if not record["correct"]][:limit]
    if not failed:
        return "(no failures on the fixed train sample)"
    lines = []
    for record in failed:
        prediction = " ".join(str(record.get("prediction") or "").split())[:160]
        error = record.get("error")
        line = (
            f"- problem={record['problem_id']} subject={record['subject']!r} level={record['level']!r} "
            f"gold={record['gold_answer']!r} pred={prediction!r}"
        )
        if error:
            line += f" error={error!r}"
        lines.append(line)
    return "\n".join(lines)


def extract_source(visible: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", visible, re.DOTALL)
    return (match.group(1) if match else visible).strip()


def propose_workflow(
    parent: dict[str, Any], optimizer: Endpoint, config: dict[str, Any], prompt_path: Path,
    round_id: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    template = prompt_path.read_text(encoding="utf-8").strip()
    parent_source = Path(parent["source_file"]).read_text(encoding="utf-8")
    user = (
        template.replace("{CURRENT_WORKFLOW_CODE}", parent_source)
        .replace("{DEV_SIZE}", str(config["search_size"]))
        .replace("{CURRENT_EM}", f"{parent['dev_em']:.3f}")
        .replace("{FAILURE_SAMPLES}", failure_samples(parent["dev_records"]))
    )
    messages = [
        {"role": "system", "content": "Output only complete Python source for the requested workflow."},
        {"role": "user", "content": user},
    ]
    attempts = []
    source = ""
    failure_reason = "no request made"
    enable_thinking = bool(config.get("enable_thinking", False))
    require_thinking = bool(config.get("require_thinking", False))
    for retry_index in range(int(config["request_retries"]) + 1):
        attempt_messages = copy.deepcopy(messages)
        if retry_index:
            attempt_messages.append({
                "role": "user",
                "content": f"The previous proposal failed validation: {failure_reason}. Return a corrected complete Python workflow satisfying every policy constraint.",
            })
        digest = hashlib.sha256(f"{config['generation_seed']}:optimizer:{round_id}:{retry_index}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        attempt = shared.request_chat(
            optimizer.api_base, optimizer.api_model, optimizer.api_key, attempt_messages,
            int(config["max_new_tokens_optimizer"]), float(config["temperature_optimizer"]),
            float(config["top_p"]), seed, optimizer.timeout,
            enable_thinking,
        )
        attempt.update({"retry_index": retry_index, "messages": attempt_messages, "seed": seed})
        if not attempt["request_ok"]:
            failure_reason = attempt["exception"]
        elif not enable_thinking and attempt["thinking"].strip():
            failure_reason = "unexpected thinking output while thinking is disabled"
        elif require_thinking and not attempt["thinking"].strip():
            failure_reason = "thinking is empty"
        else:
            source = extract_source(attempt["visible_output"])
            try:
                validate_workflow_source(source)
                failure_reason = ""
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
        attempt["parse_ok"] = not failure_reason
        attempt["validation_error"] = failure_reason or None
        attempts.append(attempt)
        if not failure_reason:
            return source, attempts, None
    return source, attempts, failure_reason


def write_workflow(run_dir: Path, round_id: int, suffix: str, source: str) -> Path:
    path = run_dir / "workflows" / f"round_{round_id:02d}_{suffix}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(path)
    return path


def best_node(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [node for node in nodes if node["parse_ok"]]
    if not valid:
        raise ValueError("search produced no valid workflow")
    return max(valid, key=lambda node: (node["dev_em"], -node["round_id"]))


def token_usage_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        for op in record.get("op_records", []):
            for attempt in op.get("attempts", []):
                usage = attempt.get("usage") or {}
                for key in totals:
                    totals[key] += int(usage.get(key) or 0)
    return totals
