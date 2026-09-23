"""MultiPL-E adapters for the reusable MuSiQue AFlow implementation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_AFLOW_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "AFlow"
MULTIPL_E_SCRIPTS = PROJECT_ROOT / "Code" / "MultiPL-E" / "scripts"
for path in (PACKAGE_PARENT, MUSIQUE_AFLOW_DIR, MULTIPL_E_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multipl_e_completion_adapter import (  # noqa: E402
    PROMPT_PROTOCOL,
    build_chat_user_prompt,
    normalize_completion,
)


SCHEMA_VERSION = 1
TIE_BREAK_PRIORITY = ("A3", "A2", "A1")


@dataclass(frozen=True)
class MultiPLEProblem:
    id: str
    name: str
    root_dataset: str
    language: str
    rendered_text: str
    row: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
    return rows


def build_problem(root_dataset: str, language: str, row: dict[str, Any]) -> MultiPLEProblem:
    name = row["name"]
    return MultiPLEProblem(
        id=f"{root_dataset}/{language}/{name}",
        name=name,
        root_dataset=root_dataset,
        language=language,
        rendered_text=build_chat_user_prompt(
            language,
            row["prompt"],
            row.get("tests", ""),
            row.get("stop_tokens") or [],
        ),
        row=row,
    )


def load_manifest_tasks(
    manifests: list[Path],
    *,
    partition: str,
    languages: list[str] | None = None,
) -> list[MultiPLEProblem]:
    if partition not in {"train", "test"}:
        raise ValueError("partition must be train or test")
    tasks = []
    seen = set()
    file_key = f"{partition}_file"
    for manifest_arg in manifests:
        manifest_path = manifest_arg.resolve()
        manifest = load_json(manifest_path)
        root_dataset = manifest["root_dataset"]
        for language, details in sorted(manifest["languages"].items()):
            if languages and language not in languages:
                continue
            rows = read_jsonl((manifest_path.parent / details[file_key]).resolve())
            expected = {item["problem_id"] for item in details[partition]}
            if {row.get("name") for row in rows} != expected:
                raise ValueError(
                    f"Manifest mismatch for {root_dataset}/{language}/{partition}"
                )
            for row in rows:
                key = (root_dataset, language, row["name"])
                if key in seen:
                    raise ValueError(f"Duplicate task: {key}")
                seen.add(key)
                tasks.append(build_problem(root_dataset, language, row))
    return tasks


def select_search_tasks(
    tasks: list[MultiPLEProblem], size: int, seed: int
) -> list[MultiPLEProblem]:
    if size < 1:
        raise ValueError("search size must be positive")
    if size > len(tasks):
        raise ValueError(f"search size {size} exceeds train task count {len(tasks)}")
    selected = list(tasks)
    random.Random(seed).shuffle(selected)
    return selected[:size]


def _load_json_object(raw_output: str) -> Any | None:
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                payload, _ = json.JSONDecoder().raw_decode(text[start:])
                return payload
            except json.JSONDecodeError:
                pass
    return None


def parse_solve_preserving_code(_self: Any, raw_output: str) -> dict[str, str] | None:
    payload = _load_json_object(raw_output)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str) and answer.strip():
            return {"reasoning": reasoning.strip(), "answer": answer}
    return None


def parse_answer_preserving_code(_self: Any, raw_output: str) -> dict[str, str] | None:
    payload = _load_json_object(raw_output)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"answer": answer}
    return None


def configure_core() -> dict[str, Any]:
    import operators
    import optimizer
    import workflow
    from jca.src import grader

    operators.PROMPT_SOLVE_PATH = THIS_DIR / "prompts" / "op_solve.md"
    operators.PROMPT_ENSEMBLE_PATH = THIS_DIR / "prompts" / "op_ensemble.md"
    operators.PROMPT_ANSWER_PATH = THIS_DIR / "prompts" / "op_answer.md"
    operators.Ops._parse_solve = parse_solve_preserving_code
    operators.Ops._parse_answer = parse_answer_preserving_code
    optimizer.PROMPT_OPTIMIZER_PATH = THIS_DIR / "prompts" / "optimizer_propose.md"
    grader.normalize_answer = lambda answer: answer.rstrip() if isinstance(answer, str) else ""

    @classmethod
    def from_multipl_e(cls: type, problem: MultiPLEProblem):
        return cls(
            id=problem.id,
            question=problem.rendered_text,
            rendered_text=problem.rendered_text,
        )

    workflow.WorkflowProblem.from_musique = from_multipl_e
    return {"operators": operators, "optimizer": optimizer, "workflow": workflow}


def normalize_candidate(problem: MultiPLEProblem, answer: str) -> str:
    return normalize_completion(
        answer,
        problem.row["prompt"],
        problem.row.get("tests", ""),
        problem.row.get("stop_tokens") or [],
        problem.language,
    )


def write_gzip(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)


def result_path(root: Path, problem: MultiPLEProblem) -> Path:
    return (
        root
        / problem.root_dataset
        / problem.language
        / f"{problem.name}.results.json.gz"
    )


class PublicTestWorkflowEvaluator:
    """Score workflow candidates exclusively on train70 public tests."""

    def __init__(
        self,
        *,
        root: Path,
        evaluator_script: Path,
        eval_image: str,
        docker_exec: str,
        shards: int,
        inner_workers: int,
    ) -> None:
        self.root = root
        self.evaluator_script = evaluator_script
        self.eval_image = eval_image
        self.docker_exec = docker_exec
        self.shards = shards
        self.inner_workers = inner_workers

    def __call__(
        self,
        workflow: Any,
        problems: list[MultiPLEProblem],
        callers: Any,
        *,
        max_concurrency: int,
    ) -> tuple[float, float, list[dict[str, Any]]]:
        from workflow import run_workflow_on_problem

        round_root = self.root / f"round_{workflow.round_id:02d}"
        completion_root = round_root / "completions"
        evaluation_root = round_root / "results"
        records_by_id: dict[str, dict[str, Any]] = {}

        def generate(problem: MultiPLEProblem) -> tuple[MultiPLEProblem, Any, str, str | None]:
            record = run_workflow_on_problem(workflow, problem, callers)
            completion = ""
            error = record.error
            if not error and record.final_answer:
                try:
                    completion = normalize_candidate(problem, record.final_answer)
                except Exception as exc:
                    error = f"Normalization failed: {type(exc).__name__}: {exc}"
            elif not error:
                error = "Workflow returned an empty continuation"
            return problem, record, completion, error

        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(problems))) as pool:
            futures = {pool.submit(generate, problem): problem for problem in problems}
            for future in as_completed(futures):
                problem, record, completion, error = future.result()
                payload = dict(problem.row)
                payload["completions"] = [completion]
                write_gzip(
                    completion_root
                    / problem.root_dataset
                    / problem.language
                    / f"{problem.name}.json.gz",
                    payload,
                )
                records_by_id[problem.id] = {
                    "problem_id": problem.id,
                    "gold": "pass public train70 tests",
                    "pred": completion,
                    "em": 0.0,
                    "f1": 0.0,
                    "correct": False,
                    "error": error,
                    "n_ops": len(record.op_calls),
                    "op_kinds": [call.op for call in record.op_calls],
                    "op_records": [asdict(call) for call in record.op_calls],
                    "wall_time_s": record.wall_time_s,
                }

        command = [
            sys.executable,
            str(self.evaluator_script),
            "--input-dir",
            str(completion_root),
            "--output-dir",
            str(evaluation_root),
            "--image",
            self.eval_image,
            "--docker-exec",
            self.docker_exec,
            "--shards",
            str(min(self.shards, len(problems))),
            "--inner-workers",
            str(self.inner_workers),
            "--quiet",
        ]
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (round_root / "evaluator.log").write_text(
            (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Public-test evaluator failed for round {workflow.round_id}: "
                f"returncode={completed.returncode}"
            )

        for problem in problems:
            path = result_path(evaluation_root, problem)
            if not path.is_file():
                records_by_id[problem.id]["error"] = "Evaluator result file missing"
                continue
            result = load_json(path)
            values = result.get("results") or []
            status = values[0].get("status") if values else None
            passed = status == "OK"
            records_by_id[problem.id].update(
                em=float(passed),
                f1=float(passed),
                correct=passed,
                public_test_status=status,
            )
            if not passed and not records_by_id[problem.id].get("error"):
                records_by_id[problem.id]["error"] = (
                    f"public_test_status={status or 'missing'}"
                )
        records = [records_by_id[problem.id] for problem in problems]
        score = sum(record["correct"] for record in records) / len(records)
        summary = {
            "round_id": workflow.round_id,
            "tasks": len(records),
            "passed": sum(record["correct"] for record in records),
            "pass_rate": score,
            "evaluation_wall_time_s": round(time.monotonic() - started, 3),
        }
        (round_root / "public_test_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return score, score, records


def completion_payload(
    problem: MultiPLEProblem, completion: str, status: str, workflow_file: Path
) -> dict[str, Any]:
    payload = dict(problem.row)
    payload.update(
        {
            "completions": [completion],
            "baseline": "aflow",
            "baseline_schema_version": SCHEMA_VERSION,
            "prompt_protocol": PROMPT_PROTOCOL,
            "thinking_mode": "disabled",
            "workflow_file": str(workflow_file),
            "terminated_by": status,
        }
    )
    return payload
