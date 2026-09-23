"""Rescore existing RL rollout records with a local OpenAI-compatible judge.

This does not regenerate trajectories. It reads per-turn records produced by
rl_rollout.py, groups them by (problem_id, rollout_idx), asks a judge model to
score each turn, then rewrites only reward-related fields.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PACKAGE_PARENT, REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jca.src.data import MuSiQueProblem, load_musique  # noqa: E402
from jca.src.judge import (  # noqa: E402
    CORRECT_BONUS,
    TurnScore,
    TrajectoryReward,
    _JUDGE_SYSTEM,
    _build_user_prompt,
)
from run_sft_old_protocol import (  # noqa: E402
    OldParsedAction,
    is_empty_old_action,
    load_json_object,
    parse_old_action,
)


GroupKey = Tuple[str, str]
IndexedRecord = Tuple[int, Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore existing JCA RL rollout JSONL with a local judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rl_data/rl07081658_rollout_train_0_1000_final.jsonl"),
        help="Existing rollout JSONL to rescore. The file is never modified.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_data/rl07081658_rollout_train_0_1000_qwen14b.jsonl"),
        help="Destination JSONL containing the same turns with updated rewards.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", type=Path, default=Path("musique_data"))
    parser.add_argument("--judge-api-base", default="http://127.0.0.1:8300/v1")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="qwen14b_judge")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the run if any trajectory cannot be judged.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Only process this many pending trajectory groups.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and reconstruct groups without calling the judge or writing output.",
    )
    return parser.parse_args()


def group_key(record: Dict[str, Any]) -> GroupKey:
    problem_id = str(record.get("problem_id", ""))
    rollout_idx = str(record.get("rollout_idx", 0))
    return problem_id, rollout_idx


def read_input_records(path: Path) -> List[IndexedRecord]:
    records: List[IndexedRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records.append((len(records), record))
    return records


def group_records(records: Iterable[IndexedRecord]) -> Dict[GroupKey, List[IndexedRecord]]:
    groups: Dict[GroupKey, List[IndexedRecord]] = {}
    for idx, record in records:
        groups.setdefault(group_key(record), []).append((idx, record))
    for values in groups.values():
        values.sort(key=lambda item: (int(item[1].get("turn", 0)), item[0]))
    return groups


def load_completed_output(
    output: Path,
    group_sizes: Dict[GroupKey, int],
) -> Tuple[set[GroupKey], List[str], int]:
    """Return complete keys, retained output lines, and pruned incomplete rows."""
    if not output.exists():
        return set(), [], 0

    rows_by_key: Dict[GroupKey, List[str]] = {}
    pruned = 0
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                pruned += 1
                continue
            key = group_key(record)
            if key not in group_sizes:
                pruned += 1
                continue
            rows_by_key.setdefault(key, []).append(stripped)

    complete_keys: set[GroupKey] = set()
    retained_lines: List[str] = []
    for key, lines in rows_by_key.items():
        expected = group_sizes[key]
        if len(lines) >= expected:
            complete_keys.add(key)
            retained_lines.extend(lines[:expected])
            pruned += max(0, len(lines) - expected)
        else:
            pruned += len(lines)
    return complete_keys, retained_lines, pruned


def rewrite_retained_output(output: Path, retained_lines: List[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in retained_lines:
            handle.write(line)
            handle.write("\n")
    tmp.replace(output)


def _as_clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _fallback_parse_response(raw_output: str, row_action: Optional[str]) -> OldParsedAction:
    try:
        payload = load_json_object(raw_output)
    except Exception:
        return OldParsedAction(action=row_action)

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return OldParsedAction(action=row_action)

    action = payload.get("action") or row_action
    handoff_target = payload.get("handoff_target")
    handoff_note = payload.get("handoff_note")
    final_answer = payload.get("confirmed_answer")
    if final_answer is None:
        final_answer = payload.get("final_answer")

    return OldParsedAction(
        reasoning=_as_clean_str(payload.get("reasoning")),
        tentative_answer=_as_clean_str(payload.get("tentative_answer")),
        action=_as_clean_str(action) or None,
        handoff_target=_as_clean_str(handoff_target) or None,
        handoff_note=_as_clean_str(handoff_note) or None,
        final_answer=_as_clean_str(final_answer) or None,
    )


def parse_step(record: Dict[str, Any], prior_tentative: bool) -> OldParsedAction:
    raw_output = str(record.get("response") or "")
    agent_id = str(record.get("agent_id") or "")
    row_action = record.get("action")
    parsed = parse_old_action(
        raw_output,
        active_agent=agent_id,
        prior_tentative=prior_tentative,
    )
    if not is_empty_old_action(parsed):
        return parsed
    return _fallback_parse_response(raw_output, row_action)


def problem_to_dict(problem: MuSiQueProblem) -> Dict[str, Any]:
    return {
        "id": problem.id,
        "question": problem.question,
        "answer": problem.answer,
        "answer_aliases": problem.answer_aliases,
        "hop": problem.hop,
        "paragraphs": problem.paragraphs,
    }


def reconstruct_result(
    key: GroupKey,
    indexed_records: List[IndexedRecord],
    problem_by_id: Dict[str, MuSiQueProblem],
) -> Dict[str, Any]:
    problem_id, _rollout_idx = key
    if problem_id not in problem_by_id:
        raise KeyError(f"problem_id not found in {problem_id!r}")

    steps: List[Dict[str, Any]] = []
    prior_tentative = False
    final_answer: Optional[str] = None

    for _idx, record in indexed_records:
        parsed = parse_step(record, prior_tentative)
        if parsed.tentative_answer:
            prior_tentative = True
        if parsed.final_answer:
            final_answer = parsed.final_answer

        step = {
            "turn": int(record.get("turn", len(steps))),
            "active_agent": str(record.get("agent_id") or ""),
            "reasoning": parsed.reasoning,
            "tentative_answer": parsed.tentative_answer,
            "action": parsed.action or record.get("action"),
            "handoff_target": parsed.handoff_target,
            "handoff_note": parsed.handoff_note,
            "final_answer": parsed.final_answer,
            "raw_output": record.get("response") or "",
        }
        steps.append(step)

    last_record = indexed_records[-1][1]
    f1 = float(last_record.get("f1", 0.0) or 0.0)
    em = float(last_record.get("em", 0.0) or 0.0)

    return {
        "problem": problem_to_dict(problem_by_id[problem_id]),
        "trajectory": {
            "problem_id": problem_id,
            "steps": steps,
            "final_answer": final_answer,
            "terminated_by": last_record.get("terminated_by"),
            "error": None,
        },
        "final_answer": final_answer or "",
        "terminated_by": last_record.get("terminated_by"),
        "em": em,
        "f1": f1,
    }


def call_openai_compatible_judge(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return str(parsed["choices"][0]["message"].get("content") or "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2.0 * (attempt + 1), 8.0))

    assert last_error is not None
    raise last_error


def parse_turn_scores(content: str) -> Optional[List[TurnScore]]:
    content = (content or "").strip()
    start = content.find("[")
    end = content.rfind("]") + 1
    if start < 0 or end <= start:
        return None

    try:
        raw_scores = json.loads(content[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw_scores, list) or not raw_scores:
        return None

    turn_scores: List[TurnScore] = []
    for obj in raw_scores:
        if not isinstance(obj, dict):
            continue
        try:
            ts = TurnScore(
                turn=int(obj.get("turn", 0)),
                agent_id=str(obj.get("agent_id", "")),
                reasoning_score=float(obj.get("reasoning_score", 0.0)),
                action_score=float(obj.get("action_score", 0.0)),
                comment=str(obj.get("comment", "")),
            )
        except Exception:
            continue
        ts.reasoning_score = max(-1.0, min(1.0, ts.reasoning_score))
        ts.action_score = max(-1.0, min(1.0, ts.action_score))
        turn_scores.append(ts)

    return turn_scores or None


def judge_result(
    result: Dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> Optional[TrajectoryReward]:
    steps = result.get("trajectory", {}).get("steps", [])
    if not steps:
        return None

    f1 = float(result.get("f1", 0.0) or 0.0)
    user_prompt = _build_user_prompt(
        problem=result.get("problem", {}),
        steps=steps,
        final_answer=str(result.get("final_answer") or ""),
        is_correct=f1 > 0.0,
    )
    content = call_openai_compatible_judge(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    turn_scores = parse_turn_scores(content)
    if not turn_scores:
        return None

    if f1 > 0.0:
        for ts in reversed(turn_scores):
            matched_step = next(
                (s for s in steps if str(s.get("turn", "")) == str(ts.turn)),
                None,
            )
            if matched_step and matched_step.get("action") == "confirm_stop":
                ts.action_score = min(1.0, ts.action_score + CORRECT_BONUS)
                break

    problem = result.get("problem", {})
    trajectory = result.get("trajectory", {})
    return TrajectoryReward(
        problem_id=str(problem.get("id") or trajectory.get("problem_id", "")),
        r_task=f1,
        turn_scores=turn_scores,
    )


def rescore_records(
    indexed_records: List[IndexedRecord],
    result: Dict[str, Any],
    reward: Optional[TrajectoryReward],
    *,
    alpha: float,
    judge_model: str,
    source_input: Path,
) -> Tuple[List[Dict[str, Any]], int]:
    f1 = float(result.get("f1", 0.0) or 0.0)
    task_reward = 2.0 * f1 - 1.0
    source_tag = judge_model[:-6] if judge_model.endswith("_judge") else judge_model
    turn_score_lookup: Dict[int, Dict[str, Any]] = {}
    if reward:
        for ts in reward.turn_scores:
            turn_score_lookup[ts.turn] = {
                "reasoning_score": round(ts.reasoning_score, 4),
                "action_score": round(ts.action_score, 4),
                "judge_score": round(ts.judge_score, 4),
                "comment": ts.comment,
            }

    out_records: List[Dict[str, Any]] = []
    judge_failed_rows = 0
    for _idx, record in indexed_records:
        turn = int(record.get("turn", 0))
        ts_info = turn_score_lookup.get(turn, {})
        if ts_info:
            judge_score = float(ts_info["judge_score"])
            total_reward = alpha * task_reward + (1.0 - alpha) * judge_score
            judge_failed = False
        else:
            judge_score = 0.0
            total_reward = task_reward
            judge_failed = True
            judge_failed_rows += 1

        new_record = dict(record)
        new_record.update(
            {
                "reward": round(total_reward, 4),
                "task_reward": round(task_reward, 4),
                "judge_score": round(judge_score, 4),
                "turn_scores": ts_info,
                "judge_failed": judge_failed,
                "judge_model": judge_model,
                "reward_source": f"{source_tag}_rescore",
                "rescore_input": str(source_input),
            }
        )
        out_records.append(new_record)

    return out_records, judge_failed_rows


def process_group(
    key: GroupKey,
    indexed_records: List[IndexedRecord],
    problem_by_id: Dict[str, MuSiQueProblem],
    args: argparse.Namespace,
) -> Tuple[GroupKey, List[Dict[str, Any]], int, Optional[str]]:
    result = reconstruct_result(key, indexed_records, problem_by_id)
    try:
        reward = judge_result(
            result,
            api_base=args.judge_api_base,
            api_key=args.judge_api_key,
            model=args.judge_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
        if reward is None and args.strict:
            raise RuntimeError("judge returned no parseable turn scores")
        error = None if reward is not None else "judge returned no parseable turn scores"
    except Exception as exc:
        if args.strict:
            raise
        reward = None
        error = str(exc)

    out_records, judge_failed_rows = rescore_records(
        indexed_records,
        result,
        reward,
        alpha=args.alpha,
        judge_model=args.judge_model,
        source_input=args.input,
    )
    return key, out_records, judge_failed_rows, error


def write_records(handle, records: Iterable[Dict[str, Any]]) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    handle.flush()


def finite_mean(values: List[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    args.concurrency = max(1, int(args.concurrency))
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    print("=" * 64)
    print("JCA RL Rollout Rescore")
    print(f"input:       {args.input}")
    print(f"output:      {args.output}")
    print(f"split:       {args.split}")
    print(f"judge:       {args.judge_model} @ {args.judge_api_base}")
    print(f"alpha:       {args.alpha}")
    print(f"concurrency: {args.concurrency}")
    print("=" * 64)

    indexed_records = read_input_records(args.input)
    groups = group_records(indexed_records)
    group_sizes = {key: len(value) for key, value in groups.items()}
    ordered_keys = sorted(groups, key=lambda key: groups[key][0][0])

    print(f"loaded records={len(indexed_records)} groups={len(groups)}")
    print(f"loading MuSiQue {args.split} from {args.data_dir} ...")
    problems = load_musique(args.split, data_dir=args.data_dir)
    problem_by_id = {problem.id: problem for problem in problems}
    print(f"loaded problems={len(problem_by_id)}")

    if args.dry_run:
        sample_keys = ordered_keys[: args.limit_groups or 3]
        for key in sample_keys:
            result = reconstruct_result(key, groups[key], problem_by_id)
            prompt = _build_user_prompt(
                problem=result["problem"],
                steps=result["trajectory"]["steps"],
                final_answer=result["final_answer"],
                is_correct=float(result.get("f1", 0.0) or 0.0) > 0.0,
            )
            print(
                f"dry-run group={key} turns={len(groups[key])} "
                f"f1={result.get('f1')} prompt_chars={len(prompt)}"
            )
        print("dry-run complete; no output written")
        return

    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; use --resume or --overwrite"
        )

    completed_keys: set[GroupKey] = set()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    elif args.resume and args.output.exists():
        completed_keys, retained_lines, pruned = load_completed_output(
            args.output,
            group_sizes,
        )
        rewrite_retained_output(args.output, retained_lines)
        print(
            f"resume: complete_groups={len(completed_keys)} "
            f"retained_rows={len(retained_lines)} pruned_rows={pruned}"
        )

    pending_keys = [key for key in ordered_keys if key not in completed_keys]
    if args.limit_groups is not None:
        pending_keys = pending_keys[: args.limit_groups]
    print(f"pending groups={len(pending_keys)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_groups = len(pending_keys)
    completed = 0
    written_rows = 0
    judge_failed_rows = 0
    errors = 0
    rewards: List[float] = []
    start_time = time.time()

    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_key = {
                executor.submit(
                    process_group,
                    key,
                    groups[key],
                    problem_by_id,
                    args,
                ): key
                for key in pending_keys
            }

            for future in as_completed(future_to_key):
                key, out_records, failed_rows, error = future.result()
                write_records(handle, out_records)
                completed += 1
                written_rows += len(out_records)
                judge_failed_rows += failed_rows
                if error:
                    errors += 1
                    print(f"[judge-warning] group={key}: {error}")
                rewards.extend(float(r["reward"]) for r in out_records)

                if completed == 1 or completed % 10 == 0 or completed == total_groups:
                    elapsed = time.time() - start_time
                    avg = elapsed / max(completed, 1)
                    eta = avg * max(total_groups - completed, 0)
                    print(
                        f"progress {completed}/{total_groups} "
                        f"rows={written_rows} "
                        f"judge_failed_rows={judge_failed_rows} "
                        f"reward_mean={finite_mean(rewards):.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                    )

    print("=" * 64)
    print("Rescore complete")
    print(f"output:            {args.output}")
    print(f"new groups:        {completed}")
    print(f"new rows:          {written_rows}")
    print(f"judge error groups:{errors}")
    print(f"judge failed rows: {judge_failed_rows}")
    print(f"reward mean:       {finite_mean(rewards):.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
