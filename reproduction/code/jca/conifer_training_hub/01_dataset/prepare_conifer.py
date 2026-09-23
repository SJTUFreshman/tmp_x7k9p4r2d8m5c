#!/usr/bin/env python3
"""Normalize the public Conifer shard and make leakage-safe splits.

The Hugging Face release contains one ``train_sft`` split and no official
test split.  We therefore create deterministic *grouped* train/dev/test
holdouts.  All examples derived from the same seed prompt stay in one split;
the test file is an untouched local holdout, not an official Conifer score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "01_dataset/raw/Conifer/data/train_sft-00000-of-00001.parquet"
DEFAULT_OUTPUT = ROOT / "01_dataset/processed"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _group_id(prompt: str) -> str:
    normalized = _normalize_text(prompt).casefold()
    return "q_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _example_id(raw: dict[str, Any], row_index: int) -> str:
    # Include the complete row so repeated seed prompts remain separate
    # examples while still receiving the same group assignment.
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"cf_{row_index:06d}_{digest}"


def _extract_constraints(prompt: str) -> dict[str, Any]:
    """Extract conservative, auditable constraints from an instruction.

    These are hints for deterministic scoring, not an attempt to solve every
    natural-language requirement.  The LLM judge remains responsible for
    semantic/content quality.
    """
    text = _normalize_text(prompt)
    lower = text.casefold()
    result: dict[str, Any] = {
        "formats": [],
        "style_terms": [],
        "required_terms": [],
        "limits": {},
        "source": "heuristic_prompt_parser_v1",
    }

    format_patterns = [
        (r"\bbullet(?:ed)?\s+points?\b|\bbulleted\s+list\b", "bullets"),
        (r"\bordered\s+list\b|\bnumbered\s+list\b", "ordered_list"),
        (r"\blist\s+format\b|\bprovide a list\b", "list"),
        (r"\btable\b|\btabular\b", "table"),
        (r"\bparagraph\b", "paragraph"),
        (r"\bcode\s+snippet\b|\bcommented\s+code\b", "code"),
    ]
    for pattern, name in format_patterns:
        if re.search(pattern, lower):
            result["formats"].append(name)

    style_patterns = [
        "formal", "professional", "concise", "brief", "short", "to-the-point",
        "non-technical", "layperson", "plain language", "respectful", "friendly",
        "persuasive", "empathetic", "objective", "accessible",
    ]
    result["style_terms"] = [term for term in style_patterns if term in lower]

    word_match = re.search(
        r"(?:no more than|at most|maximum of|under|within)\s+(\d+)\s+words?",
        lower,
    )
    if word_match:
        result["limits"]["max_words"] = int(word_match.group(1))

    sentence_match = re.search(
        r"(?:exactly|at most|no more than)\s+(\d+)\s+sentences?",
        lower,
    )
    if sentence_match:
        key = "exact_sentences" if "exactly" in sentence_match.group(0) else "max_sentences"
        result["limits"][key] = int(sentence_match.group(1))

    item_match = re.search(
        r"(\d+)\s*(?:to|-|–)\s*(\d+)\s+(?:items?|examples?|points?)",
        lower,
    )
    if item_match:
        result["limits"]["min_items"] = int(item_match.group(1))
        result["limits"]["max_items"] = int(item_match.group(2))
    else:
        item_match = re.search(
            r"(?:a|an|the)?\s*(\d+)[- ]item\s+(?:list|answer)", lower
        )
        if item_match:
            result["limits"]["exact_items"] = int(item_match.group(1))

    # Explicit quoted terms are reliable enough to expose as hard checks.
    quoted = re.findall(r"[\"'‘“]([^\"'’”]{2,80})[\"'’”]", text)
    result["required_terms"] = [q.strip() for q in quoted if q.strip()]

    numbered_requirements = re.findall(
        r"(?:^|\n)\s*\d+[.)]\s*([^\n;]{3,160})", str(prompt)
    )
    # Keep only short noun-like requirements; long prose is too noisy for a
    # deterministic lexical check and is left to the judge.
    for requirement in numbered_requirements:
        cleaned = _normalize_text(requirement).rstrip(".")
        if 2 <= len(cleaned.split()) <= 12 and len(cleaned) <= 120:
            result.setdefault("numbered_requirements", []).append(cleaned)

    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, list) else list(loaded.values())
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Reading Conifer parquet requires pyarrow") from exc
    return pq.read_table(path).to_pylist()


def _normalize_row(raw: dict[str, Any], row_index: int) -> dict[str, Any]:
    prompt = _normalize_text(raw.get("prompt"))
    messages = raw.get("messages")
    if not prompt or not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index}: prompt/messages are missing")
    clean_messages: list[dict[str, str]] = []
    for msg_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"row {row_index} message {msg_index}: expected object")
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            raise ValueError(f"row {row_index} message {msg_index}: invalid role/content")
        clean_messages.append({"role": role, "content": content})
    if clean_messages[0]["role"] != "user" or clean_messages[-1]["role"] != "assistant":
        raise ValueError(f"row {row_index}: messages must start user and end assistant")
    if any(a["role"] == b["role"] for a, b in zip(clean_messages, clean_messages[1:])):
        raise ValueError(f"row {row_index}: messages must alternate user/assistant")

    pairs = [
        {"question": clean_messages[i]["content"], "answer": clean_messages[i + 1]["content"]}
        for i in range(0, len(clean_messages) - 1, 2)
    ]
    source_type = _normalize_text(raw.get("type")) or "unknown"
    group_id = _group_id(prompt)
    constraints = _extract_constraints(pairs[-1]["question"])
    return {
        "problem_id": _example_id(raw, row_index),
        "group_id": group_id,
        "source_row": row_index,
        "seed_prompt": prompt,
        "question": pairs[-1]["question"],
        "reference_answer": pairs[-1]["answer"],
        "source_type": source_type,
        "native_turns": pairs,
        "native_messages": clean_messages,
        "native_turn_count": len(pairs),
        "difficulty": len(pairs),
        "has_process_feedback": source_type.casefold() == "process feedback",
        "process_feedback": pairs[-2]["question"] if len(pairs) >= 2 and source_type.casefold() == "process feedback" else None,
        "constraints": constraints,
    }


def _assign_split(group_id: str, dev_fraction: float, test_fraction: float) -> str:
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    if value < test_fraction:
        return "test"
    if value < test_fraction + dev_fraction:
        return "dev"
    return "train"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            group = row["group_id"]
            if group in seen and seen[group] != split:
                raise AssertionError(f"group leakage: {group} in {seen[group]} and {split}")
            seen[group] = split


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Abort on malformed rows instead of recording and skipping them.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    if args.dev_fraction < 0 or args.test_fraction < 0 or args.dev_fraction + args.test_fraction >= 1:
        raise SystemExit("dev/test fractions must be non-negative and sum to less than 1")

    raw_rows = _read_rows(args.input)
    if args.max_rows > 0:
        raw_rows = raw_rows[: args.max_rows]
    normalized: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_rows):
        try:
            normalized.append(_normalize_row(raw, i))
        except (TypeError, ValueError, KeyError) as exc:
            if args.fail_on_invalid:
                raise
            invalid_rows.append({"source_row": i, "error": str(exc)})
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        splits[_assign_split(row["group_id"], args.dev_fraction, args.test_fraction)].append(row)
    _validate_splits(splits)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for split in ("train", "dev", "test") for row in splits[split]]
    _write_jsonl(args.out_dir / "all.jsonl", all_rows)
    for split in ("train", "dev", "test"):
        _write_jsonl(args.out_dir / f"{split}.jsonl", splits[split])

    type_counts = {split: dict(Counter(row["source_type"] for row in rows)) for split, rows in splits.items()}
    difficulty_counts = {split: dict(Counter(str(row["difficulty"]) for row in rows)) for split, rows in splits.items()}
    manifest = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "group_key": "casefolded whitespace-normalized seed prompt",
        "split_rule": "sha256(group_id): test then dev then train",
        "dev_fraction": args.dev_fraction,
        "test_fraction": args.test_fraction,
        "rows": len(normalized),
        "groups": len({row["group_id"] for row in normalized}),
        "split_rows": {split: len(rows) for split, rows in splits.items()},
        "split_groups": {split: len({row["group_id"] for row in rows}) for split, rows in splits.items()},
        "type_counts": type_counts,
        "difficulty_counts": difficulty_counts,
        "constraint_parser": "heuristic_prompt_parser_v1",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = {
        "rows": len(normalized),
        "groups": len({row["group_id"] for row in normalized}),
        "split_rows": manifest["split_rows"],
        "split_groups": manifest["split_groups"],
        "type_counts": type_counts,
        "difficulty_counts": difficulty_counts,
        "process_feedback_rows": sum(row["has_process_feedback"] for row in normalized),
        "format_constraint_rows": sum(bool(row["constraints"]["formats"]) for row in normalized),
        "max_word_limit_rows": sum("max_words" in row["constraints"]["limits"] for row in normalized),
        "invalid_rows": len(invalid_rows),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "invalid_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in invalid_rows),
        encoding="utf-8",
    )
    manifest["invalid_rows"] = len(invalid_rows)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
