from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [row["problem_id"] for row in rows]
    assert len(identifiers) == len(set(identifiers)), f"Duplicate IDs in {path}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root")
    args = parser.parse_args()
    sources = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in (("original", args.original), ("recovery", args.recovery), ("grader", args.grader))
    }
    if args.remote_root:
        remote_root = args.remote_root.rstrip("/")
        sources["original"]["remote_path"] = remote_root + "/runs/math/paracloud_parallel_160136/eval/last/results_magrpo.jsonl"
        sources["recovery"]["remote_path"] = remote_root + "/runs/math/paracloud_parallel_160136/eval/recover_protocol/results_magrpo.jsonl"
        sources["grader"]["remote_path"] = remote_root + "/vendor/math_eval_v4.py"
    spec = importlib.util.spec_from_file_location("packaged_math_eval", args.grader)
    grader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = grader
    spec.loader.exec_module(grader)
    original = read_rows(args.original)
    recovery = read_rows(args.recovery)
    recovery_by_id = {row["problem_id"]: row for row in recovery}
    replace_ids = [row["problem_id"] for row in original if row["final_answer"] is None]
    assert len(original) == 500
    assert len(recovery) == len(replace_ids) == 96
    assert set(replace_ids) == set(recovery_by_id), "Recovery must match exactly the original null-answer IDs"
    assert sum(row["score"] for row in original) == 264
    assert sum(row["score"] for row in recovery) == 61
    merged = []
    retained_original_f1 = []
    for original_row in original:
        replacing = original_row["final_answer"] is None
        source_row = recovery_by_id[original_row["problem_id"]] if replacing else original_row
        row = copy.deepcopy(source_row)
        detail = row["detail"]
        if row["final_answer"]:
            extracted = detail["extracted"]
            gold = detail["gold"]
            f1, eligible, numeric_f1 = grader.math_soft_f1(extracted, gold)
            if "f1" in detail:
                assert math.isclose(float(detail["f1"]), f1, rel_tol=1e-12, abs_tol=1e-12)
            additions = {
                "f1": float(f1),
                "f1_numeric_eligible": bool(eligible),
                "f1_numeric": float(numeric_f1) if numeric_f1 is not None else None,
                "f1_grader": grader.MATH_SOFT_F1_VERSION,
            }
        else:
            additions = {"f1": 0.0, "f1_numeric_eligible": False, "f1_numeric": None, "f1_grader": grader.MATH_SOFT_F1_VERSION}
        detail.update(additions)
        detail.setdefault("em", float(row["score"]))
        assert detail["em"] == row["score"]
        if not replacing:
            assert row["final_answer"] == original_row["final_answer"]
            assert row["score"] == original_row["score"]
            assert all(detail[key] == value for key, value in original_row["detail"].items())
            retained_original_f1.append(detail["f1"])
        merged.append(row)
    assert [row["problem_id"] for row in merged] == [row["problem_id"] for row in original]
    correct = sum(row["score"] for row in merged)
    assert correct == 325
    numeric = [row["detail"]["f1_numeric"] for row in merged if row["detail"]["f1_numeric"] is not None]
    remaining_null_ids = [row["problem_id"] for row in merged if row["final_answer"] is None]
    assert len(remaining_null_ids) == 2
    summary = {
        "task": "math", "headline_metric": "em", "iteration": "last",
        "evaluation_method": "original 404 answered records retained; only original 96 protocol failures rerun and replaced",
        "arms": {"magrpo": {
            "em": correct / len(merged),
            "f1": fmean(row["detail"]["f1"] for row in merged),
            "f1_numeric": fmean(numeric),
            "f1_numeric_n": len(numeric),
            "n": len(merged), "correct": int(correct),
            "answered": sum(bool(row["final_answer"]) for row in merged),
            "remaining_null": len(remaining_null_ids),
            "grader": grader.MATH_EVAL_VERSION, "f1_grader": grader.MATH_SOFT_F1_VERSION,
        }},
        "original_magrpo": {"n": 500, "correct": 264, "answered": 404, "em": 264 / 500, "f1": sum(retained_original_f1) / 500},
        "recovery_subset": {"n": 96, "correct": 61, "answered": 94, "em": 61 / 96, "f1": fmean(row["detail"]["f1"] for row in recovery)},
        "comparison_note": "Original base was evaluated under the original strict protocol; do not treat it as a fair delta against recovered MAGRPO.",
    }
    for name, path in (("original", args.original), ("recovery", args.recovery), ("grader", args.grader)):
        assert sha256(path) == sources[name]["sha256"], f"Source changed during merge: {name}"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "script_sha256": sha256(Path(__file__)),
        "counts": {"original": 500, "recovery": 96, "replaced": 96, "preserved_answer_and_score": 404, "merged": 500, "remaining_null": 2},
        "replaced_problem_ids": replace_ids,
        "remaining_null_problem_ids": remaining_null_ids,
        "checks": {"unique_source_ids": True, "exact_failed_id_match": True, "original_order_preserved": True, "original_answer_and_score_preserved": True, "existing_detail_fields_preserved": True, "recovery_f1_recomputed_and_matched": True, "264_plus_61_equals_325": True, "source_hashes_unchanged": True},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    results_path = args.output / "results_magrpo.jsonl"
    with results_path.open("x", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    manifest["outputs"] = {path.name: {"sha256": sha256(path)} for path in (results_path, summary_path)}
    with (args.output / "merge_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
