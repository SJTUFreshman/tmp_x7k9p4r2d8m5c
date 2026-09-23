#!/usr/bin/env python3
"""Record complete MAPORL seed results without requiring a separate base evaluation."""
import argparse
import json
import math
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("ledger", type=pathlib.Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(args.summary.read_text(encoding="utf-8")).get("maporl", {})
    if metrics.get("seed") != args.seed or metrics.get("mode") != "maporl":
        raise ValueError("summary does not identify the requested MAPORL seed")
    if not isinstance(metrics.get("n"), int) or metrics["n"] < 1:
        raise ValueError("summary has no evaluated problems")
    for key in ("em", "f1"):
        if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
            raise ValueError(f"summary has no finite {key}")
    if args.check_only:
        return
    rows = []
    if args.ledger.exists():
        rows = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if any(row.get("seed") == args.seed for row in rows):
        print(f"[sweep] seed {args.seed} already in ledger, not re-appending", flush=True)
        return
    row = {
        "seed": args.seed,
        "temperature": metrics.get("temperature"),
        "n": metrics["n"],
        "maporl_em": metrics["em"],
        "maporl_f1": metrics["f1"],
        "maporl_answered": metrics.get("answered"),
        "path": str(args.summary),
    }
    with args.ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows.append(row)
    values = [row["maporl_em"] for row in rows if row.get("maporl_em") is not None]
    deviation = statistics.stdev(values) if len(values) > 1 else float("nan")
    print(f"[sweep] seed {args.seed} maporl_em={metrics['em']:.5f} maporl_f1={metrics['f1']:.5f}", flush=True)
    print(f"[sweep] MAPORL over {len(values)} seeds: mean_em={statistics.fmean(values):.5f} "
          f"std={deviation:.5f} se={deviation / len(values) ** 0.5:.5f}", flush=True)


if __name__ == "__main__":
    main()
