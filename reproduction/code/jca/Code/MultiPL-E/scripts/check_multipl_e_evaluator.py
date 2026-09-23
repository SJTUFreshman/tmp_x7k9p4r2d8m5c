#!/usr/bin/env python3
"""Validate the evaluator image before starting a MultiPL-E run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


REQUIRED_MODULES = (
    "eval_adb",
    "eval_clj",
    "eval_cpp",
    "eval_cs",
    "eval_dart",
    "eval_dlang",
    "eval_elixir",
    "eval_go",
    "eval_hs",
    "eval_java",
    "eval_javascript",
    "eval_julia",
    "eval_lua",
    "eval_ocaml",
    "eval_php",
    "eval_pl",
    "eval_r",
    "eval_racket",
    "eval_ruby",
    "eval_rust",
    "eval_scala",
    "eval_sh",
    "eval_swift",
    "eval_ts",
)

REQUIRED_COMMANDS = (
    "gnatchop",
    "gnatmake",
    "clojure",
    "g++",
    "csc",
    "mono",
    "rdmd",
    "dart",
    "elixir",
    "go",
    "runghc",
    "javac",
    "java",
    "julia",
    "node",
    "lua",
    "ocaml",
    "php",
    "perl",
    "Rscript",
    "ruby",
    "racket",
    "rustc",
    "scalac",
    "scala",
    "bash",
    "swiftc",
    "tsc",
)

REQUIRED_FILES = ("/usr/multiple/javatuples-1.2.jar",)

CHECK_CODE = """
import importlib.util
import json
import shutil

modules = %r
commands = %r
files = %r
print(json.dumps({
    "missing_modules": [name for name in modules if importlib.util.find_spec(name) is None],
    "missing_commands": [name for name in commands if shutil.which(name) is None],
    "missing_files": [path for path in files if not __import__("os").path.isfile(path)],
}))
""" % (REQUIRED_MODULES, REQUIRED_COMMANDS, REQUIRED_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check MultiPL-E evaluator modules and runtimes in a container."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker-exec", default="docker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        args.docker_exec,
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--entrypoint",
        "python3",
        args.image,
        "-c",
        CHECK_CODE,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"Evaluator preflight could not start {args.docker_exec}: {exc}", file=sys.stderr)
        return 2

    if completed.returncode != 0:
        output = (completed.stdout or "") + (completed.stderr or "")
        print(f"Evaluator preflight failed for image {args.image}:\n{output}", file=sys.stderr)
        return completed.returncode or 2

    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"Evaluator preflight returned invalid JSON: {completed.stdout!r}", file=sys.stderr)
        raise SystemExit(2) from exc

    failures = False
    for key, label in (
        ("missing_modules", "evaluator modules"),
        ("missing_commands", "runtime commands"),
        ("missing_files", "runtime files"),
    ):
        missing = report.get(key, [])
        if missing:
            failures = True
            print(f"Missing {label}: {', '.join(missing)}", file=sys.stderr)

    if failures:
        return 1
    print(
        f"Evaluator preflight passed: image={args.image} "
        f"modules={len(REQUIRED_MODULES)} commands={len(REQUIRED_COMMANDS)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
