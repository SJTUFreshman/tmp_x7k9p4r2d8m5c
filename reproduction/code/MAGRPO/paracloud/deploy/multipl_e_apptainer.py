#!/usr/bin/env python3
"""Run MultiPL-E evaluator image through Apptainer instead of Docker.

This preserves evaluate_multipl_e_parallel.py's CLI. The host supplies
MULTIPLE_APPTAINER_IMAGE or --image as a local .sif path.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--image", required=True)
p.add_argument("--docker-exec", default="docker")
p.add_argument("--shards", type=int, default=1)
p.add_argument("--inner-workers", type=int, default=1)
p.add_argument("--manifest")
p.add_argument("--keep-temp", action="store_true")
p.add_argument("--quiet", action="store_true")
p.add_argument("--timings-file")
a = p.parse_args()
image = os.environ.get("MULTIPLE_APPTAINER_IMAGE", a.image)
if image.startswith("docker://"):
    raise SystemExit("MultiPL-E Apptainer shim requires a local .sif image; set MULTIPLE_APPTAINER_IMAGE")
if not Path(image).is_file():
    raise SystemExit(f"Apptainer image not found: {image}")
a.output_dir.mkdir(parents=True, exist_ok=True)
cmd = [
    os.environ.get("APPTAINER_BIN", "apptainer"), "exec",
    "--containall", "--cleanenv",
    "--no-mount", "hostfs,bind-paths",
    "--bind", f"{a.input_dir.resolve()}:/inputs:ro",
    "--bind", f"{a.output_dir.resolve()}:/outputs:rw",
    image,
    "python3", "/code/main.py",
    "--dir", "/inputs", "--output-dir", "/outputs", "--recursive",
    "--max-workers", str(max(1, a.inner_workers)),
]
if not a.quiet:
    print("running:", " ".join(cmd), flush=True)
try:
    raise SystemExit(subprocess.run(cmd, check=False).returncode)
except OSError as exc:
    raise SystemExit(f"failed to start apptainer: {exc}")
