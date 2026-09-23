"""Run the MultiPL-E image using the validated Zhongwei Apptainer command.

The input/output CLI matches evaluate_multipl_e_parallel.py. Container
parallelism is controlled by --inner-workers; --shards is accepted for CLI
compatibility. A local image is required, with no registry access at run time.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker-exec", default="docker")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--inner-workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args(argv)
    image = os.environ.get("MULTIPLE_APPTAINER_IMAGE", options.image)
    if not Path(image).is_file():
        raise RuntimeError(f"MultiPL-E requires a local Apptainer image: {image}")
    if not options.input_dir.is_dir():
        raise RuntimeError(f"MultiPL-E input directory not found: {options.input_dir}")
    if options.inner_workers < 1 or options.shards < 1:
        raise ValueError("worker counts must be positive")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("APPTAINER_BIN", "apptainer"), "exec",
        "--containall", "--cleanenv",
        "--no-mount", "hostfs,bind-paths",
        "--bind", f"{options.input_dir.resolve()}:/inputs:ro",
        "--bind", f"{options.output_dir.resolve()}:/outputs:rw",
        str(Path(image).resolve()),
        "python3", "/code/main.py",
        "--dir", "/inputs", "--output-dir", "/outputs", "--recursive",
        "--max-workers", str(options.inner_workers),
    ]
    if not options.quiet:
        print("running:", " ".join(command), flush=True)
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        raise RuntimeError(f"failed to start Apptainer: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
