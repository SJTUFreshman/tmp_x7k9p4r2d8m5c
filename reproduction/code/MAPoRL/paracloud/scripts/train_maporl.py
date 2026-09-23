from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maporl.config import load_config
from maporl.loop import train


def main():
    parser = argparse.ArgumentParser(description="Train the MAPoRL collaboration policies and per-turn critics")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument(
        "--target-iterations", type=int,
        help="Run to this completed-iteration count, including values above the config limit",
    )
    parser.add_argument("--skip-periodic-eval", action="store_true")
    parser.add_argument(
        "--ignore-wall-clock-limit", action="store_true",
        help="Resume or run without the configured wall-clock stop (config hash is unchanged)",
    )
    options = parser.parse_args()
    train(
        load_config(options.config), options.run_dir, resume=options.resume,
        max_iterations=options.max_iterations, perform_periodic_eval=not options.skip_periodic_eval,
        ignore_wall_clock_limit=options.ignore_wall_clock_limit,
        target_iterations=options.target_iterations,
    )


if __name__ == "__main__":
    main()
