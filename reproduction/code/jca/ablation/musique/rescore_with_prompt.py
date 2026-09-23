"""Run the repository rescorer with an archived MuSiQue judge prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT.parent, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--judge-prompt",
        required=True,
        choices=["original", "a3_confirm"],
    )
    args, remaining = parser.parse_known_args()

    from jca.src import judge, judge_original
    import rl_rescore_rollout

    prompt_module = judge_original if args.judge_prompt == "original" else judge
    rl_rescore_rollout.CORRECT_BONUS = prompt_module.CORRECT_BONUS
    rl_rescore_rollout._JUDGE_SYSTEM = prompt_module._JUDGE_SYSTEM
    rl_rescore_rollout._build_user_prompt = prompt_module._build_user_prompt
    sys.argv = [sys.argv[0], *remaining]
    rl_rescore_rollout.main()


if __name__ == "__main__":
    main()

