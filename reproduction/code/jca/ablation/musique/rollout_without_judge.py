"""Generate MuSiQue trajectories without calling an external judge."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT.parent, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    from jca.src import judge

    judge.JUDGE_MAX_TOKENS = 16384
    judge.JUDGE_REASONING_EFFORT = "low"
    judge.JUDGE_PARSE_RETRIES = 2
    import rl_rollout

    def skip_judge(*_args, **_kwargs):
        return None

    rl_rollout.judge_trajectory = skip_judge
    sys.argv.append("--allow-judge-failure")
    # Raw trajectories are intentionally rescored offline.  Their missing
    # online judge must therefore remain valid when a rollout is resumed.
    sys.argv.append("--resume-keep-judge-failed")
    rl_rollout.main()


if __name__ == "__main__":
    main()
