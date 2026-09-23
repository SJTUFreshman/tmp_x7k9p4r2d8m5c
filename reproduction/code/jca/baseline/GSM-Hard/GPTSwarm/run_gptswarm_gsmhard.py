"""Run the unchanged MuSiQue GPTSwarm-Fixed protocol on GSM-Hard."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GSM_BASELINE_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "GPTSwarm"
for path in (PACKAGE_PARENT, GSM_BASELINE_DIR, MUSIQUE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gsmhard_common import (  # noqa: E402
    error_recording_executor,
    load_gsm_compatible,
    patch_musique_numeric_normalizer,
    problem_payload,
)
from jca.gsm.src.data import format_problem_as_prompt  # noqa: E402
from jca.gsm.src.grader import compute_em_f1, is_correct  # noqa: E402

import nodes as nodes_core  # noqa: E402
import run_gptswarm as musique_runner  # noqa: E402
import swarm as swarm_core  # noqa: E402


patch_musique_numeric_normalizer()
nodes_core.PROMPT_IO_PATH = THIS_DIR / "prompts" / "node_io.md"
nodes_core.PROMPT_COT_PATH = THIS_DIR / "prompts" / "node_cot.md"
nodes_core.PROMPT_DEBATE_PATH = THIS_DIR / "prompts" / "node_debate.md"
nodes_core.PROMPT_AGGREGATOR_PATH = THIS_DIR / "prompts" / "node_aggregator.md"


@classmethod
def _from_gsm(cls, problem):
    return cls(
        id=problem.id,
        question=problem.question,
        rendered_text=format_problem_as_prompt(problem),
    )


swarm_core.SwarmProblem.from_musique = _from_gsm
musique_runner.TASK_NAME = "GSM-Hard"
musique_runner.load_musique = load_gsm_compatible
musique_runner.compute_em_f1 = compute_em_f1
musique_runner.is_correct = is_correct


def _error_record(problem, exc):
    error = f"{type(exc).__name__}: {exc}"
    return {
        "problem": problem_payload(problem),
        "swarm": {
            "problem_id": problem.id,
            "layers": [],
            "node_outputs": [],
            "final_answer": None,
            "error": error,
            "wall_time_s": 0.0,
        },
        "final_answer": None,
        "correct": False,
        "em": 0.0,
        "f1": 0.0,
        "wall_time_s": 0.0,
    }


musique_runner.ThreadPoolExecutor = error_recording_executor(_error_record)

_base_parse_args = musique_runner.parse_args


def _parse_args():
    args = _base_parse_args()
    if not args.enable_thinking:
        raise SystemExit("GSM-Hard GPTSwarm evaluation requires --enable-thinking")
    if args.output is None:
        args.output = (
            THIS_DIR
            / "outputs"
            / f"{args.split}_start{args.start}_n{args.limit}.jsonl"
        )
    return args


musique_runner.parse_args = _parse_args


if __name__ == "__main__":
    musique_runner.main()
