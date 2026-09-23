"""Run the unchanged MuSiQue MAD protocol on GSM-Hard."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GSM_BASELINE_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
PACKAGE_PARENT = PROJECT_ROOT.parent
MUSIQUE_DIR = PROJECT_ROOT / "baseline" / "MuSiQue" / "MAD"
for path in (PACKAGE_PARENT, GSM_BASELINE_DIR, MUSIQUE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gsmhard_common import (  # noqa: E402
    error_recording_executor,
    load_gsm_compatible,
    normalize_numeric_answer,
    patch_musique_numeric_normalizer,
    problem_payload,
)
from jca.gsm.src.grader import compute_em_f1, is_correct  # noqa: E402

import mad as mad_core  # noqa: E402
import run_mad as musique_runner  # noqa: E402


patch_musique_numeric_normalizer()
mad_core.format_problem_as_prompt = __import__(
    "jca.gsm.src.data", fromlist=["format_problem_as_prompt"]
).format_problem_as_prompt
mad_core.normalize_answer = normalize_numeric_answer
mad_core.PROMPT_ROUND0_PATH = THIS_DIR / "prompts" / "round0.md"
mad_core.PROMPT_DEBATE_PATH = THIS_DIR / "prompts" / "debate.md"

musique_runner.load_musique = load_gsm_compatible
musique_runner.compute_em_f1 = compute_em_f1
musique_runner.is_correct = is_correct


def _error_record(problem, exc):
    error = f"{type(exc).__name__}: {exc}"
    return {
        "problem": problem_payload(problem),
        "mad": {
            "error": error,
            "voter_answers": {},
            "tie_break": False,
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
    if args.output is None:
        args.output = (
            THIS_DIR
            / "outputs"
            / f"{args.split}_start{args.start}_n{args.limit}_r{args.n_rounds}.jsonl"
        )
    return args


musique_runner.parse_args = _parse_args


if __name__ == "__main__":
    musique_runner.main()
