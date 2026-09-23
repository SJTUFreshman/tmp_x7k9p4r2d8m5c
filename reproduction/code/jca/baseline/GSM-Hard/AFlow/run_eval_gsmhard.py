"""Evaluate an AFlow workflow on GSM-Hard with numeric EM/F1."""

from __future__ import annotations

from aflow_gsmhard_adapter import THIS_DIR, configure


configure()

from gsmhard_common import (  # noqa: E402
    error_recording_executor,
    load_gsm_compatible,
    problem_payload,
)
from jca.gsm.src.grader import compute_em_f1, is_correct  # noqa: E402
import run_eval as musique_runner  # noqa: E402


musique_runner._HERE = THIS_DIR
musique_runner.load_musique = load_gsm_compatible
musique_runner.compute_em_f1 = compute_em_f1
musique_runner.is_correct = is_correct


def _error_record(problem, exc):
    return {
        "problem": problem_payload(problem),
        "final_answer": None,
        "correct": False,
        "em": 0.0,
        "f1": 0.0,
        "n_ops": 0,
        "op_kinds": [],
        "error": f"{type(exc).__name__}: {exc}",
        "wall_time_s": 0.0,
    }


musique_runner.ThreadPoolExecutor = error_recording_executor(_error_record)


if __name__ == "__main__":
    musique_runner.main()
