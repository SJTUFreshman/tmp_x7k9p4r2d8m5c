"""Run the unchanged AFlow MCTS-lite search on GSM-Hard."""

from __future__ import annotations

from aflow_gsmhard_adapter import THIS_DIR, configure


configure()

from gsmhard_common import load_gsm_compatible  # noqa: E402
import run_search as musique_runner  # noqa: E402


musique_runner._HERE = THIS_DIR
musique_runner.load_musique = load_gsm_compatible


if __name__ == "__main__":
    musique_runner.main()
