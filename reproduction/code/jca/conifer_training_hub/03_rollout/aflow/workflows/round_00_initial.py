"""Initial AFlow workflow for Conifer: Solve x 3 -> ScEnsemble -> AnswerGenerate.

This is the hand-written starting point for the MCTS search (round 0). It is
deliberately equivalent to the hardcoded 5-op graph in
`run_conifer_mas.py::_run_baseline_aflow`, so round 0 of the search reproduces
the previously reported Conifer AFlow number and every later round is measured
against it.

The three solve instructions mirror the roles that graph used: an independent
draft, a coverage audit, and an edge-case/constraint stress test.
"""

import asyncio


async def workflow(problem, ops):
    candidates = await asyncio.gather(
        ops.solve(
            problem,
            instruction="Extract every explicit content and format requirement, then answer.",
        ),
        ops.solve(
            problem,
            instruction="Write a complete answer and audit factual and requirement coverage.",
        ),
        ops.solve(
            problem,
            instruction="Answer, stress-test edge cases, and satisfy every stated constraint.",
        ),
    )
    ensembled = await ops.ensemble(list(candidates), problem)
    chosen = candidates[ensembled["chosen_index"]]
    finalized = await ops.answer_generate(chosen["answer"])
    return finalized["answer"]
