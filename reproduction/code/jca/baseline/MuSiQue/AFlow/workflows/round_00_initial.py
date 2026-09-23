"""Initial AFlow workflow for MuSiQue: Solve x 3 -> ScEnsemble -> AnswerGenerate.

This is the hand-written starting point for the MCTS search (round 0). It
mimics the "AFlow-initial" workflow reported in the AFlow paper: draw N
candidates via `solve`, pick the best via `ensemble`, then normalize the
final answer via `answer_generate`.
"""

import asyncio


async def workflow(problem, ops):
    candidates = await asyncio.gather(
        ops.solve(problem, instruction=""),
        ops.solve(problem, instruction="Focus on temporal and geographic details."),
        ops.solve(problem, instruction="Verify each hop against the cited paragraphs."),
    )
    ensembled = await ops.ensemble(list(candidates), problem)
    chosen = candidates[ensembled["chosen_index"]]
    normalized = await ops.answer_generate(chosen["answer"])
    return normalized["answer"]
