"""GSM-Hard AFlow initial workflow: Solve x3 -> ensemble -> normalize."""

import asyncio


async def workflow(problem, ops):
    candidates = await asyncio.gather(
        ops.solve(problem, instruction="Derive the equation directly and track units."),
        ops.solve(problem, instruction="Solve independently and audit sign and scale."),
        ops.solve(problem, instruction="Recompute using an alternate route and check arithmetic."),
    )
    ensembled = await ops.ensemble(list(candidates), problem)
    chosen = candidates[ensembled["chosen_index"]]
    normalized = await ops.answer_generate(chosen["answer"])
    return normalized["answer"]
