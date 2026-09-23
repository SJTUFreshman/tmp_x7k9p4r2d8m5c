"""MATH AFlow initial workflow: Solve x3 -> ensemble -> answer cleanup."""

import asyncio


async def workflow(problem, ops):
    candidates = await asyncio.gather(
        ops.solve(problem, instruction="Derive a complete solution and check all conditions."),
        ops.solve(problem, instruction="Solve independently using an alternate route and audit arithmetic."),
        ops.solve(problem, instruction="Focus on edge cases, signs, simplification, and final answer form."),
    )
    decision = await ops.ensemble(list(candidates), problem)
    chosen = candidates[decision["chosen_index"]]
    final = await ops.answer_generate(chosen["answer"])
    return final["answer"]
