"""Initial MultiPL-E AFlow workflow: Solve x3 -> Ensemble -> AnswerGenerate."""

import asyncio


async def workflow(problem, ops):
    candidates = await asyncio.gather(
        ops.solve(problem, instruction="Implement directly and obey the cursor boundary."),
        ops.solve(problem, instruction="Audit algorithm, types, complexity, and edge cases."),
        ops.solve(problem, instruction="Independently solve and verify target-language syntax."),
    )
    ensembled = await ops.ensemble(list(candidates), problem)
    chosen = candidates[ensembled["chosen_index"]]
    normalized = await ops.answer_generate(chosen["answer"])
    return normalized["answer"]
