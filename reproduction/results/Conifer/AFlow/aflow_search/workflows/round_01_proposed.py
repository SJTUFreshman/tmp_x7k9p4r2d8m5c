import asyncio


async def workflow(problem, ops):
    # Generate three initial candidate answers with focused instructions
    candidates = await asyncio.gather(
        ops.solve(
            problem,
            instruction="Extract every explicit content and format requirement, then answer."
        ),
        ops.solve(
            problem,
            instruction="Write a complete answer and audit factual and requirement coverage."
        ),
        ops.solve(
            problem,
            instruction="Answer, stress-test edge cases, and satisfy every stated constraint."
        )
    )
    
    # Generate a fourth candidate specifically focused on format compliance
    format_candidate = await ops.solve(
        problem,
        instruction="Ensure the answer strictly follows the requested format and all constraints."
    )
    candidates.append(format_candidate)
    
    # Ensemble selection from all four candidates
    ensembled = await ops.ensemble(list(candidates), problem)
    chosen = candidates[ensembled["chosen_index"]]
    
    # Finalize the answer with answer generation
    finalized = await ops.answer_generate(chosen["answer"])
    return finalized["answer"]