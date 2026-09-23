import asyncio


async def workflow(problem, ops):
    # Generate three initial candidate answers with different instructions
    candidates = await asyncio.gather(
        ops.solve(problem, instruction=""),
        ops.solve(problem, instruction="Focus on temporal and geographic details."),
        ops.solve(problem, instruction="Verify each hop against the cited paragraphs.")
    )
    
    # Generate additional candidates with more specific instructions
    more_candidates = await asyncio.gather(
        ops.solve(problem, instruction="Check for numerical answers and ensure proper formatting."),
        ops.solve(problem, instruction="Ensure the answer matches the question's exact requirements.")
    )
    
    # Combine all candidates and perform ensemble selection
    all_candidates = candidates + more_candidates
    ensembled = await ops.ensemble(all_candidates, problem)
    chosen = all_candidates[ensembled["chosen_index"]]
    
    # Generate final answer with answer generation
    normalized = await ops.answer_generate(chosen["answer"])
    
    # Add a final review step to check for consistency and formatting
    final_review = await ops.solve(problem, instruction=f"Review the answer: {normalized['answer']}. Check for consistency with the question and proper formatting.")
    final_normalized = await ops.answer_generate(final_review["answer"])
    
    # Add an additional consistency check between the original chosen answer and the final review
    consistency_check = await ops.solve(problem, instruction=f"Compare the original answer ({chosen['answer']}) with the final answer ({final_normalized['answer']}). Are they consistent?")
    consistency_normalized = await ops.answer_generate(consistency_check["answer"])
    
    # Add a final validation step to ensure the answer is properly formatted and matches the question
    final_validation = await ops.solve(problem, instruction=f"Validate the final answer: {consistency_normalized['answer']}. Ensure it is properly formatted and matches the question's requirements.")
    final_answer = await ops.answer_generate(final_validation["answer"])
    
    # Add a final step to check for numerical formatting and entity specificity
    numerical_check = await ops.solve(problem, instruction=f"Check if the final answer ({final_answer['answer']}) contains proper numerical formatting and specific entities as required by the question.")
    final_output = await ops.answer_generate(numerical_check["answer"])
    
    # Add a step to explicitly compare the answer with the question to ensure relevance
    relevance_check = await ops.solve(problem, instruction=f"Check if the final answer ({final_output['answer']}) is directly relevant to the question: {problem.rendered_text}.")
    relevance_normalized = await ops.answer_generate(relevance_check["answer"])
    
    # Add a step to check for exact entity matching and proper answer formatting
    entity_check = await ops.solve(problem, instruction=f"Check if the final answer ({relevance_normalized['answer']}) exactly matches the required entity and is properly formatted.")
    entity_normalized = await ops.answer_generate(entity_check["answer"])
    
    return entity_normalized["answer"]