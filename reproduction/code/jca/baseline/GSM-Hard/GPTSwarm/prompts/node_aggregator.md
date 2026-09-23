# GPTSwarm Aggregator Node — GSM-Hard

Produce the final answer by synthesizing the predecessor solutions. Verify
the governing equation, unit consistency, sign, scale, decimal placement,
and arithmetic. Agreement is useful but not decisive; choose the value backed
by the strongest independently checked derivation.

Return only this JSON object:

{
  "reasoning": "<concise final synthesis and verification>",
  "answer": "<one decimal or scientific-notation numeric string>"
}

Return exactly one decimal or scientific-notation number in `answer`, with no
fraction expression, arithmetic expression, units, prose, alternatives, or
`\boxed{}`. Never leave it empty.
