# GPTSwarm CoT Node — GSM-Hard

Solve the numerical word problem step by step. Preserve every quantity,
derive the required equation, track units, and verify signs, scale, decimal
placement, and arithmetic before choosing the final value.

Return only this JSON object:

{
  "reasoning": "<concise step-by-step derivation and verification>",
  "answer": "<one decimal or scientific-notation numeric string>"
}

The answer must contain only a decimal or scientific-notation number, with no
fraction expression, arithmetic expression, units, prose, or `\boxed{}`.
Never leave it empty.
