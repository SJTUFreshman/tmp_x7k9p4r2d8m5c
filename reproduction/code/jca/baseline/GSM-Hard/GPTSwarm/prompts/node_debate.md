# GPTSwarm Debate Node — GSM-Hard

Critique the predecessor solutions to the numerical word problem. Check each
equation against the question and inspect unit conversions, signs, magnitude,
decimal placement, and arithmetic. Resolve disagreements using a verified
derivation rather than majority alone, then produce your own final answer.

Return only this JSON object:

{
  "reasoning": "<concise comparison and corrected derivation>",
  "answer": "<one decimal or scientific-notation numeric string>"
}

Reference concrete predecessor errors or agreements in `reasoning`. The
answer must contain only a decimal or scientific-notation number, with no
fraction expression, arithmetic expression, units, prose, or `\boxed{}`.
