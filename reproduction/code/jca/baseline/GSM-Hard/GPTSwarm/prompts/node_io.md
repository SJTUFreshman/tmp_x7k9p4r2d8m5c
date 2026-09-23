# GPTSwarm IO Node — GSM-Hard

Solve the numerical word problem independently and quickly. Extract the
quantities exactly as written, identify the requested value, and perform a
short arithmetic check. Large, small, negative, decimal, and scientific-
notation values must be preserved exactly.

Return only this JSON object:

{
  "reasoning": "<one to three concise sentences>",
  "answer": "<one decimal or scientific-notation numeric string>"
}

The answer must contain only a decimal or scientific-notation number, with no
fraction expression, arithmetic expression, units, prose, or `\boxed{}`.
Never leave it empty.
