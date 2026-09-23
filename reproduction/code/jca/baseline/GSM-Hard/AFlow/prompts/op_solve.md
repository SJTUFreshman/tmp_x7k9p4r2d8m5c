# AFlow Solve Operator — GSM-Hard

Solve the numerical word problem using the optional instruction below.
Preserve all quantities exactly as written, derive the equation, track units,
and check signs, scale, decimal placement, and arithmetic.

Optional instruction: {INSTRUCTION}

Return only:

{
  "reasoning": "<concise step-by-step derivation>",
  "answer": "<one final numeric value>"
}

Use no units, prose, or `\boxed{}` in `answer`.
