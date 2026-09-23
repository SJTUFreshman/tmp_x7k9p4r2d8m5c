# MAD Round 0 — GSM-Hard

You are one of three independent agents solving a numerical word problem.
Work from the quantities exactly as written; they may be unusually large,
small, negative, or expressed in scientific notation.

Derive the required equation step by step, track units, and recompute the
arithmetic before answering. Return exactly one JSON object:

{
  "reasoning": "<concise step-by-step derivation and arithmetic check>",
  "answer": "<one final numeric value>"
}

The answer must contain only a number, with no units, prose, or `\boxed{}`.
Never leave the answer empty. Output only the JSON object.
