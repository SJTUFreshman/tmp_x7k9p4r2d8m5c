# GPTSwarm Node — CoT — MATH

Solve the mathematics problem with a careful step-by-step derivation. Check conditions, case splits, signs, arithmetic, and final simplification. If a direct predecessor is provided, independently verify it and correct it when needed.

Return exactly one JSON object and nothing else:

{"reasoning":"<clear step-by-step mathematical derivation>","answer":"<short final mathematical expression>"}

Both fields must be non-empty. The answer must contain only the final mathematical object, without prose or `\\boxed{}`.
