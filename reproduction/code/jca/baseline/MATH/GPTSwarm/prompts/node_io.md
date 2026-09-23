# GPTSwarm Node — IO — MATH

Solve the mathematics problem directly and efficiently. Check the essential arithmetic and algebra, but keep the visible derivation concise. If a direct predecessor is provided, use it as a candidate to verify rather than as unquestioned truth.

Return exactly one JSON object and nothing else:

{"reasoning":"<concise mathematical derivation>","answer":"<short final mathematical expression>"}

Both fields must be non-empty. The answer must contain only the final mathematical object, without prose or `\\boxed{}`.
