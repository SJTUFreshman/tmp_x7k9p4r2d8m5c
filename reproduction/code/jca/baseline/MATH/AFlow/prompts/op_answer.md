# AFlow AnswerGenerate Operator — MATH

Extract and clean the final mathematical answer from the supplied text. Preserve its mathematical value and required structure.

Return exactly one JSON object and nothing else:

{"answer":"<short final mathematical expression>"}

The answer must be non-empty and contain no prose or `\\boxed{}` wrapper.
