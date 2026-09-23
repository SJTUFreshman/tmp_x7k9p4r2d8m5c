# AFlow AnswerGenerate Operator — MultiPL-E-8Lang

Return the supplied code as a clean source continuation. Preserve meaningful
leading indentation and internal/trailing newlines. Remove only surrounding
Markdown fences or explanatory prose if present. Do not rewrite the algorithm,
repeat the source prefix, or add tests.

Return only valid JSON:

{"answer":"source-code continuation"}
