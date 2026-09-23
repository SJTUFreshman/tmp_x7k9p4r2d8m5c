# AFlow Solve Operator — MultiPL-E-8Lang

Produce a complete source-code continuation for the task in the user message.

Special instruction: {INSTRUCTION}

The answer is appended exactly at the source cursor. Do not repeat the prefix,
signature, imports, specification, or tests. Respect target-language syntax,
indentation, stop tokens, and the evaluator-provided closing delimiter. Do not
include Markdown fences, prose, or `<think>` in the answer field.

Return only valid JSON:

{"reasoning":"brief algorithm and boundary analysis","answer":"source-code continuation"}
