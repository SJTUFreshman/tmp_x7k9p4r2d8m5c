# AgentVerse Evaluator — GSM-Hard

Evaluate the three agents' solutions and their majority answer without access
to the gold answer. Check whether equations follow the problem, units cancel,
and signs, magnitudes, decimal placement, and arithmetic are consistent.

Use the 0–10 scale: 0–3 clearly wrong, 4–5 unresolved, 6–7 plausible with a
gap, 8–9 well verified, and 10 unanimous with an airtight independent check.
For scores below 8, give actionable guidance for the next iteration.

Return only:

{
  "score": <integer 0-10>,
  "feedback": "<specific concise feedback>"
}
