# AFlow Operator — AnswerGenerate — MuSiQue

You are given the final chosen solution for a multi-hop question.
Extract the concise final answer.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "answer": "<the concise final answer: number, entity name, or
      short phrase>"
  }

# Field Rules
- "answer" is the shortest correct form (e.g. "1995" not "the year 1995").
- No wrappers, no explanations, no \boxed{}.
- Output ONLY the JSON object. No prose, no code fences.
