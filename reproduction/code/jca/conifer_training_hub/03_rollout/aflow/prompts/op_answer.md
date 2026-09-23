# AFlow Operator — AnswerGenerate — Conifer

You are given the chosen answer for an open-ended instruction. Return the
final deliverable text.

Unlike short-answer tasks, there is nothing to "extract" here: the answer IS
the deliverable. Your job is to hand it back clean.

# What to Do
- Strip meta-commentary: "Here is the answer", "Sure!", "I hope this helps",
  notes about the instruction, and any leftover JSON or code fences.
- Preserve the body verbatim otherwise. Keep list markers, table pipes,
  headings, line breaks, and paragraph structure exactly as they are —
  formatting is part of what is being checked.
- Do NOT summarize, shorten, expand, or reword. Do NOT add a preamble or a
  closing line. Do NOT add or remove list items.
- If the input is already clean, return it unchanged.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "answer": "<the final deliverable text>"
  }

# Field Rules
- "answer" is non-empty and carries the original formatting as literal
  newlines inside the JSON string.
- Output ONLY the JSON object. No prose, no code fences.
