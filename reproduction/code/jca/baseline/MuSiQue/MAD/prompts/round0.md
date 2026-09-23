# MAD Round 0 (Independent Answer) — MuSiQue

You are one of three independent AI agents answering a multi-hop question.

# The Task
You are given a set of paragraphs and a question. The answer requires
combining information from multiple paragraphs. Read the paragraphs
carefully and produce your best answer independently.

# This Round
This is the INDEPENDENT round. You do not see any other agent's answer.
Reason from the paragraphs alone.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "reasoning": "<3-6 sentences of reasoning, citing paragraph numbers>",
    "answer": "<a concise answer: a number, entity name, or short phrase>"
  }

# Field Rules
- "reasoning" cites paragraph numbers like [3], [11] to show which
  paragraphs support each inference step.
- "answer" is a short, self-contained final answer with no wrappers
  (no \boxed{}, no "The answer is ...", just the value).
- If you truly cannot determine the answer, still give your best guess
  based on the paragraphs. Empty answers are not allowed.

Output ONLY the JSON object. No prose, no code fences.
