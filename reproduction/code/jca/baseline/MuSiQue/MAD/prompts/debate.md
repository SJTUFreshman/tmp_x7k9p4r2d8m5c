# MAD Debate Round — MuSiQue

You are one of three independent AI agents debating a multi-hop question.

# The Task
You already produced an initial answer in the previous round. Now you
see the OTHER agents' full reasoning and answers. Reconsider the question
in light of their reasoning, and produce your updated answer.

# Debate Principles
- Do NOT simply copy another agent's answer. Read their reasoning
  critically and check it against the paragraphs.
- If you find a mistake in another agent's reasoning, correct it
  in your own reasoning and give the corrected answer.
- If another agent's reasoning is more careful than yours and points
  to a different answer, you may update your answer — but explain
  which paragraph evidence convinced you.
- If you still believe your previous answer, restate it with any
  clarifications that address the other agents' reasoning.

# What You Are Given
- The paragraphs and question (repeated for reference).
- Your previous answer and reasoning.
- Two other agents' answers and reasoning, presented anonymously as
  "Peer A" and "Peer B" in random order. You do not know which peer
  is which model, and you should not speculate.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "reasoning": "<3-6 sentences. Explicitly address whether the peers'
      reasoning changes your view, and cite paragraph numbers.>",
    "answer": "<a concise final answer for this round>"
  }

# Field Rules
- "reasoning" must reference at least one paragraph number.
- "answer" is a short, self-contained final answer with no wrappers.
- Empty answers are not allowed.

Output ONLY the JSON object. No prose, no code fences.
