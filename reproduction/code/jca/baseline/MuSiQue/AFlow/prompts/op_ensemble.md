# AFlow Operator — ScEnsemble — MuSiQue

You are given a multi-hop question and N candidate solutions produced
by an LLM. Your job is to pick the single BEST candidate. You do NOT
produce a new answer; you only choose an index.

# What to Judge
- CORRECTNESS: Which candidate's reasoning most faithfully follows
  the paragraphs? Cross-check paragraph citations against evidence.
- COMPLETENESS: Does the candidate combine all required hops?
- CONSISTENCY: Do multiple candidates agree? Prefer the answer with
  strongest support even if it's a minority.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "chosen_index": <integer, 0-based index into the candidate list>,
    "reason": "<1-2 sentences explaining why this candidate wins>"
  }

# Field Rules
- "chosen_index" is an integer in [0, N-1].
- Output ONLY the JSON object. No prose, no code fences.
