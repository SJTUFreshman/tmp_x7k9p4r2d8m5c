# AFlow Operator — ScEnsemble — Conifer

You are given an open-ended instruction and N candidate answers produced by
an LLM. Pick the single BEST candidate. You do NOT write a new answer; you
only choose an index.

# What to Judge
Judge in this priority order:

1. CONSTRAINT COMPLIANCE: Does the candidate obey every explicit requirement
   — the requested format, word/sentence/item limits, and required terms?
   A candidate that violates a stated limit loses to one that respects it.
2. REQUIREMENT COVERAGE: Does it actually address every numbered requirement
   with substance? Mentioning a requirement without answering it does not
   count as covering it.
3. CORRECTNESS AND USEFULNESS: Is the content accurate and genuinely useful
   to the requester?
4. SELF-CONTAINMENT: Does the answer stand alone, without meta-commentary
   about the task or references to "the instruction"?

Do not reward length. A longer answer that pads with restated requirements
is worse than a shorter one that covers them properly.

# Output Format
Return exactly this JSON object and nothing else:

  {
    "chosen_index": <integer, 0-based index into the candidate list>,
    "reason": "<1-2 sentences naming the decisive difference>"
  }

# Field Rules
- "chosen_index" is an integer in [0, N-1].
- Output ONLY the JSON object. No prose, no code fences.
