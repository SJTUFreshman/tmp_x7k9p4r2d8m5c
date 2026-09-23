# AFlow Optimizer — Propose a Workflow Modification

You are the "optimizer" for an automated multi-agent workflow search.
Your task is to read the current best-performing workflow and propose
a modified version that will score higher on the Conifer dev sample.

# The Task You Are Optimizing

The workflow answers open-ended instruction-following requests. Each request
carries explicit requirements: content points to cover, an output format
(bullets, ordered list, table, code, prose), word/sentence/item limits, and
required terminology. There is no single correct answer.

Each workflow is Python code implementing an async function
`workflow(problem, ops)` that returns the final answer string. It uses a
small set of allowed operators (documented below). You will produce the FULL
SOURCE of a new workflow.

# How the Score Works

The score is `hard_score`, computed mechanically from the returned string:

- 75% is the pass rate of explicit boolean checks: the answer is non-empty,
  the requested format is present, word/sentence/item limits are respected,
  and every required term appears.
- 25% is lexical coverage of the numbered requirements.

Two consequences worth designing for:

- A workflow that produces a well-researched answer in the WRONG format
  scores worse than one that respects the format. Format and limit
  compliance is where most points are won or lost.
- Many requests carry no explicit constraints at all. On those, any non-empty
  answer already scores 1.0, so there is nothing to gain — do not spend
  operator calls on them if you can avoid it.

Do NOT propose workflows whose strategy is to pad the answer with restated
requirement text. That inflates lexical coverage without answering the
request, and such proposals are rejected on review.

# Allowed Operators (whitelist — do NOT invent new ones)

You may only call these three operators via `ops`:

    await ops.solve(problem, instruction="...")
        → returns {"reasoning": str, "answer": str}
        Executes an LLM call that writes a complete answer.
        "instruction" is an optional extra hint (e.g. "re-check the word
        limit and the requested format"). Empty string is fine.

    await ops.ensemble(candidates, problem)
        → returns {"chosen_index": int, "reason": str}
        Given a list of candidate solutions (each is the dict returned by
        ops.solve), an LLM picks the best one. Must be called with at least
        2 candidates.

    await ops.answer_generate(text)
        → returns {"answer": str}
        Cleans meta-commentary off a draft and returns the deliverable,
        preserving its formatting.

# Workflow Contract

The workflow module MUST define an async function with this exact signature:

    async def workflow(problem, ops):
        # ... your code ...
        return "<final answer string>"

- `problem` exposes the rendered request via `problem.rendered_text`
  (a string). There is no gold answer available to the workflow.
- `ops` is the operator bundle described above.
- Return value is the final answer as a plain string.

# Rules for Your Proposal

1. Return ONLY valid Python 3 source code. No prose around it.
2. The workflow function must be `async def workflow(problem, ops)`.
3. You may only import from the Python stdlib (asyncio, math, random,
   json, re, typing). No network, no filesystem, no subprocess.
4. You may only call `ops.solve`, `ops.ensemble`, `ops.answer_generate`.
   Do NOT invent new operator methods.
5. The workflow MUST make at least 2 operator calls total. Trivial
   single-call workflows are rejected.
6. Keep total operator calls per problem <= 12 to avoid explosion.
7. The modification should be a MEANINGFUL change vs the current workflow —
   e.g. add a constraint-checking revision pass, vary the solve instructions
   to target format compliance, restructure how ensemble is called. Do NOT
   just rename variables.

# Current Best Workflow
```python
{CURRENT_WORKFLOW_CODE}
```

# Current Dev Performance
- Current workflow hard_score on dev-{DEV_SIZE}: {CURRENT_EM}
- Problems that failed at least one explicit check:
{FAILURE_SAMPLES}

# Your Task
Propose a modified workflow that should improve hard_score on the Conifer
dev sample. Look at which explicit checks are failing above, decide what
step is missing or weak in the current workflow, then rewrite the full
workflow source.

# Output Format
Return ONLY the Python source code of the new workflow function.
No markdown fences, no prose before or after. The FIRST characters
of your response must be `async def workflow` or a legal import
line, and the LAST non-whitespace character must belong to the
workflow function's body.
