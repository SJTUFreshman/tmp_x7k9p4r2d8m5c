# AFlow Optimizer — Propose a Workflow Modification

You are the "optimizer" for an automated multi-agent workflow search.
Your task is to read the current best-performing workflow and propose
a modified version that will score higher on MuSiQue dev.

# The Task You Are Optimizing
The workflow answers multi-hop questions from MuSiQue. Each workflow is
Python code implementing an async function `workflow(problem, ops)`
that returns the final answer string. It uses a small set of allowed
operators (documented below). You will produce the FULL SOURCE of a new
workflow.

# Allowed Operators (whitelist — do NOT invent new ones)

You may only call these three operators via `ops`:

    await ops.solve(problem, instruction="...")
        → returns {"reasoning": str, "answer": str}
        Executes an LLM call that reasons over the paragraphs.
        "instruction" is an optional extra hint (e.g. "double-check
        temporal ordering"). Empty string is fine.

    await ops.ensemble(candidates, problem)
        → returns {"chosen_index": int, "reason": str}
        Given a list of candidate solutions (each is the dict returned
        by ops.solve), an LLM picks the best one. Must be called with
        at least 2 candidates.

    await ops.answer_generate(text)
        → returns {"answer": str}
        Extracts a concise final answer string from any text.

# Workflow Contract

The workflow module MUST define an async function with this exact
signature:

    async def workflow(problem, ops):
        # ... your code ...
        return "<final answer string>"

- `problem` is a MuSiQueProblem-like object; you access its rendered
  paragraphs+question via `problem.rendered_text` (a string).
- `ops` is the operator bundle described above.
- Return value is the final answer as a plain string (no \boxed{}).

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
7. The modification should be a MEANINGFUL change vs the current
   workflow — e.g. add a review step, add self-consistency, restructure
   how ensemble is called. Do NOT just rename variables.

# Current Best Workflow
```python
{CURRENT_WORKFLOW_CODE}
```

# Current Dev Performance
- Current workflow EM on dev-{DEV_SIZE}: {CURRENT_EM}
- Common failure patterns from dev errors:
{FAILURE_SAMPLES}

# Your Task
Propose a modified workflow that should improve EM on MuSiQue dev.
Think about what step is missing or weak in the current workflow,
then rewrite the full workflow source.

# Output Format
Return ONLY the Python source code of the new workflow function.
No markdown fences, no prose before or after. The FIRST characters
of your response must be `async def workflow` or a legal import
line, and the LAST non-whitespace character must belong to the
workflow function's body.
