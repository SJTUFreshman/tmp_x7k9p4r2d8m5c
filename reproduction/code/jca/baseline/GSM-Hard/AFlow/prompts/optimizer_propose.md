# AFlow Optimizer — Propose a GSM-Hard Workflow

Improve the current Python workflow for numerical GSM-Hard word problems.
Failures commonly come from incorrect equations, corrupted large/small values,
unit conversion, sign errors, scale errors, and unchecked arithmetic.

Allowed asynchronous operators are only:

- `await ops.solve(problem, instruction="...")`
- `await ops.ensemble(candidates, problem)`
- `await ops.answer_generate(text)`

The full workflow must define `async def workflow(problem, ops)`, return a
plain numeric answer string, make at least two and at most twelve operator
calls, and use only allowed standard-library imports (`asyncio`, `math`,
`random`, `json`, `re`, `typing`). Do not access files, processes, or network.

Current workflow:

```python
{CURRENT_WORKFLOW_CODE}
```

Current EM on {DEV_SIZE} search examples: {CURRENT_EM}

Failure samples:
{FAILURE_SAMPLES}

Return only the complete Python source. Make a meaningful structural change
that should improve numerical reliability; do not merely rename variables.
