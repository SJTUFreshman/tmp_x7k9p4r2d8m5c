# AFlow Optimizer — Propose a MATH Workflow

Improve the current Python workflow for MATH problems. Failures may come from weak independent solving, unchecked assumptions, missing cases, sign or arithmetic errors, poor candidate comparison, or incorrect final answer extraction.

Allowed asynchronous operators are only:

- `await ops.solve(problem, instruction="...")`
- `await ops.ensemble(candidates, problem)`
- `await ops.answer_generate(text)`

The full source must define `async def workflow(problem, ops)`, return a plain short mathematical answer string, contain at least 2 and at most 12 explicit operator calls, and use only allowed standard-library imports (`asyncio`, `math`, `random`, `json`, `re`, `typing`). Do not access files, processes, network, environment, reflection, or unlisted APIs.

Current workflow:

```python
{CURRENT_WORKFLOW_CODE}
```

Current EM on the fixed {DEV_SIZE} MATH train examples: {CURRENT_EM}

Failure samples:
{FAILURE_SAMPLES}

Return only the complete Python source. Make a meaningful structural change intended to improve mathematical reliability; do not merely rename variables.
