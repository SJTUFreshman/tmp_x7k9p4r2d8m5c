# AFlow Workflow Optimizer — MultiPL-E-8Lang

Rewrite the current Python workflow to improve public train70 test pass rate.
Output only Python source defining `async def workflow(problem, ops)`.

Available async operators:

- `ops.solve(problem, instruction="...")`
- `ops.ensemble(candidates, problem)`
- `ops.answer_generate(text)`

You may compose these with safe Python and `asyncio`, but cannot access files,
network, subprocesses, test execution, labels, or any API other than `ops`.
The workflow must return one source-code continuation string. Keep operator
usage economical and generalize across Python, C++, Java, PHP, TypeScript, C#,
Shell, and JavaScript. Never specialize to problem IDs or memorize failures.

Current workflow evaluated on {DEV_SIZE} fixed train70 tasks with public-test
pass rate {CURRENT_EM}:

```python
{CURRENT_WORKFLOW_CODE}
```

Failure samples:

{FAILURE_SAMPLES}

Propose one improved workflow. Output Python source only.
