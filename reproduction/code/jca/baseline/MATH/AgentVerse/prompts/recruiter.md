# AgentVerse Recruiter — MATH

Design exactly three complementary roles for solving the given mathematics problem. The low-capacity role should perform a useful bounded check, the mid-capacity role should develop a solid solution, and the high-capacity role should handle the hardest reasoning or verification. Roles must be specific to the problem without assuming access to a reference answer.

Return exactly one JSON object and nothing else:

{"roles":[{"name":"...","capacity":"low","description":"..."},{"name":"...","capacity":"mid","description":"..."},{"name":"...","capacity":"high","description":"..."}]}

Use each capacity exactly once. Every name and description must be non-empty.
