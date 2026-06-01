# AgentProp Benchmark Guidance

Use AgentProp routing discipline, but keep it budget-aware.

- Classify the task before acting: setup/build, code repair, numerical/scientific,
  reverse engineering, repo hygiene, or direct-answer.
- Spend full context on implementation-sensitive and verifier-sensitive phases.
- Prefer executable checks over long speculative reasoning.
- Stop exploration when the next action is not expected to change the verifier
  outcome; write a concise final answer instead.
- For numerical or scientific tasks, confirm units, coordinate systems, schema,
  and expected output ranges before fitting or optimizing.
- If a task is direct-answer or perception-heavy, avoid heavyweight process loops.
- Preserve evidence: commands run, files changed, verification output, and any
  unresolved risk.
