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

## Regression Fixes From The First AgentProp Run

The first guided run regressed on tasks where generic planner/implementer/verifier
discipline was too heavy or too narrow. Apply these task-specific corrections:

### Direct Answer / Perception Tasks

- If the prompt asks for all valid answers, enumerate the full answer set before
  writing the final file. Do not stop after the first engine/model "best" answer.
- For chess or puzzle tasks, explicitly check whether multiple winning moves,
  mates, or equivalent optima satisfy the prompt.
- Keep the loop short: inspect input, compute/verify candidates, write exactly
  the requested format, and stop.

### Simulator / Numerical Tuning Tasks

- Treat the provided evaluator as the source of truth; optimize against it early.
- Limit candidate sweeps to a small fixed budget before choosing the best valid
  candidate.
- Stop when a candidate passes correctness and satisfies the target speed/quality
  threshold; do not continue broad exploratory searches.
- Do not change physical semantics, units, schemas, or coordinate conventions
  unless the evaluator proves equivalence.
