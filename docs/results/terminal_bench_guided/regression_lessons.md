# First Terminal-Bench Regression Lessons

This note records the concrete mistakes found in the first AgentProp-guided
Terminal-Bench run. It is based on saved local Harbor artifacts and does not add
new benchmark results.

## `chess-best-move`

- Baseline: passed.
- AgentProp-guided: failed.
- Pattern: behavioral regression.

The prompt asked for all winning moves if multiple winning moves existed. The
baseline wrote both `e2e4` and `g2g4`. The guided run used a heavier
planner/engine-verifier flow but wrote only `e2e4`, so it missed the full answer
set.

Policy fix:

- For direct-answer, puzzle, chess, and perception-heavy tasks, avoid heavyweight
  process loops.
- If the prompt asks for all valid answers, enumerate and verify the full answer
  set before writing the final file.
- Treat a single engine "best move" as insufficient when the task asks for all
  winning or equivalent moves.

## `tune-mjcf`

- Baseline: verifier reward 1.0, but with an agent timeout artifact.
- AgentProp-guided: verifier reward 0.0 with `AgentTimeoutError`.
- Pattern: timeout-sensitive policy regression.

The guided run launched a broad sequence of solver, integrator, stability, enum,
and XML-option experiments. That is useful for research, but it is too expensive
inside a benchmark task where the evaluator is available and the goal is a
passing artifact under time pressure.

Policy fix:

- Treat `/app/eval.py` or equivalent task evaluator as the source of truth.
- Use a small fixed candidate budget for simulator tuning.
- Stop as soon as a candidate passes correctness and satisfies the speed target.
- Do not keep exploring once additional experiments are unlikely to change the
  verifier outcome.

## Applied To Future Runs

These lessons are now included in the Terminal-Bench 2.1 preflight guidance at
`docs/results/terminal_bench_21_preflight/agentprop-extra-instructions.md`.
