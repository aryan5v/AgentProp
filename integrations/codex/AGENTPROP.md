# AgentProp Instructions For Codex

Use AgentProp before implementing or changing multi-agent workflow routing.

## Before Work

```bash
agentprop agent-instructions <workflow> --target codex --out reports/codex_agent_brief.md
```

Read the generated brief. Treat selected seed agents as the first recipients of
full context. Treat verifier candidates as required review/check points.

## During Work

- Keep task context concentrated in the selected seed agents.
- Use verifier/checker nodes before finalizing.
- If summarizing or pruning a handoff, state the risk.
- For policy work, run `experiments/run_experiment_suite.py` and compare learned
  methods against training-free baselines.
- For external benchmark work, run `agentprop terminal-bench prepare` first so
  the manifest, watchdog wrapper, and post-run reporting path are reviewed
  before launch.

## Before Final Answer

- State which seed agents received full context.
- State which verifier/checker reviewed the result.
- Include verification commands or artifact paths.
- Mention whether AgentProp predicted cost savings or pruning risk.
