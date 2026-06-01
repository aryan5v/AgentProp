# Remaining PRD Archive

This archive preserves the post-`0.1.0-alpha.1` work needed to turn AgentProp
from a working alpha framework into a stronger research and developer product.

The priority is:

1. make the graph-theory backbone mathematically useful
2. make ML/DL/RL policies evaluate against that backbone
3. prove value on real LLM workflows with saved traces and cost-quality results

## Reference Repos To Learn From

- `hao-ai-lab/FastVideo`: use its recipe-first structure as inspiration for
  clear training/evaluation commands, hardware expectations, and result
  artifacts.
- `huggingface/ml-intern`: use its research-agent structure as inspiration for
  env-var based credentials, sandbox/cloud execution, trace retention, and
  experiment loops.

Do not copy code from either repo. Borrow product discipline: reproducible
recipes, explicit compute assumptions, and result artifacts that make claims
auditable.

## Workstream 1: Graph-Theory Backbone

Goal: make training-free graph analysis strong enough that ML/RL must beat a
credible baseline.

Progress:

- Added closeness, in-degree, out-degree, and k-core seed-selection baselines.
- Added articulation-point, bridge, edge-bottleneck, low-reliability cut-point,
  and failure-sensitive-node diagnostics.

### Classical Algorithms

- Closeness centrality seed selection.
- K-core / K-shell seed selection.
- Explicit in-degree and out-degree seed-selection modes.
- Articulation-point bottleneck detection.
- Bridge / weak-bridge bottleneck detection for directed workflow graphs.
- High-cost and high-traffic edge bottleneck ranking.
- Low-reliability cut-point detection.
- Failure-sensitive node scoring.

Acceptance evidence:

- Unit tests on hand-built graph families: path, star, tree, dense, DAG.
- Benchmark rows comparing new algorithms against PageRank, betweenness, CELF,
  and cost-aware greedy.
- Docs explaining directed vs. undirected behavior for each algorithm.

### Pruning Algorithms

- Low-usage pruning from trace frequency.
- Betweenness-preserving pruning.
- Reachability-preserving pruning.
- Cost-aware edge pruning.
- Redundancy-based pruning from alternate paths.

Acceptance evidence:

- Pruning evaluator reports coverage, cost, reachability, and robustness deltas.
- `agentprop prune` supports `--target-token-reduction`.
- Saved pruning benchmark table across synthetic and agent-inspired workflows.

### Verifier Semantics

Define what a verifier can:

- observe: upstream nodes, downstream nodes, selected paths, or full trace
- correct: messages, node outputs, edge routing, or final answer only
- intercept: errors before downstream propagation or after propagation

Then implement:

- Betweenness verifier placement: added.
- PageRank verifier placement: added.
- Error-propagation centrality: added.
- Greedy correction-coverage placement: added.

Acceptance evidence:

- Verifier report includes coverage, expected correction delay, and interception
  estimates.
- Tests prove that verifier placement changes when graph risk/cost changes.

## Workstream 2: Metrics and Evaluation

Goal: evaluate routing with graph metrics and task-quality metrics.

### Graph Metrics

- Robustness under node failure: added.
- Robustness under edge failure: added.
- Correction propagation delay.
- Error persistence.
- Verifier interception rate.
- Failure recovery rate.
- Cost-adjusted success: added.
- Efficiency score: added.

Acceptance evidence:

- Metrics module exposes typed dataclasses.
- Markdown/JSON/HTML reports include metric tables.
- Benchmark runner can save metric artifacts reproducibly.

### Real Task Quality

Implement scorer interfaces for:

- human labels: added
- LLM-as-judge: added through injected judge adapter
- unit-test / exact-match scoring: added
- task-specific rubrics: added

Acceptance evidence:

- One scorer can evaluate saved case-study outputs.
- Scores are stored with model, prompt, timestamp, and rubric metadata.

## Workstream 3: Workflow Templates

Goal: separate graph-family benchmarks from agent-inspired benchmarks.

Add synthetic templates:

- chain: added
- star: added
- tree: added
- dense graph: added
- small-world graph: added
- random directed graph: added
- generic DAG: added
- layered pipeline: added

Acceptance evidence:

- Every template exports valid JSON.
- Every template appears in benchmark fixtures.
- Benchmark result tables label graph family and agent workflow separately.

## Workstream 4: CLI and Reports

Goal: make the framework useful without writing Python.

Add CLI commands:

- `agentprop simulate workflow.json --model independent-cascade`: added
- `agentprop prune workflow.json --target-token-reduction 0.3`: added

Extend reports:

- HTML report output: added.
- Visual graph embedded or linked from reports.
- Pruning section with target reduction and risk estimate: added.
- Robustness section: added.

Acceptance evidence:

- CLI tests cover `simulate` and `prune`: added.
- Report renderer tests cover HTML output and escaping: added.
- Report renderer tests cover pruning risk and robustness sections: added.
- Tutorial uses the CLI-only path end to end.

## Workstream 5: ML / DL

Goal: train learned policies that can be compared honestly against classical
graph algorithms.

Progress:

- Added trace-calibrated learned propagation model.
- Added learned propagation training script for trace JSON.
- Added `configs/experiment_suites/ml_core.json` and
  `experiments/run_experiment_suite.py` as an `ml-intern`-inspired recipe layer
  for reproducible ML/RL artifact runs.

### Immediate DL Additions

- MLP node-scoring baseline: added.
- GIN seed scorer: added to optional torch backend.
- Edge-pruning scorer: added as a dependency-light linear scorer.
- Verifier-placement scorer: added through verifier-placement labels and node scorers.
- Pairwise ranking loss: added dependency-light `PairwiseNodeRanker` trained
  from seed preference pairs.
- Regression target for propagation time or marginal gain: added
  `LinearNodeRegressor` trained on single-seed marginal utility targets.
- Generalization tests on unseen workflow graphs: added as a lightweight
  leave-one-workflow-out experiment.

### Later DL Additions

- Graph Transformer: added to optional torch backend.
- Heterogeneous GNN: added to optional torch backend with node-type embeddings.
- Edge-conditioned GNN: added to optional torch backend with edge-attribute gates.
- Edge-conditioned training and inference now share the full edge feature matrix
  used by pruning models: message cost, latency, relevance, reliability,
  activation probability, dependency strength, edge weight, and degree features.
- Trace-conditioned learned propagation model.

Acceptance evidence:

- Training scripts write model configs, metrics, and predictions.
- Evaluation compares learned policies against random, PageRank, CELF,
  cost-aware greedy, and Q-learning.
- Experiment suite manifest records runtime assumptions, artifact root, command
  arguments, outputs, and exit codes.
- At least one learned model beats a centrality baseline on held-out synthetic
  graphs, or docs state that it does not.

### Compute Notes

No CUDA/GPU is needed for the current tiny alpha models. Modal GPU compute
becomes useful when we add:

- larger synthetic graph sweeps
- hyperparameter searches
- larger optional torch GNN sweeps
- learned propagation from many traces

Use env vars, never committed secrets:

```bash
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

## Workstream 6: Reinforcement Learning

Goal: move beyond seed-selection Q-learning toward real workflow control.

### Environment

- Make `AgentRoutingEnv` Gymnasium-compatible: added dependency-light
  `reset_gymnasium()` and `step_gymnasium()` adapter methods.
- Add trajectory export/import: trajectory export is present in
  `experiments/run_rl_routing.py`; import/replay added through
  `agentprop.rl.replay_actions` and `experiments/replay_rl_trajectory.py`.
- Add deterministic evaluation mode: replay script accepts a propagation seed
  and trials count for reproducible trajectory checks.

### Algorithms

- REINFORCE baseline: added dependency-light tabular policy-gradient trainer
  with seed-selection and expanded-action support.
- PPO baseline: added dependency-light clipped policy-gradient trainer with
  tabular actor preferences and value baseline.
- Evaluation against greedy, CELF, GNN, and broadcast: added
  `experiments/evaluate_routing_baselines.py` for synthetic workflow
  comparisons across broadcast, classical, ML/GNN-style, Q-learning, and
  REINFORCE/PPO policies.
- Expanded-action RL evaluation: routing baseline comparison now emits
  `q_learning_expanded`, `reinforce_expanded`, and `ppo_expanded` rows with
  action traces, reward traces, activated verifiers, pruned edges, tool calls,
  and summary requests preserved for audit.
- Expanded-action reward shaping: verifier activation, safe/risky pruning, tool
  calls, and summary requests now add interpretable reward terms on top of
  coverage, cost, and propagation-time reward.

### Expanded Actions

- `SEND_CONTEXT(node)`: added.
- `ACTIVATE_VERIFIER(node)`: added.
- `SEND_MESSAGE(edge)`: added.
- `PRUNE_EDGE(edge)`: added.
- `CALL_TOOL(node)`: added.
- `REQUEST_SUMMARY(node)`: added.
- `STOP`: added.

Acceptance evidence:

- RL policy improves over random on held-out synthetic workflows.
- RL policy is compared with GNN and classical baselines using the same metrics.
- Saved trajectories show chosen actions, rewards, costs, and final proxy quality:
  added through cumulative rewards and trajectory summaries.

## Workstream 7: Real LLM Case Study

Goal: prove AgentProp helps real workflows, not only simulations.

Run the protocol in `docs/research/case_study_protocol.md`.

Progress:

- Added `benchmarks/case_study_tasks.json` with the 20-task study set described
  in the protocol.
- Added `experiments/run_case_study.py`, an offline accounting harness that
  compares broadcast, optimized-greedy, ML message-passing, and PPO routing
  arms and writes JSON, CSV, summary, quality, cost, and trace artifacts.
- Added OpenAI-compatible LLM execution support for Token Router or OpenAI-style
  endpoints. LLM mode records prompt/completion/total tokens, latency, final
  outputs, rubric scores, and trace metadata.
- Added case-study preflight mode that writes `preflight.json` with task count,
  policy arms, selected seeds, expected artifacts, and Token Router/OpenAI
  readiness without making LLM calls.
- Added `experiments/analyze_case_study.py` to turn saved results into
  `analysis.json`, `analysis.md`, `policy_comparison.csv`, token-savings plots,
  quality plots, and acceptance checks.

Still needed:

- Run the 20 tasks with real LLM calls.
- Provide `TOKEN_ROUTER_API_KEY`, `TOKEN_ROUTER_BASE_URL`, and
  `TOKEN_ROUTER_MODEL` or equivalent OpenAI env vars.
- Run the connected LLM mode and save committed or private-reviewed results.
- Save external verification command logs from an execution environment that
  actually applies generated code changes.
- Save real verifier corrections and interception events.
- Review the generated final analysis and plots after the real run.

### API Notes

Token Router or another LLM API will be needed for this workstream.

Use env vars, never committed secrets:

```bash
TOKEN_ROUTER_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

The case-study runner should support:

- fixed model list
- retry policy
- max cost ceiling
- saved prompts and responses
- redaction pass before committing artifacts

## Workstream 8: Integrations

Goal: analyze workflows users already have.

Coding-agent integration:

- Added `agentprop agent-instructions` to generate Claude Code, Codex, or
  generic coding-agent briefs from AgentProp routing recommendations.
- Added `docs/coding_agents.md` with Claude Code skill shape, Codex plugin
  shape, MCP tool recommendations, and ML/DL/RL suite usage guidance.
- Added a Claude Code skill template under `integrations/claude-code/`.
- Added a Codex instruction template under `integrations/codex/`.
- Added `agentprop-mcp`, a lightweight stdio JSON-RPC server exposing
  AgentProp analyze, optimize, report, and agent-instruction tools.

Adapters to build:

- LangGraph: added dependency-light dictionary export/import adapter.
- AutoGen: added dependency-light dictionary export/import adapter.
- CrewAI: added dependency-light dictionary export/import adapter.
- OpenAI Agents SDK: added dependency-light dictionary export/import adapter.
- LlamaIndex workflows: added dependency-light dictionary export/import adapter.

Still needed:

- Optional runtime adapters that instantiate framework-native classes when the
  corresponding packages are installed.
- Round-trip tests against real framework examples.

Acceptance evidence:

- Each adapter has a tiny fixture workflow.
- Adapter output round-trips into `AgentGraph`.
- CLI can analyze an adapter-exported graph.

## Workstream 9: Public Release Follow-Through

Current decision: keep private until a real LLM case-study result is attached,
or publish explicitly as alpha.

Before public release:

- Finish `docs/results/case_study_001/`.
- Add screenshots/plots to README.
- Confirm dependency extras install cleanly.
- Add issue templates.
- Create GitHub release from `RELEASE_NOTES.md`.

## Recommended Build Order

1. Graph-family templates and missing classical algorithms.
2. Pruning/verifier semantics and metrics.
3. `simulate` and `prune` CLI commands.
4. MLP/GIN/edge-pruning/verifier DL policies.
5. Gymnasium-compatible RL with expanded action space.
6. Real LLM case-study runner using Token Router.
7. Public release.

## What We Need From You Later

Not needed immediately:

- CUDA/GPU compute
- Modal credentials
- Token Router key

Useful soon:

- Token Router key once we start the real LLM case-study runner.
- Modal GPU access once we scale DL/RL experiments beyond small graph smoke
  tests.

When credentials are needed, provide them as local environment variables or a
`.env` file that stays ignored by Git.
