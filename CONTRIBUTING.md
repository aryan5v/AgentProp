# Contributing to AgentProp

AgentProp is early, but the contribution standard is intentionally high: every feature should make the framework more useful to developers and more defensible for research.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Quality Gates

Before opening a PR or pushing a release candidate:

```bash
ruff check .
mypy src
pytest
```

## Design Principles

- Keep the core package lightweight.
- Make ML/DL/RL dependencies optional.
- Prefer clear graph contracts over ad hoc workflow assumptions.
- Every algorithm should be benchmarkable.
- Every benchmark should be reproducible.
- Avoid overclaiming zero forcing; treat it as one propagation model among several.

## Adding Algorithms

When adding a new algorithm:

- Put training-free methods in `src/agentprop/algorithms/`.
- Put propagation models in `src/agentprop/propagation/`.
- Add the method to the benchmark runner if it selects seeds or routes context.
- Add tests on at least one built-in workflow.
- Document what objective the method optimizes.

## Adding Workflow Fixtures

Workflow fixtures live in `benchmarks/workflows/`.

Each fixture should:

- Pass `AgentGraph.from_json`.
- Use supported `NodeType` values.
- Include non-negative token/message costs.
- Include reliability and error-rate assumptions when meaningful.
- Be summarized in `benchmarks/manifest.md`.

## Research Contributions

Research-facing contributions should connect to at least one of:

- influence maximization
- graph propagation
- randomized zero forcing
- verifier placement
- topology pruning
- cost-quality evaluation
- GNN/RL routing policies
