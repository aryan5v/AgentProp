# Contributing to AgentProp

AgentProp is early, but the contribution standard is intentionally high: every feature should make the framework more useful to developers and more defensible for research.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test   # or: pytest
```

With an editable install, `experiments/` and `examples/` run without
`PYTHONPATH=src`. See [docs/environment.md](docs/environment.md).

## Data, Secrets, And Benchmark Artifacts

AgentProp is a **public** repository. Do not commit anything that should stay
private or local.

**Never commit:**

- API keys, tokens, `.env` files, or Harbor/Modal/provider credentials
- Local experiment output (`results/`, `reports/`, `benchmark-results/`)
- In-progress benchmark files (`PARTIAL_REPORT.md`, `results_partial.json`,
  `checkpoint.json`)
- Raw LLM prompts or traces from paid runs
- Terminal-Bench launch bundles (manifest, runbook, watchdog scripts) — generate
  these locally with `agentprop terminal-bench prepare`

**Safe to commit under `docs/results/` only when sanitized:**

- Final `REPORT.md` and `results.json` with limitations stated
- `outputs.jsonl` rows with task metadata, pass/fail, and token counts — not raw
  prompts or secrets

See [docs/results/ARTIFACTS.md](docs/results/ARTIFACTS.md) for the manifest of
intentional public artifacts. CI runs a secret scan on every pull request.

**Keep internal notes local:**

- Paper outlines, research backlogs, release/publishing runbooks, extended
  postmortems, and "what we should do next" roadmaps belong under `docs/local/`
  (gitignored). Copy [docs/local/README.example.md](docs/local/README.example.md)
  as a starting point.
- If a doc has blockers, remaining work, or reads like notes to the team, it does
  not belong in the public `docs/` tree.

## Quality Gates

Before opening a PR or pushing a release candidate:

```bash
ruff check .
mypy src
pytest
```

## Contribution Workflow

- Contributors should open pull requests against `main`.
- Maintainers can push directly only for release hygiene, CI fixes, or small docs
  updates; feature work should still go through review.
- Public claims must link to commands, saved artifacts, or clearly marked
  limitations.
- New routing policies should report both cost and quality/risk evidence.

## Design Principles

- Keep the core package lightweight.
- Make ML/DL/RL dependencies optional.
- Prefer clear graph contracts over ad hoc workflow assumptions.
- Every algorithm should be benchmarkable.
- Every benchmark should be reproducible.
- Avoid overclaiming zero forcing; treat it as one propagation model among several.
- Prefer quality-aware defaults when a workflow has context-sensitive roles.

## Adding Algorithms

When adding a new algorithm:

- Put training-free methods in `src/agentprop/algorithms/`.
- Put propagation models in `src/agentprop/propagation/`.
- Add the method to the benchmark runner if it selects seeds or routes context.
- Add tests on at least one built-in workflow.
- Document what objective the method optimizes.
- Add routing-risk output if the method can reduce context to critical nodes.

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
