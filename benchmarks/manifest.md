# Benchmark Manifest

## Workflow Fixtures

## Synthetic Graph Families

These fixtures stress the graph-theory backbone separately from agent-specific
workflow semantics:

- `workflows/chain.json`: propagation path, cut points, and bridge detection.
- `workflows/star.json`: hub dominance and supervisor-style bottlenecks.
- `workflows/tree.json`: branching propagation and articulation structure.
- `workflows/dense_graph.json`: redundant paths and centrality ties.
- `workflows/small_world_graph.json`: local neighborhoods plus shortcuts.
- `workflows/random_directed_graph.json`: deterministic random directed stress case.
- `workflows/generic_dag.json`: layered directed acyclic propagation.
- `workflows/layered_pipeline.json`: planner, worker, verifier, output layers.

Research reproduction scripts (from repo checkout with editable install):

```bash
python experiments/failure_localization_study.py
python experiments/quality_cascade_vs_ic.py
```

Recommended graph-family smoke test:

```bash
agentprop benchmark chain --algorithms closeness k-core in-degree out-degree --models zero-forcing --trials 1
```

### `workflows/planner_coder_tester_reviewer.json`

The first end-to-end demo workflow. It models a common software-agent loop:

```text
Planner -> Coder -> Tester -> Reviewer -> Final
Planner -> Reviewer
Tester -> Coder
```

This fixture is designed to test:

- Context seeding
- Correction propagation from tester back to coder
- Verifier placement around tester/reviewer
- Cost savings against broadcast routing
- Low-weight edge pruning

Expected qualitative behavior:

- `planner` should often rank highly because it reaches coder and reviewer.
- `tester` is a strong verifier candidate because it detects implementation errors and routes corrections back to coder.
- `planner -> reviewer` is useful but lower-weight than the main pipeline.
- Broadcast routing wastes tokens by giving full context to every downstream node upfront.

### `workflows/research_writer_verifier.json`

Parallel research branches feed a writer and verifier:

```text
Planner -> Researcher A -> Writer -> Verifier -> Final
Planner -> Researcher B -> Writer
```

This fixture tests redundant research inputs, merge bottlenecks at `writer`, and verifier placement after synthesis.

### `workflows/debate_judge.json`

Three agents independently receive the prompt and send arguments to a judge:

```text
Prompt -> Agent A/B/C -> Judge -> Final
```

This fixture tests many-to-one aggregation, debate routing, and the difference between context seeding the prompt source versus downstream participants.

### `workflows/rag_pipeline.json`

A retrieval-augmented workflow:

```text
Query -> Retriever -> Summarizer -> Reasoner -> Verifier -> Final
Retriever -> Reasoner
```

This fixture tests tool nodes, retrieval shortcuts, verifier placement, and whether direct retrieval-to-reasoning edges reduce propagation delay.

### `workflows/tool_use_pipeline.json`

Planner dispatches to search and code tools, then analyst/tester nodes produce final output:

```text
Planner -> SearchTool -> Analyst -> Final
Planner -> CodeTool -> Tester -> Final
Tester -> CodeTool
```

This fixture tests tool-call dependencies, correction loops, and cost-aware pruning around tool outputs.

### `workflows/hub_and_spoke_supervisor.json`

A supervisor coordinates specialist agents and sends aggregated work to a verifier:

```text
Supervisor <-> Specialists
Supervisor -> Verifier -> Final
```

This fixture tests supervisor bottlenecks, hub-and-spoke broadcast waste, and resilience when specialist communication is expensive.

## Recommended First Benchmark

```bash
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50
```

For a fixture path instead of a built-in template:

```bash
agentprop optimize benchmarks/workflows/rag_pipeline.json --budget 2
```
