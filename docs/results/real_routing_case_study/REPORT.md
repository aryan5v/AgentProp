# AgentProp Real-Routing Case Study

- Model: `gemini-3.1-pro-preview`
- Workflow: `planner_coder_tester_reviewer`
- AgentProp seed stages (budget 2, greedy + IndependentCascade): `planner, tester`
- Non-seed stages receive a compressed conventions summary; the tester always receives full conventions (it is a verifier).
- Tasks: 10 self-contained coding problems with executable tests.

## Headline result

| Arm | Success rate | Mean tokens/task | Total tokens |
| --- | --- | --- | --- |
| broadcast (full context to all) | 100% (10/10) | 9243 | 92434 |
| agentprop (full context to seeds only) | 70% (7/10) | 7725 | 77254 |

- **Measured token saving (agentprop vs broadcast): +16.4%**
- **Success-rate change: -30%** (70% vs 100%)

## AgentProp cost-model prediction vs measured reality

Edge weights were re-fit from the captured broadcast-arm traces via `trace_loader.graph_from_trace_dict`, then AgentProp's own cost model (`broadcast_cost` vs `seeded_routing_cost`) was evaluated on the fitted graph.

- AgentProp **predicted** token saving on the fitted graph: +22.6%
- **Measured** token saving in the real run: +16.4%
- Prediction error: 6.2%

## Per-task detail

| Task | broadcast | tokens | agentprop | tokens | fail reason (agentprop) |
| --- | --- | --- | --- | --- | --- |
| to_base | PASS | 5725 | PASS | 9673 |  |
| rle_encode | PASS | 10069 | PASS | 4796 |  |
| roman_to_int | PASS | 10509 | FAIL | 8669 | `AssertionError` |
| is_valid_ipv4 | PASS | 12594 | PASS | 9147 |  |
| add_fractions | PASS | 12421 | FAIL | 9334 | `AssertionError: expected ValueError for zero den` |
| merge_intervals | PASS | 6706 | PASS | 4894 |  |
| longest_unique | PASS | 5139 | PASS | 6798 |  |
| is_balanced | PASS | 8325 | FAIL | 7657 | `AssertionError` |
| spiral_order | PASS | 10989 | PASS | 6979 |  |
| top_k_words | PASS | 9957 | PASS | 9307 |  |

## Interpretation

**Cost side — works.** AgentProp's routing cut mean token cost by 16.4% (9243 -> 7725 tokens/task) by sending full shared context only to seed stages. The trace-fit cost model predicted 22.6%; the 6.2% gap is itself a useful signal that the hardcoded non-seed compression factor should be calibrated from measured tokens.

**Quality side — exposes a real weakness.** 3 of 10 task(s) regressed (broadcast PASS -> agentprop FAIL): `roman_to_int, add_fractions, is_balanced`. Mechanism: AgentProp's topology-based `greedy_seed_selection` chose `planner, tester` as seeds and left the **coder** as a non-seed node, so it received only a compressed summary and dropped convention-dependent edge cases (e.g. empty-input and invalid-input handling). The coder is the most context-sensitive node in a coding workflow, yet graph centrality does not see that. **Conclusion: the thesis holds on cost but needs redirection on seed selection — routing must be role/quality-aware, not topology-only.**

See `docs/research/real_routing_case_study_findings.md` for the failure dissection and a concrete roadmap to make AgentProp better.

### Honest scope and limits (this is a *conservative* test)

AgentProp targets multi-agent workflows where shared context is broadcast to many agents. This experiment is a real multi-agent workflow (4 agent stages with cross-agent context routing), but it under-exercises AgentProp's strength on three axes, so the savings here are a floor, not a ceiling:

1. **Small shared payload.** The only context routed/compressed is a ~500-token static conventions doc; the inter-agent transcript still flows in full. Real systems broadcast growing transcripts, shared memory, and large retrieved documents — far more to save on.
2. **Small, sparse graph.** A 4-stage near-linear pipeline has little broadcast redundancy. AgentProp should save more on dense, star, `hub_and_spoke_supervisor`, and `rag_pipeline` topologies with many agents reading from large shared `DOCUMENT`/`MEMORY` nodes.
3. **Reasoning-dominated cost.** With a thinking model, per-agent completion tokens dwarf the input/context tokens that routing controls — so total-token savings are noisy per task. In context-heavy workflows the input side is the dominant cost and AgentProp's lever is larger.

Other caveats: self-contained coding problems (SWE-bench-*style*), not full SWE-bench repository tasks; single model, N=10, one trial per arm — treat magnitudes as directional, not definitive.

