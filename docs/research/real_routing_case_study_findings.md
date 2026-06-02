# Real-Routing Case Study — Findings

This is the analysis companion to the machine-generated
[`docs/results/real_routing_case_study/REPORT.md`](../results/real_routing_case_study/REPORT.md).
It records the first **real-LLM** validation of AgentProp's context-routing thesis
and what it implies for the library.

## What was run

- **Harness:** [`experiments/run_real_routing_case_study.py`](../../experiments/run_real_routing_case_study.py)
- **Model:** `gemini-3.1-pro-preview` (OpenAI-compatible endpoint)
- **Workflow:** `planner_coder_tester_reviewer` (AgentProp built-in template) — a real
  **multi-agent** loop, four distinct agent stages with context flowing along the
  workflow graph.
- **Tasks:** 10 self-contained coding problems with **executable tests**
  ([`benchmarks/real_routing_tasks.json`](../../benchmarks/real_routing_tasks.json)).
  Each task's edge cases depend on a shared *team conventions* document
  (empty-input, invalid-input, casing, reduction rules).
- **Two arms, where routing genuinely changes context sharing:**
  - `broadcast` — every agent receives the **full** conventions document.
  - `agentprop` — only the seed agents chosen by `greedy_seed_selection` +
    `IndependentCascade` receive the full document; non-seed agents receive a
    one-time **compressed summary**. (The tester always gets full context — it is
    a verifier.)
- **Success** = real subprocess execution of the task's test suite (true pass/fail).
- **Cost** = real summed `usage` tokens from the provider.
- Edge weights were re-fit from the captured traces via `trace_loader`, and
  AgentProp's own cost model was compared against the measured saving.

## Headline numbers

| Arm | Success | Mean tokens/task | Total tokens |
| --- | --- | --- | --- |
| broadcast | 100% (10/10) | 9243 | 92434 |
| agentprop | 70% (7/10) | 7725 | 77254 |

- Measured token saving: **+16.4%**
- Success change: **−30%** (three regressions)
- AgentProp predicted saving (cost model on the re-fit graph): **+22.6%**
  → measured **+16.4%** (prediction error 6.2 points).

Reproducibility note: an earlier identical run produced +8.1% saving with **one**
regression (`roman_to_int`). The token magnitudes move run-to-run (see the
thinking-variance caveat below), but the **direction is stable**: AgentProp cuts
cost and the non-seed coder regresses on convention-dependent edge cases.

## The three regressions, dissected

All three failures are the **same mechanism**: AgentProp seeded `planner, tester`
and left the **`coder`** as a non-seed node, so it built the implementation from
only the *compressed* conventions summary and silently dropped the exact rules the
summary elided.

**`roman_to_int`** — empty-input and invalid-input rules dropped:

```python
# agentprop (coder had summary only) — FAIL: roman_to_int('') == 0 -> got None
if not s:
    return None              # should return 0 (empty-input rule)
...
if s[i] not in roman_values:
    return None              # should raise ValueError (invalid-input rule)
```

**`add_fractions`** — invalid-input rule dropped:

```python
# agentprop — FAIL: expected ValueError for zero denominator
if den_a == 0 or den_b == 0:
    return ""                # should raise ValueError
```

**`is_balanced`** — empty-input rule dropped:

```python
# agentprop — FAIL: is_balanced('') is True -> got False
if not s:
    return False             # should return True (empty-input rule)
```

In every case the broadcast arm (coder with full conventions) handled the edge
case correctly. The tester (a seed, full context) did not rescue any of them. This
is not a model failure — it is a **routing** failure: the most context-sensitive
agent in a coding workflow was the one AgentProp chose to starve.

## What this validates and what it redirects

- **Validated — the cost machinery is real.** Routing context to a seed set
  genuinely reduced tokens (16.4%), and the analysis loop (trace ingest → weight
  refit → cost model) runs end-to-end on real data. The cost model's prediction
  (22.6%) was in the right ballpark (6.2-point error).
- **Redirected — seed selection is blind to task-semantic importance.** Graph
  centrality picked planner/tester; the *coder* is what actually determines output
  quality. Centrality is not a proxy for criticality.

## Important caveat: routing saves *input* tokens; reasoning tokens are noise

AgentProp's lever is the **input/context** side — it sends fewer context tokens to
non-seed agents. But `gemini-3.1-pro-preview` is a reasoning model, so most tokens
are **completion/thinking** tokens, which are stochastic (a stage may "think" for
1k or 3k tokens on the same prompt). The agentprop arm also pays a fixed overhead:
one extra summarizer call. As a result, *total*-token savings are noisy per task
and can even invert (e.g. `to_base` cost more under agentprop this run) even though
the run aggregate is a clear saving. The honest reading: **AgentProp is an
input/context-cost optimizer**; scoring it on total tokens against a thinking model
understates and obscures the effect. A future harness revision should record
prompt vs completion tokens separately so the input-side saving is legible.

## Scope: this is a *conservative* test of AgentProp

It is a genuine multi-agent workflow with real cross-agent context routing, but it
**under-exercises AgentProp's strength** on three axes, so these savings are a
floor, not a ceiling:

1. **Small shared payload.** The only context routed/compressed is a ~500-token
   static conventions doc; the inter-agent transcript still flows in full. Real
   systems broadcast growing transcripts, shared memory, and large retrieved
   documents — far more to save on.
2. **Small, sparse graph.** A 4-stage near-linear pipeline has little broadcast
   redundancy. AgentProp should save more on dense, star,
   `hub_and_spoke_supervisor`, and `rag_pipeline` topologies with many agents
   reading from large shared `DOCUMENT`/`MEMORY` nodes.
3. **Reasoning-dominated cost.** Per-agent completion tokens dwarf the input tokens
   routing controls; in context-heavy workflows the input side is the dominant
   cost and AgentProp's lever is larger.

**Recommended next experiment:** run this same protocol on
`hub_and_spoke_supervisor` or `rag_pipeline` with large shared `DOCUMENT`/`MEMORY`
nodes, measuring input vs output tokens separately. That is where broadcasting
context to every agent is genuinely expensive and AgentProp should show its real
value.

## Roadmap — how AgentProp gets better

Each item is motivated directly by the evidence above and is enabled by the data
this harness now produces (per-node context level → real pass/fail + tokens).

1. **Role/criticality-aware seed selection.** Weight nodes by a
   *context-sensitivity* score (how much does success drop if this node loses full
   context?) and forbid starving high-sensitivity nodes. The unused
   `AgentNode.importance_score` field is the natural hook into
   `greedy_seed_selection`.
2. **Quality-aware objective.** Optimize `expected_success − λ·tokens` instead of
   coverage-vs-cost alone. `expected_success` can be a small model trained on trace
   outcomes — which this experiment generates.
3. **Calibrate the cost model from measured tokens.** Replace the fixed `0.35×`
   non-seed compression factor in `seeded_routing_cost` with a per-node/per-edge
   ratio fit from real `usage`, and split input vs output tokens.
4. **Graded compression, not binary full/summary.** Route context *volume*
   proportional to `AgentEdge.relevance × node_sensitivity` rather than an
   all-or-nothing cliff.
5. **Co-optimize verifier placement with routing.** Place/strengthen a verifier
   immediately downstream of any summary-starved high-sensitivity node.
6. **Risk-annotated recommendations.** Emit a warning when a routing choice starves
   a critical node ("saves 16% but routes the coder to summary — success at risk"),
   so a human or guard can veto.
7. **Online, category-conditioned adaptation.** Turn the existing (tabular,
   offline) RL env into a per-task-category bandit over routing policies that
   updates on real pass/fail. Failures and savings cluster by task type.

## Reproduce

```bash
# Real run (needs an OpenAI-compatible endpoint + key):
TOKEN_ROUTER_API_KEY=... \
TOKEN_ROUTER_BASE_URL=https://.../v1 \
TOKEN_ROUTER_MODEL=<model> \
PYTHONPATH=src python experiments/run_real_routing_case_study.py --max-tokens 4000

# Plumbing self-test (no key; validates the task suites against reference solutions):
PYTHONPATH=src python experiments/run_real_routing_case_study.py --fake

# Regenerate REPORT.md from saved results.json (no API calls):
PYTHONPATH=src python experiments/run_real_routing_case_study.py --report-only
```
