# Quality-Aware Routing

AgentProp routes context through multi-agent workflows with an explicit
cost-quality tradeoff. The default routing stack now protects nodes that are
semantically sensitive to context, not only nodes that are central in the graph.

## Role and Criticality

Each `AgentNode` can define `importance_score` on a `0..1` scale. Seed selection
uses that score to avoid starving critical nodes of context. In coding workflows,
the built-in `planner_coder_tester_reviewer` template marks the coder and tester
as high-sensitivity nodes.

```python
from agentprop.algorithms import quality_aware_greedy_seed_selection
from agentprop.propagation import IndependentCascade
from agentprop.workflows import planner_coder_tester_reviewer

graph = planner_coder_tester_reviewer()
seeds = quality_aware_greedy_seed_selection(
    graph,
    2,
    propagation_model=IndependentCascade(seed=0),
)
```

## Quality-Aware Objective

`QualityAwareRoutingObjective` scores a routing plan as:

```text
expected_success - lambda * total_cost
```

The fallback estimator uses reliability, error rate, context allocation, and
node importance when no calibration data is available.
When routed task rows include pass/fail or quality outcomes plus
`context_allocations`, fit an empirical expected-success profile:

```python
from agentprop.evaluation import (
    QualityAwareRoutingObjective,
    calibrate_expected_success,
)

profile = calibrate_expected_success(rows)
objective = QualityAwareRoutingObjective(success_profile=profile)
```

The empirical profile learns which nodes appear context-sensitive in real runs.
For example, if compressed coder context repeatedly coincides with failed
verification while full coder context passes, future coder-starved plans receive
a lower expected-success score.

## Calibrated and Graded Context

Binary full-context versus summary-only routing is often too coarse. AgentProp
therefore computes per-node context allocations:

- seeds receive `100%`
- inactive nodes receive `0%`
- active non-seed nodes receive a graded allocation based on measured
  compression ratios, incoming edge relevance, and node sensitivity

`calibrate_context_compression()` can fit per-node compression ratios from
case-study rows that include `stage_tokens` or `stage_prompt_tokens`.

## Risk Signals

Every `RecommendationReport` includes:

- `context_allocations`
- `routing_risks`
- `quality_objective_score`

Risk signals are emitted when a high-sensitivity node receives compressed
context. Reports and JSON output make these risks visible so a developer or
guardrail can veto a recommendation that saves tokens at unacceptable quality
risk.

## Verifier Coupling

`context_sensitive_verifier_placement()` ranks verifier locations using both
error-propagation centrality and compression risk. This couples verifier
placement to routing: if a high-sensitivity node receives less context, the next
downstream verifier should receive stronger attention.

## Online Adaptation

`CategoryBanditRoutingPolicy` is a lightweight online bandit over routing
policies. It updates per task category using real pass/fail, quality score, and
token savings, which lets AgentProp learn that some task families need more
context on implementation nodes while others can tolerate more aggressive
compression.
