# Statistics and Learning

Tools for putting uncertainty, calibration, and learned policies behind
AgentProp recommendations. Load this when the user asks about confidence
intervals, "is this savings number real", guaranteed verification gates,
routing that improves with use, or evaluating a policy without redeploying it.

## Confidence intervals on anything stochastic

```python
from agentprop.evaluation import bootstrap_mean_interval, bootstrap_difference_interval

ci = bootstrap_mean_interval(per_run_tokens, seed=0)
print(ci)  # mean [95% CI lower, upper]

delta = bootstrap_difference_interval(treatment_tokens, control_tokens, seed=0)
```

`analyze()` reports already carry `seed_coverage` as an interval. When the
user compares two arms (with/without control), use the difference interval and
report whether it excludes zero — never a bare point estimate.

## Calibrated FORCE_VERIFY gate (conformal)

```python
from agentprop.ml import ConformalRiskGate

gate = ConformalRiskGate(alpha=0.1)  # miss at most 10% of true failures
gate.calibrate(risk_scores, failed_outcomes)  # one score+label per past step
if gate.should_flag(current_risk):
    ...  # force verification
gate.save("reports/risk_gate.json")
```

The guarantee is distribution-free and finite-sample: with calibration data
exchangeable with live traffic, the expected miss rate on true failures is at
most `alpha`. Requires at least a handful of positive (failed) outcomes to
calibrate.

## Cascade-risk escalation (forward simulation)

```python
from agentprop.runtime import CascadeRiskAdvisor

advisor = CascadeRiskAdvisor(graph, impact_threshold=0.5)
decision = advisor.advise(controller_decision, features, node_id=active_node)
```

Simulates how a failure at the active node would propagate (Independent
Cascade over edge activation probabilities) and upgrades a borderline
CONTINUE to FORCE_VERIFY when the lower CI bound of downstream impact crosses
the threshold. It never downgrades a decision, so wrapping is always safe.

## Thompson-sampling routing

```python
from agentprop.rl import ThompsonSamplingRoutingPolicy

policy = ThompsonSamplingRoutingPolicy(
    arms=("broadcast", "quality-aware-greedy"), default_arm="broadcast"
)
arm = policy.choose(category)            # explores, decaying automatically
arm = policy.exploit(category)           # serving: posterior means only
policy.update(category, arm, reward=r, passed=ok)
```

Exploration decays as evidence accumulates; categories with fewer than
`min_successes` successful outcomes fall back to `default_arm`. Seed priors
from simulations or a similar workflow's history with `seed_prior`.

## Off-policy evaluation (would this policy have saved money?)

```python
from agentprop.rl import load_logged_decisions, weighted_importance_sampling, doubly_robust

logs = load_logged_decisions("reports/rewards.jsonl", behavior_probability=lambda row: 0.5)
result = doubly_robust(logs, target_policy=lambda cat: {"quality-aware-greedy": 1.0})
print(result.estimate, result.effective_sample_size)
```

Evaluates a candidate policy on logged reward records without redeploying.
Always report the effective sample size; below ~30 the estimate is not
trustworthy. Reward records come from `RuntimeRewardLogger` and include
graph-position features (schema v2, `docs/reward_record_schema.md`).

## Learned propagation that transfers

```python
from agentprop.propagation import FeatureCalibratedPropagation

model = FeatureCalibratedPropagation().fit(observations)  # pooled across workflows
p = model.edge_probability(new_graph, "planner", "coder")
result = model.simulate(new_graph, seeds=["planner"], trials=100)
```

Unlike `LearnedPropagation` (memorizes per-edge probabilities), this fits a
logistic model over edge features, so predictions work on graphs never seen
in training. Use it when the user has traces from several workflows and wants
calibrated propagation on a new one.
