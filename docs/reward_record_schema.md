# Reward Record Schema

Every controlled run logs one JSONL reward record via `RuntimeRewardLogger`
(`agentprop.runtime.control_loop`). Phase-2 contextual bandits and phase-3
policy transfer train on these historical records, so the schema is versioned.

## Schema version 2 (current)

Version 1 fields (unchanged):

| Field | Type | Meaning |
| --- | --- | --- |
| `task_id` | str | Caller-supplied task identifier |
| `category` | str | Task category used by the bandit |
| `strategy` | str | Arm chosen for the run |
| `action` | str | Last controller action (`CONTINUE`, `FORCE_VERIFY`, ...) |
| `passed` | bool | Independent task outcome |
| `token_savings` | float | Savings vs `baseline_tokens`, bounded [-1, 1] |
| `quality_score` / `quality_loss` | float? | Optional graded quality |
| `regression_risk` / `timeout_risk` | float | Risk signals at outcome time |
| `state` / `features` | object | `ExecutionStateFeatures` snapshot |
| `bandit_values` | object | Post-update arm values for the category |

Version 2 adds a `graph_features` object (`agentprop.rl.graph_features`):

| Field | Type | Meaning |
| --- | --- | --- |
| `graph_features.schema_version` | int | `REWARD_RECORD_SCHEMA_VERSION` (2) |
| `graph_features.workflow_embedding` | dict[str, float] | Mean/max pooled node + edge features, node-type histogram, node/edge counts, max DAG depth |
| `graph_features.resolving_coverage_active` | float | Resolving coverage of the active verifier set |
| `graph_features.active_verifiers` | list[str] | Verifier node ids active during the run |
| `graph_features.node` | object? | Present when a routing node is known: `node_id`, `node_type`, `depth`, `quality_score` (quality-cascade value at the node), `resolving_coverage_contribution` |

The workflow embedding is a flat name-to-float map; consumers can recover a
fixed-length vector by sorting keys. Records without `graph_features` are
schema version 1 and remain readable.

## Why graph-position features

Contextual bandits for LLM routing are well studied at the query level
(MixLLM, PILOT, BaRP). AgentProp's differentiator is the **workflow-graph
context** of each decision — where in the DAG the decision happens, how much
quality survives to that point, and how observable a failure there would be.
Logging these from the first real run means later learning phases start with
training data instead of a cold start. See the
[RL flywheel issues](https://github.com/aryan5v/AgentProp/issues/61) for the
phased plan.
