# Learned Propagation

AgentProp includes a trace-calibrated propagation model named `learned`.

The model estimates edge activation probabilities from workflow traces, then
runs an Independent-Cascade-style simulation with those learned probabilities.
This is the bridge between synthetic graph propagation and real LLM workflow
behavior.

## Trace Format

The model consumes the same trace events as trace ingestion:

```json
{
  "events": [
    {
      "source": "planner",
      "target": "coder",
      "success": true,
      "token_cost": 500,
      "latency": 0.7
    }
  ]
}
```

## Training

```bash
PYTHONPATH=src:. python experiments/train_learned_propagation.py \
  --trace traces/workflow_run.json \
  --seed-node planner \
  --out results/propagation/learned.json
```

## CLI Usage

After converting a trace to workflow JSON, the learned model can be used through
the existing CLI:

```bash
agentprop trace traces/workflow_run.json --out results/trace_workflow.json
agentprop simulate results/trace_workflow.json --model learned --seeds planner
agentprop optimize results/trace_workflow.json --model learned --budget 2
```

## Current Semantics

For a source node, the model estimates:

```text
P(target activates | source active)
```

using smoothed successful edge counts divided by source event counts. If a graph
has trace-derived metadata but no separately fitted model, the fallback learned
model reads edge reliability and activation probability from the graph.

## Next Step

Once real LLM case-study traces exist, this model should be compared against
Independent Cascade, Linear Threshold, Randomized Zero Forcing, and learned
GNN/RL policies.

## Feature-calibrated propagation (transfers to unseen graphs)

`LearnedPropagation` memorizes per-edge probabilities, so it cannot score an
edge it never saw. `FeatureCalibratedPropagation`
(`agentprop.propagation.feature_calibrated`) instead fits a logistic model
from edge features (relevance, reliability, degrees, costs) to activation
probability, pooled across any number of workflows — predictions then work on
structurally new graphs. `fit` takes `(graph, {(source, target): activated})`
observations; `simulate` runs IC with the predicted probabilities; `save` /
`load` round-trip the parameters as JSON.
