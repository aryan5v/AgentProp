# Deep Learning Backend

AgentProp keeps deep-learning dependencies optional.

The core package exposes:

- graph feature extraction
- greedy-labeled seed-selection examples
- empirical outcome-labeled routing examples
- pairwise seed-ranking examples
- marginal-gain regression targets
- a lightweight trainable scorer
- a message-passing-style scorer
- `agentprop.dl.GraphEncoderConfig`
- optional torch import guardrails
- optional torch-backed GNN seed scoring

## Torch Backend

The optional torch backend includes:

- GCN seed scorer
- GraphSAGE seed scorer
- GAT seed scorer
- GIN seed scorer
- Graph Transformer seed scorer
- Heterogeneous GNN seed scorer with node-type embeddings
- Edge-conditioned GNN seed scorer with full edge-feature gates over message
  cost, latency, relevance, reliability, activation probability, dependency
  strength, edge weight, and source/target degree
- training loops that consume `SeedSelectionExample`

Run it with:

```bash
pip install "agentprop[dl]"
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture graphsage --epochs 100
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture graph_transformer --epochs 100
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture edge_conditioned --epochs 100
```

The base training loop imitates greedy influence-maximization labels and scores
nodes as seed candidates. It carries the same rich edge feature matrix used by
edge-pruning models into torch GNN training so `edge_conditioned` sees workflow
cost, reliability, and dependency signals during both training and inference. It
is intentionally small enough to inspect while still using real `torch.nn.Module`
models and gradient descent.

Greedy-labeled examples are behavior-cloning baselines: they test whether a
model can approximate graph heuristics, not whether it can beat them. To train
against real task success, pass empirical routed rows with task outcomes,
selected seeds, and context allocations:

```bash
PYTHONPATH=src:. python experiments/train_seed_scorer.py \
  --model linear \
  --empirical-results results/case_study/results.json \
  --workflow planner_coder_tester_reviewer \
  --out results/ml/empirical_seed_scorer.json
```

Empirical training skips retry-recommended infra/timeout rows, then labels
high-context or selected-seed nodes by observed task success or quality score.
That is the path for learned policies to disagree with topology-only baselines
using real evidence.

Verifier-placement scorers can also train from empirical rows when rows include
observed `activated_verifiers`, `verifier_nodes`, or `verifier_placements`.
Successful verifier activations become positive examples; failed activations
become negative examples. Rows without observed verifier decisions are skipped.

Edge-pruning scorers can use the same empirical row format when rows include
`pruned_edges`. Successful pruned edges become positive examples; failed pruned
edges become negative examples. Rows without observed pruning decisions are
skipped because they do not identify which edge choice affected quality.

Dependency-light ML baselines include:

- linear node scorer
- MLP node scorer with trainable hidden and output layers
- pairwise node ranker
- marginal-gain node regressor
- linear edge-pruning scorer
- verifier-placement labels from risk-aware verifier placement
- held-out workflow generalization evaluation

Run them with:

```bash
PYTHONPATH=src:. python experiments/train_seed_scorer.py --model mlp --task seed
PYTHONPATH=src:. python experiments/train_seed_scorer.py --model pairwise --task seed
PYTHONPATH=src:. python experiments/train_seed_scorer.py --model regression --task seed
PYTHONPATH=src:. python experiments/train_seed_scorer.py --model mlp --task verifier
PYTHONPATH=src:. python experiments/train_seed_scorer.py --task verifier --empirical-results results/case_study/results.json
PYTHONPATH=src:. python experiments/train_edge_pruning_scorer.py
PYTHONPATH=src:. python experiments/train_edge_pruning_scorer.py --empirical-results results/case_study/results.json
PYTHONPATH=src:. python experiments/evaluate_ml_generalization.py --model mlp
```

Use `--l2-penalty` on lightweight ML training scripts to reduce memorization on
small workflow collections:

```bash
PYTHONPATH=src:. python experiments/train_seed_scorer.py --model mlp --l2-penalty 0.001
PYTHONPATH=src:. python experiments/train_edge_pruning_scorer.py --l2-penalty 0.001
```

Dependency-light ML scorers can be saved as JSON checkpoints:

```python
from agentprop.ml import load_ml_model, save_ml_model

save_ml_model(scorer, "results/models/mlp_seed_scorer.json", metadata={"task": "seed"})
checkpoint = load_ml_model("results/models/mlp_seed_scorer.json")
scorer = checkpoint.model
```

Experiment runs can also register checkpoints and metrics into a model registry:

```bash
PYTHONPATH=src:. python experiments/train_seed_scorer.py \
  --model mlp \
  --registry-root results/ml_core/model_registry \
  --run-id mlp_seed_scorer

PYTHONPATH=src:. python experiments/run_rl_routing.py \
  --policy ppo \
  --registry-root results/ml_core/model_registry \
  --run-id ppo_routing_policy
```

The registry writes `registry.json` with artifact id, kind, checkpoint path,
metrics path, source script, tags, metadata, and timestamps. This gives ML/RL
experiments a stable artifact index without requiring a heavyweight tracking
server.

## ML Core Experiment Suite

AgentProp also includes a recipe-style ML/RL suite inspired by the config-first
discipline in `huggingface/ml-intern`: keep credentials in environment
variables, write every artifact under one private result root, and make runtime
expectations explicit.

Dry-run the suite:

```bash
PYTHONPATH=src:. python experiments/run_experiment_suite.py \
  --config configs/experiment_suites/ml_core.json \
  --artifact-root results/ml_core \
  --dry-run
```

Run a small local sweep:

```bash
PYTHONPATH=src:. python experiments/run_ml_rl_sweep.py \
  --config configs/sweeps/ml_rl_smoke.json \
  --artifact-root results/ml_rl_smoke
```

The sweep runner expands grid parameters, executes each configured experiment,
writes a `sweep_manifest.json`, registers metric artifacts, and reuses the model
registry hooks in training scripts for checkpoints.

Run one recipe:

```bash
PYTHONPATH=src:. python experiments/run_experiment_suite.py \
  --only ml_generalization_mlp \
  --artifact-root results/ml_core
```

The default suite runs:

- checkpointed MLP seed scoring
- checkpointed edge-pruning scoring
- held-out workflow ML generalization
- classical vs ML/GNN-style vs seed-only and expanded-action RL routing baselines
- checkpointed PPO trajectory export with final cost-adjusted proxy quality

## Still Planned

- larger torch sweeps that compare all GNN architectures with ranking,
  regression, and RL policies

## Dependency Policy

Core install:

```bash
pip install agentprop
```

Future DL install:

```bash
pip install "agentprop[dl]"
```

The base framework must remain usable without PyTorch.
