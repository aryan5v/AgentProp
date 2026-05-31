# Deep Learning Backend

AgentProp keeps deep-learning dependencies optional.

The core package exposes:

- graph feature extraction
- greedy-labeled seed-selection examples
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
- training loops that consume `SeedSelectionExample`

Run it with:

```bash
pip install "agentprop[dl]"
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture graphsage --epochs 100
```

The base training loop imitates greedy influence-maximization labels and scores
nodes as seed candidates. It is intentionally small enough to inspect while
still using real `torch.nn.Module` models and gradient descent.

Dependency-light ML baselines include:

- linear node scorer
- MLP node scorer
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
PYTHONPATH=src:. python experiments/train_edge_pruning_scorer.py
PYTHONPATH=src:. python experiments/evaluate_ml_generalization.py --model mlp
```

## Still Planned

- larger torch experiments that compare ranking/regression heads with GNN/RL
  policies

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
