# Deep Learning Backend

AgentProp keeps deep-learning dependencies optional.

The core package exposes:

- graph feature extraction
- greedy-labeled seed-selection examples
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
- training loops that consume `SeedSelectionExample`

Run it with:

```bash
pip install "agentprop[dl]"
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture graphsage --epochs 100
```

The base training loop imitates greedy influence-maximization labels and scores
nodes as seed candidates. It is intentionally small enough to inspect while
still using real `torch.nn.Module` models and gradient descent.

## Still Planned

- edge-pruning scorer
- verifier-placement scorer
- benchmark comparison between training-free, GNN, and RL policies

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
