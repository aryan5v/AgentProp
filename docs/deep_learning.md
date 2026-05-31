# Deep Learning Roadmap

AgentProp keeps deep-learning dependencies optional.

The core package already exposes:

- graph feature extraction
- greedy-labeled seed-selection examples
- a lightweight trainable scorer
- a message-passing-style scorer
- `agentprop.dl.GraphEncoderConfig`
- optional torch import guardrails

## Planned Torch Backend

The torch-backed backend should add:

- GCN seed scorer
- GraphSAGE seed scorer
- GAT seed scorer
- edge-pruning scorer
- verifier-placement scorer
- training loops that consume `SeedSelectionExample`

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
