"""Torch-backed GNN seed-selection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentprop.core import AgentGraph, NodeType
from agentprop.dl.encoders import GraphEncoderConfig, require_torch
from agentprop.ml.datasets import SeedSelectionExample
from agentprop.ml.features import extract_graph_features


@dataclass(slots=True)
class TorchTrainingResult:
    """Summary of a torch GNN training run."""

    architecture: str
    epochs: int
    final_loss: float
    losses: list[float]


class TorchGNNSeedScorer:
    """Torch-backed GCN, GraphSAGE, or GAT node scorer."""

    def __init__(self, config: GraphEncoderConfig) -> None:
        self.config = config
        self.torch = require_torch()
        self.model = _build_model(self.torch, config)

    def score_nodes(self, graph: AgentGraph) -> dict[str, float]:
        """Return node seed probabilities for a graph."""

        batch = _graph_to_tensors(self.torch, graph)
        self.model.eval()
        with self.torch.no_grad():
            logits = self.model(batch["x"], batch["adjacency"])
            probabilities = self.torch.sigmoid(logits).detach().cpu().tolist()
        seed_candidates = {
            node.id for node in graph.nodes() if node.type != NodeType.OUTPUT
        }
        return {
            node_id: float(probability)
            for node_id, probability in zip(batch["node_ids"], probabilities, strict=True)
            if node_id in seed_candidates
        }


def train_torch_seed_scorer(
    examples: list[SeedSelectionExample],
    *,
    config: GraphEncoderConfig | None = None,
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> tuple[TorchGNNSeedScorer, TorchTrainingResult]:
    """Train a torch GNN scorer with greedy seed-selection labels."""

    if not examples:
        raise ValueError("examples must not be empty")

    feature_count = len(examples[0].features.feature_names)
    scorer = TorchGNNSeedScorer(config or GraphEncoderConfig(input_dim=feature_count))
    torch = scorer.torch
    optimizer = torch.optim.Adam(scorer.model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []

    for _ in range(epochs):
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        for example in examples:
            batch = _example_to_tensors(torch, example)
            logits = scorer.model(batch["x"], batch["adjacency"])
            total_loss = total_loss + loss_fn(logits, batch["y"])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))

    return scorer, TorchTrainingResult(
        architecture=scorer.config.architecture,
        epochs=epochs,
        final_loss=losses[-1],
        losses=losses,
    )


def _graph_to_tensors(torch: Any, graph: AgentGraph) -> dict[str, Any]:
    features = extract_graph_features(graph)
    node_ids = list(features.node_features)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    x = torch.tensor(
        [features.node_features[node_id] for node_id in node_ids],
        dtype=torch.float32,
    )
    adjacency = torch.eye(len(node_ids), dtype=torch.float32)
    for edge in graph.edges():
        if edge.source in index and edge.target in index:
            source = index[edge.source]
            target = index[edge.target]
            adjacency[source, target] = float(edge.weight)
            adjacency[target, source] = max(adjacency[target, source], float(edge.weight))
    return {"node_ids": node_ids, "x": x, "adjacency": adjacency}


def _example_to_tensors(torch: Any, example: SeedSelectionExample) -> dict[str, Any]:
    node_ids = list(example.features.node_features)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    x = torch.tensor(
        [example.features.node_features[node_id] for node_id in node_ids],
        dtype=torch.float32,
    )
    y = torch.tensor([example.labels[node_id] for node_id in node_ids], dtype=torch.float32)
    adjacency = torch.eye(len(node_ids), dtype=torch.float32)
    for node_id, neighbors in example.neighbors.items():
        if node_id not in index:
            continue
        source = index[node_id]
        for neighbor in neighbors:
            if neighbor in index:
                adjacency[source, index[neighbor]] = 1.0
    return {"node_ids": node_ids, "x": x, "y": y, "adjacency": adjacency}


def _build_model(torch: Any, config: GraphEncoderConfig) -> Any:
    architecture = config.architecture.lower()
    if architecture == "gcn":
        return _GCN(torch, config)
    if architecture == "graphsage":
        return _GraphSAGE(torch, config)
    if architecture == "gat":
        return _GAT(torch, config)
    raise ValueError("architecture must be one of: gcn, graphsage, gat")


def _activation(torch: Any, x: Any, dropout: Any, training: bool) -> Any:
    return dropout(torch.relu(x)) if training else torch.relu(x)


def _normalized_adjacency(torch: Any, adjacency: Any) -> Any:
    degree = adjacency.sum(dim=1).clamp(min=1.0)
    inv_sqrt_degree = degree.pow(-0.5)
    return inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)


class _GCN:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.hidden = torch.nn.ModuleList(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim)
                    for _ in range(max(config.layers - 1, 0))
                )
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(self, x: Any, adjacency: Any) -> Any:
                norm = _normalized_adjacency(torch, adjacency)
                h = _activation(torch, self.input(norm @ x), self.dropout, self.training)
                for layer in self.hidden:
                    h = _activation(torch, layer(norm @ h), self.dropout, self.training)
                return self.output(norm @ h).squeeze(-1)

        return Model()


class _GraphSAGE:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(config.input_dim * 2, config.hidden_dim)
                self.hidden = torch.nn.ModuleList(
                    torch.nn.Linear(config.hidden_dim * 2, config.hidden_dim)
                    for _ in range(max(config.layers - 1, 0))
                )
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(self, x: Any, adjacency: Any) -> Any:
                h = self._layer(x, adjacency, self.input)
                for layer in self.hidden:
                    h = self._layer(h, adjacency, layer)
                return self.output(h).squeeze(-1)

            def _layer(self, h: Any, adjacency: Any, layer: Any) -> Any:
                degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
                neighbor_mean = adjacency @ h / degree
                combined = torch.cat([h, neighbor_mean], dim=1)
                return _activation(torch, layer(combined), self.dropout, self.training)

        return Model()


class _GAT:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.attention = torch.nn.Linear(config.hidden_dim * 2, 1)
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(self, x: Any, adjacency: Any) -> Any:
                h = torch.relu(self.input(x))
                source = h.unsqueeze(1).repeat(1, h.shape[0], 1)
                target = h.unsqueeze(0).repeat(h.shape[0], 1, 1)
                scores = self.attention(torch.cat([source, target], dim=-1)).squeeze(-1)
                scores = scores.masked_fill(adjacency <= 0, -1e9)
                attention = torch.softmax(scores, dim=1)
                attended = attention @ h
                attended = self.dropout(torch.relu(attended))
                return self.output(attended).squeeze(-1)

        return Model()
