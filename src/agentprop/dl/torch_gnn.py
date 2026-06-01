"""Torch-backed GNN graph-policy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentprop.core import AgentGraph, NodeType
from agentprop.dl.encoders import GraphEncoderConfig, require_torch
from agentprop.ml.datasets import SeedSelectionExample, VerifierPlacementExample
from agentprop.ml.features import EdgeFeatures, extract_edge_features, extract_graph_features


@dataclass(slots=True)
class TorchTrainingResult:
    """Summary of a torch GNN training run."""

    architecture: str
    epochs: int
    final_loss: float
    losses: list[float]


class TorchGNNSeedScorer:
    """Torch-backed GNN node scorer."""

    def __init__(self, config: GraphEncoderConfig) -> None:
        self.config = config
        self.torch = require_torch()
        self.model = _build_model(self.torch, config)

    def score_nodes(self, graph: AgentGraph) -> dict[str, float]:
        """Return node probabilities for the configured graph-policy task."""

        batch = _graph_to_tensors(self.torch, graph, edge_feature_dim=self.config.edge_feature_dim)
        self.model.eval()
        with self.torch.no_grad():
            logits = self.model(
                batch["x"],
                batch["adjacency"],
                batch["node_type_ids"],
                batch["edge_features"],
            )
            probabilities = self.torch.sigmoid(logits).detach().cpu().tolist()
        seed_candidates = _eligible_nodes_for_task(graph, self.config.task)
        return {
            node_id: float(probability)
            for node_id, probability in zip(batch["node_ids"], probabilities, strict=True)
            if node_id in seed_candidates
        }


def train_torch_seed_scorer(
    examples: list[SeedSelectionExample | VerifierPlacementExample],
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
            batch = _example_to_tensors(
                torch,
                example,
                edge_feature_dim=scorer.config.edge_feature_dim,
            )
            logits = scorer.model(
                batch["x"],
                batch["adjacency"],
                batch["node_type_ids"],
                batch["edge_features"],
            )
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


def _graph_to_tensors(
    torch: Any,
    graph: AgentGraph,
    *,
    edge_feature_dim: int,
) -> dict[str, Any]:
    features = extract_graph_features(graph)
    node_ids = list(features.node_features)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    x = torch.tensor(
        [features.node_features[node_id] for node_id in node_ids],
        dtype=torch.float32,
    )
    adjacency = torch.eye(len(node_ids), dtype=torch.float32)
    edge_features = _edge_features_to_tensor(
        torch,
        extract_edge_features(graph),
        node_ids,
        index,
        edge_feature_dim=edge_feature_dim,
    )
    for edge in graph.edges():
        if edge.source in index and edge.target in index:
            source = index[edge.source]
            target = index[edge.target]
            adjacency[source, target] = float(edge.weight)
            adjacency[target, source] = max(adjacency[target, source], float(edge.weight))
    node_type_ids = torch.tensor(
        [_node_type_index(graph.node(node_id).type) for node_id in node_ids],
        dtype=torch.long,
    )
    return {
        "node_ids": node_ids,
        "x": x,
        "adjacency": adjacency,
        "node_type_ids": node_type_ids,
        "edge_features": edge_features,
    }


def _example_to_tensors(
    torch: Any,
    example: SeedSelectionExample | VerifierPlacementExample,
    *,
    edge_feature_dim: int,
) -> dict[str, Any]:
    node_ids = list(example.features.node_features)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    x = torch.tensor(
        [example.features.node_features[node_id] for node_id in node_ids],
        dtype=torch.float32,
    )
    y = torch.tensor([example.labels[node_id] for node_id in node_ids], dtype=torch.float32)
    adjacency = torch.eye(len(node_ids), dtype=torch.float32)
    edge_features = _edge_features_to_tensor(
        torch,
        example.edge_features,
        node_ids,
        index,
        edge_feature_dim=edge_feature_dim,
    )
    for node_id, neighbors in example.neighbors.items():
        if node_id not in index:
            continue
        source = index[node_id]
        for neighbor in neighbors:
            if neighbor in index:
                adjacency[source, index[neighbor]] = 1.0
                if edge_features[source, index[neighbor]].sum() == 0:
                    edge_features[source, index[neighbor]] = torch.ones(
                        edge_feature_dim,
                        dtype=torch.float32,
                    )
    node_type_ids = torch.zeros(len(node_ids), dtype=torch.long)
    return {
        "node_ids": node_ids,
        "x": x,
        "y": y,
        "adjacency": adjacency,
        "node_type_ids": node_type_ids,
        "edge_features": edge_features,
    }


def _edge_features_to_tensor(
    torch: Any,
    edge_features: EdgeFeatures | None,
    node_ids: list[str],
    index: dict[str, int],
    *,
    edge_feature_dim: int,
) -> Any:
    tensor = torch.zeros(
        (len(node_ids), len(node_ids), edge_feature_dim),
        dtype=torch.float32,
    )
    if edge_features is None:
        return tensor
    for (source_id, target_id), values in edge_features.edge_features.items():
        if source_id not in index or target_id not in index:
            continue
        source = index[source_id]
        target = index[target_id]
        encoded = _fit_edge_feature_dim(values, edge_feature_dim)
        tensor[source, target] = torch.tensor(encoded, dtype=torch.float32)
        tensor[target, source] = torch.tensor(encoded, dtype=torch.float32)
    return tensor


def _fit_edge_feature_dim(values: list[float], edge_feature_dim: int) -> list[float]:
    if edge_feature_dim < 1:
        raise ValueError("edge_feature_dim must be at least 1")
    if len(values) >= edge_feature_dim:
        return values[:edge_feature_dim]
    return [*values, *([0.0] * (edge_feature_dim - len(values)))]


def _build_model(torch: Any, config: GraphEncoderConfig) -> Any:
    architecture = config.architecture.lower()
    if architecture == "gcn":
        return _GCN(torch, config)
    if architecture == "graphsage":
        return _GraphSAGE(torch, config)
    if architecture == "gat":
        return _GAT(torch, config)
    if architecture == "gin":
        return _GIN(torch, config)
    if architecture == "graph_transformer":
        return _GraphTransformer(torch, config)
    if architecture == "heterogeneous":
        return _HeterogeneousGNN(torch, config)
    if architecture == "edge_conditioned":
        return _EdgeConditionedGNN(torch, config)
    raise ValueError(
        "architecture must be one of: gcn, graphsage, gat, gin, "
        "graph_transformer, heterogeneous, edge_conditioned"
    )


def _eligible_nodes_for_task(graph: AgentGraph, task: str) -> set[str]:
    if task == "verifier":
        return {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}
    if task == "seed":
        return {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}
    raise ValueError("task must be one of: seed, verifier")


def _activation(torch: Any, x: Any, dropout: Any, training: bool) -> Any:
    return dropout(torch.relu(x)) if training else torch.relu(x)


def _normalized_adjacency(torch: Any, adjacency: Any) -> Any:
    degree = adjacency.sum(dim=1).clamp(min=1.0)
    inv_sqrt_degree = degree.pow(-0.5)
    return inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)


def _node_type_index(node_type: NodeType) -> int:
    ordered_types = list(NodeType)
    return ordered_types.index(node_type) if node_type in ordered_types else 0


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

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
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

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
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

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
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


class _GIN:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.epsilon = torch.nn.Parameter(torch.tensor(0.0))
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.hidden = torch.nn.ModuleList(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim)
                    for _ in range(max(config.layers - 1, 0))
                )
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
                h = self._gin_layer(x, adjacency, self.input)
                for layer in self.hidden:
                    h = self._gin_layer(h, adjacency, layer)
                return self.output(h).squeeze(-1)

            def _gin_layer(self, h: Any, adjacency: Any, layer: Any) -> Any:
                neighbor_sum = adjacency @ h
                mixed = (1.0 + self.epsilon) * h + neighbor_sum
                return _activation(torch, layer(mixed), self.dropout, self.training)

        return Model()


class _GraphTransformer:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                heads = max(1, min(config.attention_heads, config.hidden_dim))
                while config.hidden_dim % heads != 0 and heads > 1:
                    heads -= 1
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.attention = torch.nn.MultiheadAttention(
                    config.hidden_dim,
                    heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.feed_forward = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.dropout),
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                self.norm_attention = torch.nn.LayerNorm(config.hidden_dim)
                self.norm_output = torch.nn.LayerNorm(config.hidden_dim)
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
                h = torch.relu(self.input(x)).unsqueeze(0)
                attn_mask = adjacency <= 0
                attended, _ = self.attention(h, h, h, attn_mask=attn_mask)
                h = self.norm_attention(h + attended)
                h = self.norm_output(h + self.feed_forward(h))
                return self.output(h.squeeze(0)).squeeze(-1)

        return Model()


class _HeterogeneousGNN:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.type_embedding = torch.nn.Embedding(
                    config.node_type_count,
                    config.hidden_dim,
                )
                self.hidden = torch.nn.ModuleList(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim)
                    for _ in range(max(config.layers - 1, 0))
                )
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
                norm = _normalized_adjacency(torch, adjacency)
                h = self.input(x)
                if node_type_ids is not None:
                    h = h + self.type_embedding(node_type_ids)
                h = _activation(torch, norm @ h, self.dropout, self.training)
                for layer in self.hidden:
                    h = _activation(torch, layer(norm @ h), self.dropout, self.training)
                return self.output(h).squeeze(-1)

        return Model()


class _EdgeConditionedGNN:
    def __new__(cls, torch: Any, config: GraphEncoderConfig) -> Any:
        class Model(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(config.input_dim, config.hidden_dim)
                self.edge_gate = torch.nn.Linear(config.edge_feature_dim, 1)
                self.hidden = torch.nn.ModuleList(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim)
                    for _ in range(max(config.layers - 1, 0))
                )
                self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)
                self.dropout = torch.nn.Dropout(config.dropout)

            def forward(
                self,
                x: Any,
                adjacency: Any,
                node_type_ids: Any | None = None,
                edge_features: Any | None = None,
            ) -> Any:
                h = _activation(torch, self.input(x), self.dropout, self.training)
                conditioned_adjacency = self._conditioned_adjacency(adjacency, edge_features)
                h = conditioned_adjacency @ h
                for layer in self.hidden:
                    h = _activation(
                        torch,
                        layer(conditioned_adjacency @ h),
                        self.dropout,
                        self.training,
                    )
                return self.output(h).squeeze(-1)

            def _conditioned_adjacency(self, adjacency: Any, edge_features: Any | None) -> Any:
                if edge_features is None:
                    return _normalized_adjacency(torch, adjacency)
                gates = torch.sigmoid(self.edge_gate(edge_features).squeeze(-1))
                weighted = adjacency * gates
                return _normalized_adjacency(torch, weighted)

        return Model()
