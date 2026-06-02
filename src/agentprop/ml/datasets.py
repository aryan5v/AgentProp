"""Dataset builders for learned AgentProp policies."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.algorithms import (
    greedy_seed_selection,
    low_weight_edges,
    risk_aware_verifier_placement,
)
from agentprop.core import AgentGraph, NodeType
from agentprop.ml.features import (
    EdgeFeatures,
    GraphFeatures,
    extract_edge_features,
    extract_graph_features,
)
from agentprop.propagation import IndependentCascade, PropagationModel


@dataclass(slots=True)
class SeedSelectionExample:
    """Supervised example for node-scoring seed policies."""

    features: GraphFeatures
    labels: dict[str, float]
    positive_seeds: list[str]
    budget: int
    neighbors: dict[str, list[str]]
    edge_features: EdgeFeatures | None = None


@dataclass(slots=True)
class SeedRankingExample:
    """Preference and regression targets for seed-ranking policies."""

    features: GraphFeatures
    utility_targets: dict[str, float]
    preference_pairs: list[tuple[str, str]]
    budget: int
    seed_candidates: list[str]


@dataclass(slots=True)
class EdgePruningExample:
    """Supervised example for edge-pruning policies."""

    features: EdgeFeatures
    labels: dict[tuple[str, str], float]
    positive_edges: list[tuple[str, str]]
    sample_weight: float = 1.0


@dataclass(slots=True)
class EmpiricalEdgePruningExample:
    """Edge-pruning example labeled by real task outcome."""

    features: EdgeFeatures
    labels: dict[tuple[str, str], float]
    outcome_score: float
    task_id: str
    policy: str
    pruned_edges: list[tuple[str, str]]
    task_category: str | None = None
    cost_adjusted_success: float | None = None
    sample_weight: float = 1.0


@dataclass(slots=True)
class VerifierPlacementExample:
    """Supervised example for verifier-placement policies."""

    features: GraphFeatures
    labels: dict[str, float]
    positive_verifiers: list[str]
    budget: int
    neighbors: dict[str, list[str]]
    edge_features: EdgeFeatures | None = None


@dataclass(slots=True)
class EmpiricalVerifierPlacementExample:
    """Verifier-placement example labeled by real task outcome."""

    features: GraphFeatures
    labels: dict[str, float]
    outcome_score: float
    task_id: str
    policy: str
    activated_verifiers: list[str]
    budget: int
    task_category: str | None = None
    cost_adjusted_success: float | None = None
    sample_weight: float = 1.0
    neighbors: dict[str, list[str]] | None = None
    edge_features: EdgeFeatures | None = None


@dataclass(slots=True)
class EmpiricalRoutingExample:
    """Node-policy example labeled by real task outcome instead of a heuristic teacher."""

    features: GraphFeatures
    labels: dict[str, float]
    outcome_score: float
    task_id: str
    policy: str
    selected_seeds: list[str]
    context_allocations: dict[str, float]
    budget: int
    task_category: str | None = None
    cost_adjusted_success: float | None = None
    sample_weight: float = 1.0
    neighbors: dict[str, list[str]] | None = None
    edge_features: EdgeFeatures | None = None


def build_seed_selection_example(
    graph: AgentGraph,
    *,
    budget: int,
    propagation_model: PropagationModel | None = None,
    trials: int = 50,
) -> SeedSelectionExample:
    """Label nodes by whether greedy influence maximization selected them."""

    model = propagation_model or IndependentCascade(seed=0)
    positive_seeds = greedy_seed_selection(
        graph,
        budget,
        propagation_model=model,
        trials=trials,
    )
    positives = set(positive_seeds)
    features = extract_graph_features(graph)
    labels = {node_id: 1.0 if node_id in positives else 0.0 for node_id in features.node_features}
    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    return SeedSelectionExample(
        features=features,
        labels=labels,
        positive_seeds=positive_seeds,
        budget=budget,
        neighbors=neighbors,
        edge_features=extract_edge_features(graph),
    )


def build_empirical_routing_example(
    graph: AgentGraph,
    row: dict[str, object],
    *,
    high_context_threshold: float = 0.80,
    default_budget: int | None = None,
) -> EmpiricalRoutingExample | None:
    """Build a node-label example from a real routed task row.

    Rows with retryable infra/timeout labels are skipped because they should not
    train correctness or routing quality. Positive labels come from high-context
    or selected-seed nodes in successful rows; the same choices receive zero
    labels when the task failed.
    """

    if bool(row.get("retry_recommended")):
        return None

    outcome = _empirical_outcome(row)
    if outcome is None:
        return None

    selected_seeds = _string_list(row.get("selected_seeds"))
    context_allocations = _string_float_dict(row.get("context_allocations"))
    budget = default_budget if default_budget is not None else max(1, len(selected_seeds))
    features = extract_graph_features(graph)
    labels = {}
    credited_nodes = set(selected_seeds)
    credited_nodes.update(
        node_id
        for node_id, ratio in context_allocations.items()
        if ratio >= high_context_threshold
    )
    for node_id in features.node_features:
        labels[node_id] = outcome if node_id in credited_nodes else 0.0

    return EmpiricalRoutingExample(
        features=features,
        labels=labels,
        outcome_score=outcome,
        task_id=str(row.get("task_id") or row.get("task_name") or "unknown-task"),
        policy=str(row.get("policy") or "unknown-policy"),
        selected_seeds=selected_seeds,
        context_allocations=context_allocations,
        budget=budget,
        task_category=_optional_string(row.get("category")),
        cost_adjusted_success=_optional_float(row.get("cost_adjusted_success")),
        sample_weight=_empirical_sample_weight(row),
        neighbors={
            node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
            for node_id in features.node_features
        },
        edge_features=extract_edge_features(graph),
    )


def build_empirical_routing_examples(
    graph: AgentGraph,
    rows: list[dict[str, object]],
    *,
    high_context_threshold: float = 0.80,
    default_budget: int | None = None,
) -> list[EmpiricalRoutingExample]:
    """Build all usable empirical node-policy examples from result rows."""

    examples = []
    for row in rows:
        example = build_empirical_routing_example(
            graph,
            row,
            high_context_threshold=high_context_threshold,
            default_budget=default_budget,
        )
        if example is not None:
            examples.append(example)
    return examples


def build_empirical_edge_pruning_example(
    graph: AgentGraph,
    row: dict[str, object],
) -> EmpiricalEdgePruningExample | None:
    """Build an edge-pruning example from a real routed task row.

    Only observed pruning decisions are credited. A pruned edge receives the
    task outcome as its target, so successful prunes become positive examples
    and failed prunes become negative examples. Rows with no pruned edges carry
    no direct pruning signal and are skipped.
    """

    if bool(row.get("retry_recommended")):
        return None

    outcome = _empirical_outcome(row)
    if outcome is None:
        return None

    pruned_edges = _edge_tuple_list(row.get("pruned_edges"))
    if not pruned_edges:
        return None

    features = extract_edge_features(graph)
    pruned = set(pruned_edges)
    labels = {
        edge_id: outcome if edge_id in pruned else 0.0
        for edge_id in features.edge_features
    }
    return EmpiricalEdgePruningExample(
        features=features,
        labels=labels,
        outcome_score=outcome,
        task_id=str(row.get("task_id") or row.get("task_name") or "unknown-task"),
        policy=str(row.get("policy") or "unknown-policy"),
        pruned_edges=pruned_edges,
        task_category=_optional_string(row.get("category")),
        cost_adjusted_success=_optional_float(row.get("cost_adjusted_success")),
        sample_weight=_empirical_sample_weight(row),
    )


def build_empirical_edge_pruning_examples(
    graph: AgentGraph,
    rows: list[dict[str, object]],
) -> list[EmpiricalEdgePruningExample]:
    """Build all usable empirical edge-pruning examples from result rows."""

    examples = []
    for row in rows:
        example = build_empirical_edge_pruning_example(graph, row)
        if example is not None:
            examples.append(example)
    return examples


def build_empirical_verifier_placement_example(
    graph: AgentGraph,
    row: dict[str, object],
    *,
    default_budget: int | None = None,
) -> EmpiricalVerifierPlacementExample | None:
    """Build a verifier-placement example from a real routed task row.

    Only observed verifier activations/placements are credited. A verifier node
    receives the task outcome as its target, so successful verifier choices
    become positive examples and failed choices become negative examples.
    """

    if bool(row.get("retry_recommended")):
        return None

    outcome = _empirical_outcome(row)
    if outcome is None:
        return None

    features = extract_graph_features(graph)
    activated_verifiers = [
        verifier
        for verifier in _observed_verifier_nodes(row)
        if verifier in features.node_features
    ]
    if not activated_verifiers:
        return None

    activated = set(activated_verifiers)
    labels = {
        node_id: outcome if node_id in activated else 0.0
        for node_id in features.node_features
    }
    budget = default_budget if default_budget is not None else max(1, len(activated_verifiers))
    return EmpiricalVerifierPlacementExample(
        features=features,
        labels=labels,
        outcome_score=outcome,
        task_id=str(row.get("task_id") or row.get("task_name") or "unknown-task"),
        policy=str(row.get("policy") or "unknown-policy"),
        activated_verifiers=activated_verifiers,
        budget=budget,
        task_category=_optional_string(row.get("category")),
        cost_adjusted_success=_optional_float(row.get("cost_adjusted_success")),
        sample_weight=_empirical_sample_weight(row),
        neighbors={
            node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
            for node_id in features.node_features
        },
        edge_features=extract_edge_features(graph),
    )


def build_empirical_verifier_placement_examples(
    graph: AgentGraph,
    rows: list[dict[str, object]],
    *,
    default_budget: int | None = None,
) -> list[EmpiricalVerifierPlacementExample]:
    """Build all usable empirical verifier-placement examples from result rows."""

    examples = []
    for row in rows:
        example = build_empirical_verifier_placement_example(
            graph,
            row,
            default_budget=default_budget,
        )
        if example is not None:
            examples.append(example)
    return examples


def build_seed_ranking_example(
    graph: AgentGraph,
    *,
    budget: int,
    propagation_model: PropagationModel | None = None,
    trials: int = 50,
    margin: float = 0.01,
) -> SeedRankingExample:
    """Build marginal-gain and pairwise-preference targets for seed policies."""

    model = propagation_model or IndependentCascade(seed=0)
    features = extract_graph_features(graph)
    seed_candidates = _seed_eligible_nodes(graph)
    utility_targets = {
        node_id: _single_seed_utility(graph, node_id, model, trials)
        for node_id in seed_candidates
    }
    for node_id in features.node_features:
        utility_targets.setdefault(node_id, 0.0)

    preference_pairs = []
    for winner in seed_candidates:
        for loser in seed_candidates:
            if winner == loser:
                continue
            if utility_targets[winner] > utility_targets[loser] + margin:
                preference_pairs.append((winner, loser))

    return SeedRankingExample(
        features=features,
        utility_targets=utility_targets,
        preference_pairs=preference_pairs,
        budget=budget,
        seed_candidates=seed_candidates,
    )


def build_edge_pruning_example(
    graph: AgentGraph,
    *,
    fraction: float = 0.2,
) -> EdgePruningExample:
    """Label low-weight edges as pruning candidates."""

    positive_edges = low_weight_edges(graph, fraction=fraction)
    positives = set(positive_edges)
    features = extract_edge_features(graph)
    labels = {
        edge_id: 1.0 if edge_id in positives else 0.0
        for edge_id in features.edge_features
    }
    return EdgePruningExample(
        features=features,
        labels=labels,
        positive_edges=positive_edges,
    )


def build_verifier_placement_example(
    graph: AgentGraph,
    *,
    budget: int,
) -> VerifierPlacementExample:
    """Label nodes selected by risk-aware verifier placement."""

    positive_verifiers = risk_aware_verifier_placement(graph, budget)
    positives = set(positive_verifiers)
    features = extract_graph_features(graph)
    labels = {
        node_id: 1.0 if node_id in positives else 0.0
        for node_id in features.node_features
    }
    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    return VerifierPlacementExample(
        features=features,
        labels=labels,
        positive_verifiers=positive_verifiers,
        budget=budget,
        neighbors=neighbors,
        edge_features=extract_edge_features(graph),
    )


def _seed_eligible_nodes(graph: AgentGraph) -> list[str]:
    return [node.id for node in graph.nodes() if node.type != NodeType.OUTPUT]


def _single_seed_utility(
    graph: AgentGraph,
    node_id: str,
    model: PropagationModel,
    trials: int,
) -> float:
    result = model.simulate(graph, [node_id], trials=trials)
    propagation_time = result.expected_propagation_time or result.propagation_time
    return result.coverage - 0.02 * float(propagation_time)


def _empirical_outcome(row: dict[str, object]) -> float | None:
    verification = row.get("verification_passed")
    if isinstance(verification, bool):
        return 1.0 if verification else 0.0
    quality_passed = row.get("quality_passed")
    if isinstance(quality_passed, bool):
        return 1.0 if quality_passed else 0.0
    quality_score = _optional_float(row.get("quality_score"))
    if quality_score is not None:
        return max(0.0, min(1.0, quality_score))
    passed = row.get("passed")
    if isinstance(passed, bool):
        return 1.0 if passed else 0.0
    return None


def _empirical_sample_weight(row: dict[str, object]) -> float:
    cost_adjusted = _optional_float(row.get("cost_adjusted_success"))
    if cost_adjusted is None:
        return 1.0
    return max(0.05, min(2.0, 1.0 + cost_adjusted))


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    if isinstance(value, tuple):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _string_float_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, raw_value in value.items():
        numeric = _optional_float(raw_value)
        if numeric is not None:
            result[str(key)] = max(0.0, min(1.0, numeric))
    return result


def _edge_tuple_list(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list | tuple):
        return []
    edges = []
    for item in value:
        if isinstance(item, list | tuple) and len(item) == 2:
            source, target = item
            if isinstance(source, str) and isinstance(target, str):
                edges.append((source, target))
        elif isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
            if isinstance(source, str) and isinstance(target, str):
                edges.append((source, target))
    return edges


def _observed_verifier_nodes(row: dict[str, object]) -> list[str]:
    for key in (
        "activated_verifiers",
        "verifier_nodes",
        "verifier_placements",
        "placed_verifiers",
        "recommended_verifiers",
    ):
        verifiers = _string_list(row.get(key))
        if verifiers:
            return verifiers
    return []


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None
