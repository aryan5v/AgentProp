"""Runtime controller for executing AgentProp graphs with real context routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import networkx as nx

from agentprop.algorithms import greedy_seed_selection, quality_aware_greedy_seed_selection
from agentprop.core import AgentGraph, AgentNode, NodeType
from agentprop.evaluation import graded_context_allocations, routing_risks
from agentprop.propagation import IndependentCascade

SeedSelectorName = Literal["quality_aware", "greedy"]
ActivationMode = Literal["all", "propagated"]


class RuntimeNodeExecutor(Protocol):
    """Callable that executes one concrete node request."""

    def __call__(self, request: RuntimeNodeRequest) -> RuntimeNodeResult:
        """Execute one graph node."""


class ContextCompressor(Protocol):
    """Callable that compresses shared context for non-seed nodes."""

    def __call__(self, context: str, *, task: str, target_ratio: float) -> str:
        """Return a shorter context view."""


@dataclass(frozen=True, slots=True)
class RuntimeControllerConfig:
    """Execution policy for an AgentProp runtime run."""

    seed_budget: int = 2
    trials: int = 50
    seed_selector: SeedSelectorName = "quality_aware"
    seed: int = 0
    force_verifier_full_context: bool = True
    compressed_context_ratio: float = 0.35
    full_context_threshold: float = 0.95
    fixed_seeds: tuple[str, ...] = ()
    activation_mode: ActivationMode = "all"
    allow_cycles: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeNodeRequest:
    """The exact runtime input visible to one graph node."""

    node: AgentNode
    task: str
    visible_context: str
    context_ratio: float
    full_context: bool
    selected_seeds: tuple[str, ...]
    upstream_outputs: Mapping[str, str]
    all_outputs: Mapping[str, str]
    risks: tuple[dict[str, object], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeNodeResult:
    """Result emitted by one runtime node."""

    node_id: str
    output: str
    passed: bool | None = None
    intercept: bool = False
    token_cost: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeRunResult:
    """Complete AgentProp runtime execution trace."""

    selected_seeds: tuple[str, ...]
    activated_nodes: tuple[str, ...]
    context_ratios: Mapping[str, float]
    risks: tuple[dict[str, object], ...]
    node_results: tuple[RuntimeNodeResult, ...]
    final_output: str
    passed: bool | None
    trace_events: tuple[dict[str, object], ...]


class AgentPropRuntimeController:
    """Execute a workflow graph while AgentProp controls context and verification."""

    def __init__(
        self,
        graph: AgentGraph,
        *,
        config: RuntimeControllerConfig | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.graph = graph
        self.config = config or RuntimeControllerConfig()
        self.compressor = compressor

    def run(
        self,
        *,
        task: str,
        shared_context: str,
        executor: RuntimeNodeExecutor | Callable[[RuntimeNodeRequest], RuntimeNodeResult],
        metadata: Mapping[str, object] | None = None,
    ) -> RuntimeRunResult:
        """Run every executable graph node with controller-managed context."""

        selected_seeds = tuple(self._select_seeds())
        propagation = IndependentCascade(seed=self.config.seed).simulate(
            self.graph,
            list(selected_seeds),
            trials=self.config.trials,
        )
        if self.config.activation_mode == "propagated":
            activated_nodes = set(propagation.activated_nodes)
        else:
            activated_nodes = {node.id for node in self.graph.nodes()}
        context_ratios = graded_context_allocations(
            self.graph,
            seeds=list(selected_seeds),
            activated_nodes=activated_nodes,
            min_ratio=self.config.compressed_context_ratio,
        )
        if self.config.force_verifier_full_context:
            for node in self.graph.nodes():
                if node.type == NodeType.VERIFIER:
                    context_ratios[node.id] = 1.0

        risks = tuple(
            risk.to_dict() for risk in routing_risks(self.graph, context_ratios=context_ratios)
        )
        compressed_context_cache: dict[float, str] = {}
        outputs: dict[str, str] = {}
        results: list[RuntimeNodeResult] = []
        trace_events: list[dict[str, object]] = []
        run_metadata = dict(metadata or {})

        for node in self._execution_order():
            ratio = context_ratios.get(node.id, 0.0)
            full_context = ratio >= self.config.full_context_threshold
            visible_context = self._visible_context(
                task=task,
                shared_context=shared_context,
                ratio=ratio,
                full_context=full_context,
                cache=compressed_context_cache,
            )
            upstream = {
                source: outputs[source]
                for source in self.graph.predecessors(node.id)
                if source in outputs
            }
            request = RuntimeNodeRequest(
                node=node,
                task=task,
                visible_context=visible_context,
                context_ratio=ratio,
                full_context=full_context,
                selected_seeds=selected_seeds,
                upstream_outputs=upstream,
                all_outputs=dict(outputs),
                risks=risks,
                metadata=run_metadata,
            )
            result = executor(request)
            if result is None:
                raise RuntimeError(f"Executor returned None for node {node.id!r}")
            if result is None:
                raise ValueError(f"Executor returned None for node {node.id!r}")
            if result.node_id != node.id:
                raise ValueError(f"Executor returned {result.node_id!r} for node {node.id!r}")
            results.append(result)
            outputs[node.id] = result.output
            trace_events.append(
                {
                    "node_id": node.id,
                    "node_type": node.type.value,
                    "context_ratio": ratio,
                    "full_context": full_context,
                    "is_seed": node.id in selected_seeds,
                    "passed": result.passed,
                    "intercept": result.intercept,
                    "token_cost": result.token_cost,
                }
            )
            if result.intercept:
                break

        final = results[-1] if results else None
        return RuntimeRunResult(
            selected_seeds=selected_seeds,
            activated_nodes=tuple(sorted(activated_nodes)),
            context_ratios=dict(context_ratios),
            risks=risks,
            node_results=tuple(results),
            final_output=final.output if final is not None else "",
            passed=final.passed if final is not None else None,
            trace_events=tuple(trace_events),
        )

    def _select_seeds(self) -> list[str]:
        if self.config.fixed_seeds:
            return list(self.config.fixed_seeds[: self.config.seed_budget])
        model = IndependentCascade(seed=self.config.seed)
        if self.config.seed_selector == "greedy":
            return greedy_seed_selection(
                self.graph,
                self.config.seed_budget,
                propagation_model=model,
                trials=self.config.trials,
            )
        return quality_aware_greedy_seed_selection(
            self.graph,
            self.config.seed_budget,
            propagation_model=model,
            trials=self.config.trials,
        )

    def _execution_order(self) -> list[AgentNode]:
        nx_graph = self.graph.to_networkx()
        if nx.is_directed_acyclic_graph(nx_graph):
            node_ids = list(nx.topological_sort(nx_graph))
            return [self.graph.node(str(node_id)) for node_id in node_ids]
        if not self.config.allow_cycles:
            raise ValueError(
                "The workflow graph contains cycles, which is not supported "
                "for single-pass execution."
            )
        return self.graph.nodes()

    def _visible_context(
        self,
        *,
        task: str,
        shared_context: str,
        ratio: float,
        full_context: bool,
        cache: dict[float, str],
    ) -> str:
        if full_context:
            return shared_context
        if ratio <= 0.0:
            return ""
        rounded_ratio = round(ratio, 2)
        if rounded_ratio not in cache:
            if self.compressor is not None:
                cache[rounded_ratio] = self.compressor(
                    shared_context,
                    task=task,
                    target_ratio=rounded_ratio,
                )
            else:
                cache[rounded_ratio] = _truncate_context(shared_context, rounded_ratio)
        return cache[rounded_ratio]


def _truncate_context(context: str, ratio: float) -> str:
    if not context:
        return ""
    bounded_ratio = max(0.0, min(1.0, ratio))
    words = context.split()
    if len(words) <= 1:
        return context[: max(1, int(len(context) * bounded_ratio))]
    keep = max(1, int(len(words) * bounded_ratio))
    return " ".join(words[:keep])
