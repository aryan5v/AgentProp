"""Council: supervise multiple models as one graph for any multi-step task.

The Council decomposes a task into a sub-task DAG, assigns each sub-task to the
cheapest capable model in a pool, executes with optional retrieval tools,
verifies and quarantines unsupported outputs, then quality-weighted synthesizes
a final answer — all under an AgentProp ``ControlSession``.

It is task-, model-, and topology-agnostic: deep research, coding, and
multi-agent workflows are the same machinery with different scorers, tools, and
decompositions. Benchmark-specific logic (e.g. DRACO) lives in
``agentprop.evaluation``, never here.
"""

from agentprop.council.model_pool import ModelPool, ModelResponse, ModelSpec
from agentprop.council.retrieval import (
    NullRetrieval,
    OpenRouterWebSearch,
    RetrievalResult,
    RetrievalTool,
)

__all__ = [
    "ModelPool",
    "ModelResponse",
    "ModelSpec",
    "NullRetrieval",
    "OpenRouterWebSearch",
    "RetrievalResult",
    "RetrievalTool",
]
