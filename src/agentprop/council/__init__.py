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

from agentprop.council.assignment import Assigner, Assignment, subtask_features
from agentprop.council.claim_check import (
    CheckedSubAnswer,
    ClaimChecker,
    evidence_support_risk,
)
from agentprop.council.council import Council, CouncilResult
from agentprop.council.model_pool import ModelPool, ModelResponse, ModelSpec
from agentprop.council.planner import LLMPlanner, Plan, SubTask, parse_plan
from agentprop.council.retrieval import (
    NullRetrieval,
    OpenRouterWebSearch,
    RetrievalResult,
    RetrievalTool,
)
from agentprop.council.synthesizer import SynthesisResult, Synthesizer

__all__ = [
    "Assigner",
    "Assignment",
    "CheckedSubAnswer",
    "ClaimChecker",
    "Council",
    "CouncilResult",
    "LLMPlanner",
    "ModelPool",
    "ModelResponse",
    "ModelSpec",
    "NullRetrieval",
    "OpenRouterWebSearch",
    "Plan",
    "RetrievalResult",
    "RetrievalTool",
    "SubTask",
    "SynthesisResult",
    "Synthesizer",
    "evidence_support_risk",
    "parse_plan",
    "subtask_features",
]
