"""Built-in workflow templates."""

from agentprop.workflows.templates import (
    WORKFLOW_TEMPLATES,
    debate_judge,
    hub_and_spoke_supervisor,
    planner_coder_tester_reviewer,
    rag_pipeline,
    research_writer_verifier,
    tool_use_pipeline,
)
from agentprop.workflows.export import export_builtin_workflows

__all__ = [
    "WORKFLOW_TEMPLATES",
    "debate_judge",
    "hub_and_spoke_supervisor",
    "planner_coder_tester_reviewer",
    "rag_pipeline",
    "research_writer_verifier",
    "tool_use_pipeline",
    "export_builtin_workflows",
]
