"""Built-in workflow templates."""

from agentprop.workflows.export import export_builtin_workflows
from agentprop.workflows.templates import (
    WORKFLOW_DESCRIPTIONS,
    WORKFLOW_TEMPLATES,
    chain_workflow,
    debate_judge,
    dense_workflow,
    generic_dag_workflow,
    hub_and_spoke_supervisor,
    inject_quality_decay,
    layered_pipeline_workflow,
    planner_coder_tester_reviewer,
    rag_pipeline,
    random_directed_workflow,
    research_writer_verifier,
    small_world_workflow,
    star_workflow,
    tool_use_pipeline,
    tree_workflow,
)

__all__ = [
    "WORKFLOW_DESCRIPTIONS",
    "WORKFLOW_TEMPLATES",
    "chain_workflow",
    "debate_judge",
    "dense_workflow",
    "generic_dag_workflow",
    "hub_and_spoke_supervisor",
    "inject_quality_decay",
    "layered_pipeline_workflow",
    "planner_coder_tester_reviewer",
    "rag_pipeline",
    "random_directed_workflow",
    "research_writer_verifier",
    "small_world_workflow",
    "star_workflow",
    "tree_workflow",
    "tool_use_pipeline",
    "export_builtin_workflows",
]
