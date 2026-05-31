import pytest

from agentprop.core import AgentGraph, WorkflowValidationError


def test_workflow_validation_rejects_unknown_edge_endpoint() -> None:
    with pytest.raises(WorkflowValidationError, match="unknown target node"):
        AgentGraph.from_dict(
            {
                "nodes": [{"id": "planner"}],
                "edges": [{"source": "planner", "target": "missing"}],
            }
        )


def test_workflow_validation_rejects_invalid_probability() -> None:
    with pytest.raises(WorkflowValidationError, match="between 0 and 1"):
        AgentGraph.from_dict(
            {
                "nodes": [{"id": "planner", "reliability": 1.5}],
                "edges": [],
            }
        )
