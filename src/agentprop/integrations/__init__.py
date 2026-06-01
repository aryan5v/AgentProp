"""Integrations for importing external workflow traces and framework specs."""

from agentprop.integrations.agent_instructions import (
    CodingAgentTarget,
    render_coding_agent_instructions,
)
from agentprop.integrations.framework_adapters import (
    SUPPORTED_FRAMEWORKS,
    graph_from_autogen_dict,
    graph_from_crewai_dict,
    graph_from_framework_dict,
    graph_from_langgraph_dict,
    graph_from_llamaindex_dict,
    graph_from_openai_agents_dict,
    to_autogen_dict,
    to_crewai_dict,
    to_framework_dict,
    to_langgraph_dict,
    to_llamaindex_dict,
    to_openai_agents_dict,
)
from agentprop.integrations.trace_loader import (
    TraceLoadResult,
    graph_from_trace,
    graph_from_trace_dict,
)

__all__ = [
    "CodingAgentTarget",
    "SUPPORTED_FRAMEWORKS",
    "TraceLoadResult",
    "graph_from_autogen_dict",
    "graph_from_crewai_dict",
    "graph_from_framework_dict",
    "graph_from_langgraph_dict",
    "graph_from_llamaindex_dict",
    "graph_from_openai_agents_dict",
    "graph_from_trace",
    "graph_from_trace_dict",
    "render_coding_agent_instructions",
    "to_autogen_dict",
    "to_crewai_dict",
    "to_framework_dict",
    "to_langgraph_dict",
    "to_llamaindex_dict",
    "to_openai_agents_dict",
]
