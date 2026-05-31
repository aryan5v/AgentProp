"""Shared core enums."""

from enum import StrEnum


class NodeType(StrEnum):
    """Supported node categories in an agent workflow graph."""

    AGENT = "AGENT"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    DOCUMENT = "DOCUMENT"
    VERIFIER = "VERIFIER"
    PLANNER = "PLANNER"
    EXECUTOR = "EXECUTOR"
    REVIEWER = "REVIEWER"
    OUTPUT = "OUTPUT"
    CUSTOM = "CUSTOM"
