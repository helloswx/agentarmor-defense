"""
Tool Registry — stores metadata about each available tool.

For each tool, the registry records:
  - Its function signature (parameter names and types)
  - Internal data flow: which inputs map to which outputs / side-effect data nodes
  - Security policy annotations (e.g., a tool's ToolName node requires Int >= MEDIUM)
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from agentarmor.types import SecurityType, IntegrityLevel, ConfidentialityLevel, RuleType


@dataclass
class ToolMeta:
    """
    Security metadata for a single tool.

    Parameters
    ----------
    name : str
        Tool function name (e.g. "transfer_money", "read_file").
    params : list[str]
        Ordered list of parameter names.
    side_effects : list[str]
        Names of **data** nodes produced as side effects by this tool
        (e.g. ``["email_data"]`` for a ``search_email`` tool).
    output_data : str | None
        Data node name for the primary return value, if any.
    tool_type : SecurityType
        The intrinsic security type of the tool implementation (trusted code).
    toolname_policy : tuple[RuleType, IntegrityLevel] | None
        Policy constraint for the ToolName node (e.g. Forbid when Int < MEDIUM).
    """
    name: str
    params: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    output_data: str | None = None
    tool_type: SecurityType = field(default_factory=lambda:
        SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.MEDIUM))
    toolname_policy: tuple[RuleType, IntegrityLevel] | None = None
    param_policies: dict[str, tuple[RuleType, IntegrityLevel]] = field(default_factory=dict)


class ToolRegistry:
    """
    Registry of all tools available to the agent.

    Used by the graph constructor (step 7) to complement the DFG with
    implicit data flows inside each tool, and by the annotator to assign
    initial security types to tool-related nodes.
    """

    def __init__(self):
        self._tools: dict[str, ToolMeta] = {}

    def register(self, meta: ToolMeta) -> None:
        self._tools[meta.name] = meta

    def get(self, name: str) -> ToolMeta | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
