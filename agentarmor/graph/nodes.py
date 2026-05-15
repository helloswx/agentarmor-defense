"""
Graph node definitions for the Control Flow Graph, Data Flow Graph,
and Program Dependence Graph.

Node types follow Table 3 in the paper.
"""

from dataclasses import dataclass, field
from typing import Any

from agentarmor.types import NodeType, SecurityType, RuleType


@dataclass
class GraphNode:
    """A node in any of the three graph representations."""
    node_type: NodeType
    label: str
    node_id: str
    content: str = ""
    security_type: SecurityType | None = None
    rule: tuple[RuleType, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id
