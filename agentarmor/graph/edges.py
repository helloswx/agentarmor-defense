"""
Graph edge definitions.
"""

from dataclasses import dataclass
from enum import Enum

from agentarmor.types import EdgeType


class DependencyPattern(str, Enum):
    """The 8 reasoning patterns from paper Table 1."""
    DirectUserRequest = "DirectUserRequest"
    IndirectExecution = "IndirectExecution"
    ParameterizedExecution = "ParameterizedExecution"
    FunctionalExecution = "FunctionalExecution"
    ConditionalExecution = "ConditionalExecution"
    TransferExecution = "TransferExecution"
    MultipleSourceExecution = "MultipleSourceExecution"
    UnauthorizedIndirectExecution = "UnauthorizedIndirectExecution"


@dataclass
class GraphEdge:
    """An edge in the graph."""
    source: str  # source node id
    target: str  # target node id
    edge_type: EdgeType
    pattern: DependencyPattern | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))

    def __eq__(self, other) -> bool:
        if not isinstance(other, GraphEdge):
            return False
        return (self.source == other.source and
                self.target == other.target and
                self.edge_type == other.edge_type)
