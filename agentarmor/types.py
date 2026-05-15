"""
Security type system for AgentArmor.

Defines the dual security lattice (Integrity + Confidentiality) used to
annotate nodes in the Program Dependence Graph.
"""

from enum import Enum
from dataclasses import dataclass, field


class IntegrityLevel(str, Enum):
    """Integrity dimension: trustworthiness of data source."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __repr__(self):
        return f"Int:{self.value[0]}"


class ConfidentialityLevel(str, Enum):
    """Confidentiality dimension: sensitivity of data."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __repr__(self):
        return f"Con:{self.value[0]}"


@dataclass(frozen=True)
class SecurityType:
    """
    A pair of (Integrity, Confidentiality) levels forming a node's security type.

    The security lattice join rules:
      - Integrity join: weakest (min) — LOW dominates HIGH
      - Confidentiality join: strongest (max) — HIGH dominates LOW
    """
    integrity: IntegrityLevel = IntegrityLevel.MEDIUM
    confidentiality: ConfidentialityLevel = ConfidentialityLevel.MEDIUM

    def join(self, other: "SecurityType") -> "SecurityType":
        """Merge two types using the security lattice join."""
        int_order = {IntegrityLevel.HIGH: 2, IntegrityLevel.MEDIUM: 1, IntegrityLevel.LOW: 0}
        con_order = {ConfidentialityLevel.HIGH: 2, ConfidentialityLevel.MEDIUM: 1, ConfidentialityLevel.LOW: 0}

        new_int = min(self.integrity, other.integrity, key=lambda x: int_order[x])
        new_con = max(self.confidentiality, other.confidentiality, key=lambda x: con_order[x])
        return SecurityType(new_int, new_con)

    def dominates_integrity(self, required: IntegrityLevel) -> bool:
        """Check if this type's integrity meets or exceeds required."""
        order = {IntegrityLevel.HIGH: 2, IntegrityLevel.MEDIUM: 1, IntegrityLevel.LOW: 0}
        return order[self.integrity] >= order[required]

    def __repr__(self):
        return f"{{{self.integrity!r}, {self.confidentiality!r}}}"


class NodeType(str, Enum):
    """Node types as defined in paper Table 3."""
    SystemPrompt = "SystemPrompt"
    UserPrompt = "UserPrompt"
    LLM = "LLM"
    Thought = "Thought"
    ToolName = "ToolName"
    ToolParam = "ToolParam"
    Tool = "Tool"
    Observation = "Observation"
    Data = "Data"


class EdgeType(str, Enum):
    """Edge types for the three graph layers."""
    ControlFlow = "ControlFlow"
    ControlDependency = "ControlDependency"
    DataFlow = "DataFlow"
    DataDependency = "DataDependency"
    PrincipalInput = "PrincipalInput"
    PrincipalOutput = "PrincipalOutput"


class RuleType(str, Enum):
    """Rule types for graph inspector enforcement."""
    FORBID = "Forbid"
    REQUIRE = "Require"
    RESTRICT = "Restrict"
