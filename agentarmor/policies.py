"""
Security policy definitions for the Graph Inspector.

Policies are declarative rules that are checked against the annotated PDG.
Each policy defines a constraint on security-type combinations that,
if violated, results in the action being blocked.
"""

from dataclasses import dataclass, field

from agentarmor.types import NodeType, SecurityType, IntegrityLevel, ConfidentialityLevel, RuleType


@dataclass
class SecurityPolicy:
    """
    A security policy that constrains which security types are allowed.

    Example: "Forbid ToolName with Int < MEDIUM"
      → node_types = [ToolName, ToolParam]
      → rule = FORBID
      → condition = lambda st: st.integrity < MEDIUM
    """
    name: str
    description: str
    node_types: list[NodeType]
    rule: RuleType
    min_integrity: IntegrityLevel = IntegrityLevel.LOW
    min_confidentiality: ConfidentialityLevel = ConfidentialityLevel.LOW

    def check(self, node_type: NodeType, security_type: SecurityType) -> str | None:
        """
        Evaluate policy against a node. Returns violation message or None.
        """
        if node_type not in self.node_types:
            return None

        if self.rule == RuleType.FORBID:
            if not security_type.dominates_integrity(self.min_integrity):
                return (f"Forbid: {node_type.value} has Int={security_type.integrity.value} "
                        f"(requires >= {self.min_integrity.value})")

        elif self.rule == RuleType.REQUIRE:
            if not security_type.dominates_integrity(self.min_integrity):
                return (f"Require: {node_type.value} needs Int >= {self.min_integrity.value} "
                        f"(got {security_type.integrity.value})")

        elif self.rule == RuleType.RESTRICT:
            if not security_type.dominates_integrity(self.min_integrity):
                return (f"Restrict: {node_type.value} has Int={security_type.integrity.value} "
                        f"(restricted below {self.min_integrity.value})")

        return None


DEFAULT_POLICIES: list[SecurityPolicy] = [
    SecurityPolicy(
        name="block-low-integrity-tool-name",
        description="Block tool calls where the ToolName comes from untrusted sources",
        node_types=[NodeType.ToolName],
        rule=RuleType.FORBID,
        min_integrity=IntegrityLevel.MEDIUM,
    ),
    SecurityPolicy(
        name="block-low-integrity-tool-params",
        description="Block tool calls where parameters come from untrusted sources",
        node_types=[NodeType.ToolParam],
        rule=RuleType.FORBID,
        min_integrity=IntegrityLevel.MEDIUM,
    ),
]
