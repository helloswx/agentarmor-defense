"""
Graph Inspector — enforces security policies on the annotated PDG.

Three-phase process (paper §5.3, Fig. 5):
  1. Rule Extraction — extract policy rules that apply to each node.
  2. Constraint Evaluation — evaluate rules against node security types.
  3. Violation Resolution — produce a SecurityDecision (allow/block).
"""

from dataclasses import dataclass, field

from agentarmor.types import NodeType, SecurityType, IntegrityLevel, ConfidentialityLevel, EdgeType
from agentarmor.graph.pdg import ProgramDependenceGraph
from agentarmor.graph.nodes import GraphNode
from agentarmor.policies import SecurityPolicy, DEFAULT_POLICIES


@dataclass
class Violation:
    """A single policy violation found during inspection."""
    node_id: str
    node_type: NodeType
    node_label: str
    policy_name: str
    message: str
    security_type: SecurityType


@dataclass
class SecurityDecision:
    """Result of the graph inspection."""
    action: str  # "allow" or "block"
    violations: list[Violation] = field(default_factory=list)
    blocked_tool: str | None = None
    summary: str = ""


class GraphInspector:
    """
    Inspects the annotated PDG and enforces security policies.

    If any policy is violated, the inspector can either block the
    violating tool call (default) or report and allow (audit mode).
    """

    # Edge types that carry semantic dependency (used for flow checks)
    _DEPENDENCY_EDGES = {EdgeType.DataDependency, EdgeType.ControlDependency}

    def __init__(self, policies: list[SecurityPolicy] | None = None,
                 audit_mode: bool = False,
                 allow_transfer_execution: bool = True):
        self.policies = policies or DEFAULT_POLICIES
        self.audit_mode = audit_mode
        self.allow_transfer_execution = allow_transfer_execution

    def inspect(self, pdg: ProgramDependenceGraph) -> SecurityDecision:
        """
        Run the full inspection on the annotated PDG.

        Phase 1: Rule extraction + node constraint evaluation.
        Phase 2: Edge-level information-flow validation.
        Phase 3: Violation resolution.

        Returns a SecurityDecision with action='block' if any
        violation is found (non-audit mode).
        """
        violations: list[Violation] = []

        # Phase 1 & 2a: Rule extraction + node constraint evaluation
        for node in pdg.nodes:
            if node.security_type is None:
                continue

            # Per-node rule check (if node has its own rule)
            if node.rule is not None:
                rule_type, min_level = node.rule
                if rule_type == "Forbid":
                    if not node.security_type.dominates_integrity(min_level):
                        violations.append(Violation(
                            node_id=node.node_id,
                            node_type=node.node_type,
                            node_label=node.label,
                            policy_name=f"node-rule-{node.node_id}",
                            message=(f"Forbid: {node.node_type.value} has "
                                     f"Int={node.security_type.integrity.value} "
                                     f"(requires >= {min_level.value})"),
                            security_type=node.security_type,
                        ))
                elif rule_type == "Require":
                    if not node.security_type.dominates_integrity(min_level):
                        violations.append(Violation(
                            node_id=node.node_id,
                            node_type=node.node_type,
                            node_label=node.label,
                            policy_name=f"node-rule-{node.node_id}",
                            message=(f"Require: {node.node_type.value} needs "
                                     f"Int >= {min_level.value} "
                                     f"(got {node.security_type.integrity.value})"),
                            security_type=node.security_type,
                        ))

            # Global policy checks (fallback when node has no per-node rule)
            if node.rule is None:
                for policy in self.policies:
                    msg = policy.check(node.node_type, node.security_type)
                    if msg:
                        violations.append(Violation(
                            node_id=node.node_id,
                            node_type=node.node_type,
                            node_label=node.label,
                            policy_name=policy.name,
                            message=msg,
                            security_type=node.security_type,
                        ))

        # Phase 2b: Edge-level information-flow checks (§5.3 step 2)
        edge_violations = self._check_edge_flows(pdg)
        violations.extend(edge_violations)

        # Phase 3: Resolution
        if not violations:
            return SecurityDecision(action="allow", summary="All checks passed.")

        # Collect blocked tool names for reporting
        blocked_tools_violations = [v for v in violations if v.node_type == NodeType.ToolName]
        blocked_tool = blocked_tools_violations[0].node_label if blocked_tools_violations else None

        if self.audit_mode:
            return SecurityDecision(
                action="allow",
                violations=violations,
                blocked_tool=blocked_tool,
                summary=f"AUDIT: {len(violations)} violation(s) logged but not blocked.",
            )

        return SecurityDecision(
            action="block",
            violations=violations,
            blocked_tool=blocked_tool,
            summary=f"BLOCKED: {len(violations)} violation(s) found. "
                     f"Blocked tool: {blocked_tool}.",
        )

    def _check_edge_flows(self, pdg: ProgramDependenceGraph) -> list[Violation]:
        """
        Check information flow along all dependency edges (§5.3 step 2).

        For every DataDependency / ControlDependency edge source→target:
          - Confidentiality: information must not flow from higher to lower
            confidentiality (source.con <= target.con).
          - Integrity: must not be influenced by lower-integrity sources
            (target.int <= source.int).

        If allow_transfer_execution is True, edges sourced from UserPrompt
        that delegate to Observation are allowed even if they would otherwise
        violate integrity (the TransferExecution pattern).
        """
        violations: list[Violation] = []
        con_order = {ConfidentialityLevel.HIGH: 2,
                     ConfidentialityLevel.MEDIUM: 1,
                     ConfidentialityLevel.LOW: 0}
        int_order = {IntegrityLevel.HIGH: 2,
                     IntegrityLevel.MEDIUM: 1,
                     IntegrityLevel.LOW: 0}

        for edge in pdg.edges:
            if edge.edge_type not in self._DEPENDENCY_EDGES:
                continue

            src_node = pdg.get_node(edge.source)
            tgt_node = pdg.get_node(edge.target)
            if src_node is None or tgt_node is None:
                continue
            if src_node.security_type is None or tgt_node.security_type is None:
                continue

            src_st = src_node.security_type
            tgt_st = tgt_node.security_type

            # Confidentiality: source.con must not be higher than target.con
            if con_order[src_st.confidentiality] > con_order[tgt_st.confidentiality]:
                violations.append(Violation(
                    node_id=tgt_node.node_id,
                    node_type=tgt_node.node_type,
                    node_label=tgt_node.label,
                    policy_name="info-flow-confidentiality",
                    message=(f"Confidentiality downgrade: "
                             f"{src_node.label}({src_st.confidentiality.value}) → "
                             f"{tgt_node.label}({tgt_st.confidentiality.value})"),
                    security_type=tgt_st,
                ))

            # Integrity: target must not be influenced by lower-integrity source
            if int_order[tgt_st.integrity] > int_order[src_st.integrity]:
                # Skip TransferExecution pattern when allowed
                if self.allow_transfer_execution:
                    if src_node.node_type == NodeType.UserPrompt:
                        continue

                violations.append(Violation(
                    node_id=tgt_node.node_id,
                    node_type=tgt_node.node_type,
                    node_label=tgt_node.label,
                    policy_name="info-flow-integrity",
                    message=(f"Integrity upgrade: "
                             f"{src_node.label}(Int={src_st.integrity.value}) → "
                             f"{tgt_node.label}(Int={tgt_st.integrity.value})"),
                    security_type=tgt_st,
                ))

        return violations
