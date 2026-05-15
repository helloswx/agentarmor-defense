"""Tests for the Graph Inspector."""

from agentarmor.types import NodeType, IntegrityLevel, ConfidentialityLevel, SecurityType
from agentarmor.graph.pdg import ProgramDependenceGraph
from agentarmor.graph.nodes import GraphNode
from agentarmor.inspector import GraphInspector
from agentarmor.policies import SecurityPolicy, DEFAULT_POLICIES


def test_inspector_allows_high_integrity():
    pdg = ProgramDependenceGraph()
    tn = GraphNode(
        node_type=NodeType.ToolName, label="read_file",
        node_id="n1", content="read_file",
        security_type=SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.MEDIUM),
    )
    pdg.add_node(tn)
    inspector = GraphInspector(DEFAULT_POLICIES)
    decision = inspector.inspect(pdg)
    assert decision.action == "allow"


def test_inspector_blocks_low_integrity_toolname():
    pdg = ProgramDependenceGraph()
    tn = GraphNode(
        node_type=NodeType.ToolName, label="transfer_money",
        node_id="n1", content="transfer_money",
        security_type=SecurityType(IntegrityLevel.LOW, ConfidentialityLevel.MEDIUM),
    )
    pdg.add_node(tn)
    inspector = GraphInspector(DEFAULT_POLICIES)
    decision = inspector.inspect(pdg)
    assert decision.action == "block"
    assert len(decision.violations) == 1
    assert "Int=LOW" in decision.violations[0].message


def test_inspector_blocks_low_integrity_param():
    pdg = ProgramDependenceGraph()
    tp = GraphNode(
        node_type=NodeType.ToolParam, label="receiver",
        node_id="n2", content="hacker_account",
        security_type=SecurityType(IntegrityLevel.LOW, ConfidentialityLevel.MEDIUM),
    )
    pdg.add_node(tp)
    inspector = GraphInspector(DEFAULT_POLICIES)
    decision = inspector.inspect(pdg)
    assert decision.action == "block"


def test_inspector_audit_mode():
    pdg = ProgramDependenceGraph()
    tn = GraphNode(
        node_type=NodeType.ToolName, label="dangerous_tool",
        node_id="n1", content="dangerous_tool",
        security_type=SecurityType(IntegrityLevel.LOW, ConfidentialityLevel.LOW),
    )
    pdg.add_node(tn)
    inspector = GraphInspector(DEFAULT_POLICIES, audit_mode=True)
    decision = inspector.inspect(pdg)
    assert decision.action == "allow"
    assert len(decision.violations) == 1


def test_inspector_allows_medium_integrity():
    pdg = ProgramDependenceGraph()
    tn = GraphNode(
        node_type=NodeType.ToolName, label="safe_tool",
        node_id="n1", content="safe_tool",
        security_type=SecurityType(IntegrityLevel.MEDIUM, ConfidentialityLevel.MEDIUM),
    )
    pdg.add_node(tn)
    inspector = GraphInspector(DEFAULT_POLICIES)
    decision = inspector.inspect(pdg)
    assert decision.action == "allow"
