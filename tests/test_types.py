"""Tests for the security type system."""

from agentarmor.types import (
    SecurityType, IntegrityLevel, ConfidentialityLevel,
    NodeType, EdgeType, RuleType
)


def test_security_type_join_integrity():
    """Integrity join should pick the weakest (LOW dominates HIGH)."""
    high = SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.HIGH)
    low = SecurityType(IntegrityLevel.LOW, ConfidentialityLevel.LOW)
    merged = high.join(low)
    assert merged.integrity == IntegrityLevel.LOW
    assert merged.confidentiality == ConfidentialityLevel.HIGH


def test_security_type_join_medium():
    """Test mid-range join."""
    high = SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.MEDIUM)
    med = SecurityType(IntegrityLevel.MEDIUM, ConfidentialityLevel.LOW)
    merged = high.join(med)
    assert merged.integrity == IntegrityLevel.MEDIUM
    assert merged.confidentiality == ConfidentialityLevel.MEDIUM


def test_dominates_integrity():
    """Test the dominates_integrity check."""
    st = SecurityType(IntegrityLevel.MEDIUM, ConfidentialityLevel.HIGH)
    assert st.dominates_integrity(IntegrityLevel.MEDIUM)
    assert st.dominates_integrity(IntegrityLevel.LOW)
    assert not st.dominates_integrity(IntegrityLevel.HIGH)


def test_node_types():
    """Verify all node types from Table 3 are defined."""
    expected = {"SystemPrompt", "UserPrompt", "LLM", "Thought",
                "ToolName", "ToolParam", "Tool", "Observation", "Data"}
    actual = set(n.value for n in NodeType)
    assert expected == actual


def test_edge_types():
    expected = {"ControlFlow", "ControlDependency", "DataFlow",
                "DataDependency", "PrincipalInput", "PrincipalOutput"}
    actual = set(e.value for e in EdgeType)
    assert expected == actual
