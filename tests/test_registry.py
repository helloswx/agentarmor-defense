"""Tests for the property registries."""

from agentarmor.types import SecurityType, IntegrityLevel, ConfidentialityLevel
from agentarmor.registry.tool_registry import ToolRegistry, ToolMeta
from agentarmor.registry.data_registry import DataRegistry, DataMeta


def test_tool_registry_register_and_get():
    reg = ToolRegistry()
    meta = ToolMeta(
        name="transfer_money",
        params=["account", "amount"],
        side_effects=["transaction_log"],
        output_data="transfer_result",
    )
    reg.register(meta)
    assert "transfer_money" in reg
    retrieved = reg.get("transfer_money")
    assert retrieved.name == "transfer_money"
    assert retrieved.params == ["account", "amount"]
    assert retrieved.side_effects == ["transaction_log"]


def test_tool_registry_missing():
    reg = ToolRegistry()
    assert reg.get("nonexistent") is None


def test_data_registry_defaults():
    reg = DataRegistry()
    reg.default_agent_registry()
    sys_meta = reg.get("system_prompt")
    assert sys_meta is not None
    assert sys_meta.security_type.integrity == IntegrityLevel.HIGH
    ext = reg.get("external_data")
    assert ext is not None
    assert ext.security_type.integrity == IntegrityLevel.LOW
