"""
AgentArmor — top-level defense interface.

Assembles the full 4-stage pipeline:
  AgentTrace → Graph Constructor → Graph Annotator → Graph Inspector → SecurityDecision
"""

from dataclasses import dataclass
from typing import Any

from agentarmor.trace import AgentTrace
from agentarmor.graph.constructor import GraphConstructor
from agentarmor.graph.dependency_analyzer import DependencyAnalyzer
from agentarmor.graph.pdg import ProgramDependenceGraph
from agentarmor.annotator import GraphAnnotator
from agentarmor.inspector import GraphInspector, SecurityDecision
from agentarmor.policies import SecurityPolicy, DEFAULT_POLICIES
from agentarmor.registry.tool_registry import ToolRegistry, ToolMeta
from agentarmor.registry.data_registry import DataRegistry


@dataclass
class AgentArmorConfig:
    """Configuration for the AgentArmor defense system."""
    model: str = "gpt-4o-mini"
    policies: list[SecurityPolicy] | None = None
    audit_mode: bool = False
    allow_transfer_execution: bool = True


class AgentArmor:
    """
    Main defense system that secures an LLM agent by analyzing its
    execution trace through the PDG-based pipeline.

    Usage::

        armor = AgentArmor(tool_registry, data_registry, llm_client)

        # Build trace as the agent runs
        trace = AgentTrace()
        trace.add_system("You are a banking assistant...")
        trace.add_user("Transfer $100 to account ABC123")
        trace.add_assistant("I will transfer...", "transfer_money",
                            {"account": "ABC123", "amount": 100})
        trace.add_tool("transfer_money", {...}, "Success")

        # Inspect
        decision = armor.inspect(trace)
        if decision.action == "block":
            print(f"Blocked: {decision.summary}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        data_registry: DataRegistry,
        llm_client: Any = None,
        config: AgentArmorConfig | None = None,
    ):
        self.tool_registry = tool_registry
        self.data_registry = data_registry
        self.config = config or AgentArmorConfig()

        if llm_client is None:
            try:
                from agentarmor.llm_client import LLMClient
                import os
                api_key = os.environ.get("LAOZHANG_API_KEY", "")
                llm_client = LLMClient(api_key=api_key, default_model=self.config.model)
            except Exception:
                llm_client = None

        self._dep_analyzer = DependencyAnalyzer(llm_client, self.config.model)
        self._constructor = GraphConstructor(
            self.tool_registry, self.data_registry, self._dep_analyzer
        )
        self._annotator = GraphAnnotator(self.tool_registry, self.data_registry)
        self._inspector = GraphInspector(
            policies=self.config.policies or DEFAULT_POLICIES,
            audit_mode=self.config.audit_mode,
            allow_transfer_execution=self.config.allow_transfer_execution,
        )

    def inspect(self, trace: AgentTrace) -> SecurityDecision:
        """
        Run the full pipeline on an agent trace.

        Returns SecurityDecision with action='allow' or 'block'.
        """
        pdg = self._constructor.construct(trace)
        self._annotator.annotate(pdg)
        decision = self._inspector.inspect(pdg)
        return decision

    def inspect_tool_call(
        self,
        user_message: str,
        thought: str,
        tool_name: str,
        tool_params: dict,
        system_message: str = "",
        prior_observations: list[str] | None = None,
    ) -> SecurityDecision:
        """
        Convenience method: inspect a single tool call with minimal setup.

        This builds an on-the-fly trace and runs the pipeline.
        Useful for inline checks before executing a tool.
        """
        trace = AgentTrace()
        if system_message:
            trace.add_system(system_message)
        trace.add_user(user_message)

        # Add prior observations as context
        for obs in (prior_observations or []):
            trace.add_tool("_context", {}, obs)

        trace.add_assistant(thought, tool_name, tool_params)
        return self.inspect(trace)
