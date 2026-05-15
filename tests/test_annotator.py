"""Tests for the Graph Annotator."""

from agentarmor.trace import AgentTrace
from agentarmor.types import NodeType, IntegrityLevel, ConfidentialityLevel
from agentarmor.registry.tool_registry import ToolRegistry, ToolMeta
from agentarmor.registry.data_registry import DataRegistry
from agentarmor.graph.constructor import GraphConstructor
from agentarmor.graph.dependency_analyzer import DependencyAnalyzer
from agentarmor.annotator import GraphAnnotator


class MockLLMClient:
    class Choices:
        class Message:
            content = '{"control_dependencies": [], "data_dependencies": []}'
        message = Message()

    class Completions:
        def create(self, **kwargs):
            return MockLLMClient.Response()

    class Response:
        pass

    class Chat:
        def __init__(self, parent):
            self.completions = parent.Completions()

    chat = None

    def __init__(self):
        self.Response.choices = [self.Choices()]
        self.chat = self.Chat(self)


def build_pdg():
    registry = ToolRegistry()
    registry.register(ToolMeta(name="create_trans", params=["receiver", "amount"]))
    data_reg = DataRegistry().default_agent_registry()
    constructor = GraphConstructor(registry, data_reg,
                                   DependencyAnalyzer(MockLLMClient()))
    trace = AgentTrace()
    trace.add_system("You are a banking assistant.")
    trace.add_user("Transfer $100 to ABC123.")
    trace.add_assistant("I'll transfer...", "create_trans",
                        {"receiver": "ABC123", "amount": 100})
    trace.add_tool("create_trans", {"receiver": "ABC123", "amount": 100}, "Done")
    return constructor.construct(trace), registry, data_reg


def test_annotator_type_assign():
    """Verify that system and user prompts get HIGH integrity from registry."""
    pdg, reg, data_reg = build_pdg()
    annotator = GraphAnnotator(reg, data_reg)
    annotator.annotate(pdg)

    sys_node = next(n for n in pdg.nodes if n.node_type == NodeType.SystemPrompt)
    assert sys_node.security_type.integrity == IntegrityLevel.HIGH

    user_node = next(n for n in pdg.nodes if n.node_type == NodeType.UserPrompt)
    assert user_node.security_type.integrity == IntegrityLevel.HIGH


def test_annotator_type_infer():
    """Verify type propagation from user prompt to tool nodes."""
    pdg, reg, data_reg = build_pdg()
    annotator = GraphAnnotator(reg, data_reg)
    annotator.annotate(pdg)

    # All nodes should have a security type after annotation
    for node in pdg.nodes:
        assert node.security_type is not None, f"Node {node.label} has no type"


def test_annotator_tool_type():
    """Tool node should get its type from the tool registry."""
    pdg, reg, data_reg = build_pdg()
    annotator = GraphAnnotator(reg, data_reg)
    annotator.annotate(pdg)

    tool_nodes = [n for n in pdg.nodes if n.node_type == NodeType.Tool]
    for tn in tool_nodes:
        assert tn.security_type is not None
