"""
Runtime trace data models for LLM agent execution.

Models each message type in the agent's execution loop as described
in the paper: SystemMessage, UserMessage, LLMMessage, ToolMessage.
"""

from datetime import datetime
from typing import Any
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class SystemMessage:
    """Initial system-level prompt given to the agent."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: f"sys_{uuid4().hex[:8]}")

    @property
    def node_label(self) -> str:
        return "System"


@dataclass
class UserMessage:
    """User-issued command or query."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: f"usr_{uuid4().hex[:8]}")

    @property
    def node_label(self) -> str:
        return "User"


@dataclass
class LLMMessage:
    """LLM response containing thought process and optional tool call."""
    thought: str
    tool_name: str | None = None
    tool_params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: f"llm_{uuid4().hex[:8]}")

    @property
    def has_tool_call(self) -> bool:
        return self.tool_name is not None

    @property
    def node_label(self) -> str:
        return "LLM"


@dataclass
class ToolMessage:
    """Result of a tool invocation."""
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: f"tool_{uuid4().hex[:8]}")

    @property
    def node_label(self) -> str:
        return "Observation"


@dataclass
class AgentTrace:
    """
    A complete execution trace of an LLM agent.

    Sequence: SystemMessage -> (UserMessage -> LLMMessage -> ToolMessage -> LLMMessage -> ...)
    """
    events: list = field(default_factory=list)
    id: str = field(default_factory=lambda: f"trace_{uuid4().hex[:8]}")

    def add_system(self, content: str) -> "AgentTrace":
        self.events.append(SystemMessage(content=content))
        return self

    def add_user(self, content: str) -> "AgentTrace":
        self.events.append(UserMessage(content=content))
        return self

    def add_assistant(self, thought: str, tool_name: str | None = None,
                       tool_params: dict | None = None) -> "AgentTrace":
        self.events.append(LLMMessage(
            thought=thought,
            tool_name=tool_name,
            tool_params=tool_params or {}
        ))
        return self

    def add_tool(self, tool_name: str, params: dict | None = None,
                 observation: str = "") -> "AgentTrace":
        self.events.append(ToolMessage(
            tool_name=tool_name,
            params=params or {},
            observation=observation
        ))
        return self

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        return self.events[idx]

    def __iter__(self):
        return iter(self.events)
