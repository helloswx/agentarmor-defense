"""
Data Registry — stores initial security types for data sources.

Pre-defines the security types for "root" nodes like system prompts,
user prompts, and known data sources before the agent starts running.
The graph annotator uses these as seeds for type propagation.
"""

from dataclasses import dataclass, field

from agentarmor.types import SecurityType, IntegrityLevel, ConfidentialityLevel


@dataclass
class DataMeta:
    """Security metadata for a data flow node."""
    name: str
    security_type: SecurityType = field(default_factory=lambda:
        SecurityType(IntegrityLevel.MEDIUM, ConfidentialityLevel.MEDIUM))
    description: str = ""


class DataRegistry:
    """
    Registry of known data sources and their default security types.

    Nodes whose type can be determined before execution starts
    (SystemPrompt, UserPrompt, known file/db handles) are stored here.
    """

    def __init__(self):
        self._entries: dict[str, DataMeta] = {}

    def register(self, meta: DataMeta) -> None:
        self._entries[meta.name] = meta

    def get(self, name: str) -> DataMeta | None:
        return self._entries.get(name)

    def default_agent_registry(self) -> "DataRegistry":
        """Build a registry with sensible defaults for an LLM agent."""
        self.register(DataMeta(
            name="system_prompt",
            security_type=SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.MEDIUM),
            description="System-level instruction"
        ))
        self.register(DataMeta(
            name="user_prompt",
            security_type=SecurityType(IntegrityLevel.HIGH, ConfidentialityLevel.MEDIUM),
            description="Direct user input — trusted integrity source"
        ))
        self.register(DataMeta(
            name="external_data",
            security_type=SecurityType(IntegrityLevel.LOW, ConfidentialityLevel.MEDIUM),
            description="Data from external tools / web — untrusted"
        ))
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._entries
