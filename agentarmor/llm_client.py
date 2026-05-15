"""
LLM client wrapper for the laozhang.ai API (OpenAI-compatible).

Provides a thin wrapper around the OpenAI Python SDK configured
to use the laozhang.ai endpoint. Supports all models available
through the platform.

Usage:
    from agentarmor.llm_client import LLMClient
    client = LLMClient(api_key="sk-xxx")
    response = client.chat("gpt-4o-mini", [{"role": "user", "content": "Hello"}])
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResponse:
    """Normalized chat completion response."""
    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMClient:
    """
    OpenAI-compatible client pointed at laozhang.ai API.

    Parameters
    ----------
    api_key : str
        API key from laozhang.ai console.
    base_url : str
        API base URL. Defaults to https://api.laozhang.ai/v1.
    default_model : str
        Default model for dependency analysis. gpt-4o-mini is the
        recommended model per the paper's implementation.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.laozhang.ai/v1",
        default_model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(
        self,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Parameters
        ----------
        model : str
            Model name. Defaults to self.default_model (gpt-4o-mini).
        messages : list[dict]
            List of message dicts with 'role' and 'content'.
        temperature : float
            Sampling temperature. 0.0 for deterministic output.
        max_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        ChatResponse with .content, .model, .usage, .raw fields.
        """
        model = model or self.default_model
        messages = messages or []

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response,
        )
