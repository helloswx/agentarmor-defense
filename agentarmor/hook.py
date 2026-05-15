"""
Runtime hook that intercepts LLM agent execution to build traces.

Provides a context manager / decorator that wraps an agent's execution
loop and captures the trace (SystemMessage, UserMessage, LLMMessage,
ToolMessage) for consumption by the AgentArmor pipeline.
"""

import functools
from collections.abc import Callable
from typing import Any

from agentarmor.trace import AgentTrace


class AgentHook:
    """
    A hook that wraps an LLM agent's tool-calling loop.

    Usage (context manager)::

        trace = AgentTrace()
        hook = AgentHook(trace)
        with hook:
            agent.run(user_input)
        # trace is now populated

    Usage (manual)::

        hook = AgentHook(trace)
        hook.on_user_message("transfer $100 to ABC")
        hook.on_assistant_message("I will transfer...", "transfer_money",
                                   {"account": "ABC", "amount": 100})
        hook.on_tool_result("transfer_money", {"account": "ABC", "amount": 100},
                            "Transfer successful")
    """

    def __init__(self, trace: AgentTrace | None = None):
        self.trace = trace or AgentTrace()

    def on_system_message(self, content: str) -> None:
        self.trace.add_system(content)

    def on_user_message(self, content: str) -> None:
        self.trace.add_user(content)

    def on_assistant_message(self, thought: str, tool_name: str | None = None,
                             tool_params: dict[str, Any] | None = None) -> None:
        self.trace.add_assistant(thought, tool_name, tool_params)

    def on_tool_result(self, tool_name: str, params: dict[str, Any] | None = None,
                       observation: str = "") -> None:
        self.trace.add_tool(tool_name, params, observation)

    def __enter__(self) -> "AgentHook":
        return self

    def __exit__(self, *args) -> None:
        pass


def trace_tool_call(hook: AgentHook):
    """
    Decorator that wraps a tool function to capture its inputs/outputs.

    Usage::

        @trace_tool_call(hook)
        def transfer_money(account: str, amount: float):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Capture the function name and args as tool params
            params = {}
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            params.update(dict(zip(arg_names, args)))
            params.update(kwargs)
            hook.on_tool_result(func.__name__, params, str(result))
            return result
        return wrapper
    return decorator
