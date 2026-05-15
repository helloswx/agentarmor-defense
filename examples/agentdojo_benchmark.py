"""
AgentDojo Integration — plugs AGENTARMOR into the AgentDojo benchmark.

Creates a BasePipelineElement that intercepts tool calls, builds an
AgentTrace, runs the AGENTARMOR pipeline, and blocks malicious calls.

Usage:
    python examples/agentdojo_benchmark.py --suite banking --attack important_instructions
    python examples/agentdojo_benchmark.py --suite workspace --all-attacks
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

from agentdojo.agent_pipeline.agent_pipeline import (
    AgentPipeline, load_system_message,
)
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop, ToolsExecutor, tool_result_to_str,
)
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime, FunctionCall
from agentdojo.logging import Logger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

# Monkey-patch: laozhang.ai API rejects null content in messages.
# AgentDojo's _content_blocks_to_openai_content_blocks returns None when
# message["content"] is None (common for assistant messages with tool_calls).
# Replace with "" so the API accepts the request.
import agentdojo.agent_pipeline.llms.openai_llm as _oai_llm
_orig_content_blocks = _oai_llm._content_blocks_to_openai_content_blocks
def _patched_content_blocks(message):
    result = _orig_content_blocks(message)
    return "" if result is None else result
_oai_llm._content_blocks_to_openai_content_blocks = _patched_content_blocks

# AGENTARMOR imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentarmor.trace import AgentTrace
from agentarmor.types import NodeType, EdgeType, IntegrityLevel, ConfidentialityLevel, RuleType
from agentarmor.registry.tool_registry import ToolRegistry, ToolMeta
from agentarmor.registry.data_registry import DataRegistry
from agentarmor.graph.dependency_analyzer import DependencyAnalyzer
from agentarmor.graph.constructor import GraphConstructor
from agentarmor.annotator import GraphAnnotator
from agentarmor.inspector import GraphInspector
from agentarmor.policies import SecurityPolicy, DEFAULT_POLICIES
from agentarmor.llm_client import LLMClient


# =============================================================================
# AGENTARMOR Defense Pipeline Element
# =============================================================================

class AgentArmorDefense(BasePipelineElement):
    """
    Pipeline element that intercepts tool calls and runs AGENTARMOR
    before allowing execution.

    This element sits between the LLM and the ToolsExecutor in the
    pipeline. It builds an incremental AgentTrace from the message
    history, constructs the PDG, and blocks tool calls that violate
    security policies.

    Args:
        tool_registry: ToolRegistry with metadata for all agent tools.
        llm_client: LLMClient for the dependency analyzer.
        allow_transfer_execution: Whether to allow the TransferExecution pattern.
        audit_mode: If True, report violations but don't block.
    """

    name = "agentarmor"

    def __init__(
        self,
        tool_registry: ToolRegistry,
        data_registry: DataRegistry,
        llm_client: LLMClient,
        allow_transfer_execution: bool = True,
        audit_mode: bool = False,
    ):
        self.tool_registry = tool_registry
        self.data_registry = data_registry
        self.llm_client = llm_client
        self.allow_transfer_execution = allow_transfer_execution
        self.audit_mode = audit_mode
        self._trace: AgentTrace | None = None
        self._system_msg: str = ""
        self._last_query: str = ""  # Track task changes for trace reset
        self._blocked_count = 0
        self._total_count = 0
        self._recorded_tool_ids: set[str] = set()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: list[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, list[ChatMessage], dict]:
        """Intercept tool calls and run AGENTARMOR inspection."""
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args

        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if last_msg.get("tool_calls") is None or len(last_msg["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        logger = Logger().get()

        # Build AgentTrace from message history on first call, or reset
        # when a new task starts. We detect a new task by checking if the
        # message history has been reset to just system+user (no tool calls).
        # Using query text alone doesn't work because utility and security
        # tests often share the same user prompt.
        is_new_task = (
            self._trace is None
            or self._is_fresh_conversation(messages)
        )
        if is_new_task:
            self._trace = AgentTrace()
            self._recorded_tool_ids = set()
            self._system_msg = self._build_system_from_messages(messages)
            if self._system_msg:
                self._trace.add_system(self._system_msg)
            self._trace.add_user(query)

        # Sync tool results from previous rounds into _trace so the
        # dependency analyzer can see observations (including injected content).
        self._sync_tool_results_from_messages(messages)

        # For each tool call in this assistant message, check with AGENTARMOR
        for tool_call in last_msg["tool_calls"]:
            tool_name = tool_call.function
            tool_args = dict(tool_call.args) if tool_call.args else {}
            thought = self._extract_thought(last_msg)

            # Build partial trace for this tool call
            check_trace = AgentTrace()
            for event in self._trace.events:
                event_type = type(event).__name__
                if event_type == 'SystemMessage':
                    check_trace.add_system(event.content)
                elif event_type == 'UserMessage':
                    check_trace.add_user(event.content)
                elif event_type == 'LLMMessage':
                    check_trace.add_assistant(
                        event.thought, event.tool_name, event.tool_params)
                elif event_type == 'ToolMessage':
                    check_trace.add_tool(
                        event.tool_name, event.params, event.observation)

            # Add current thought + tool call
            check_trace.add_assistant(thought, tool_name, tool_args)

            # Run AGENTARMOR
            try:
                dep_analyzer = DependencyAnalyzer(self.llm_client)
                constructor = GraphConstructor(
                    self.tool_registry, self.data_registry, dep_analyzer)
                annotator = GraphAnnotator(self.tool_registry, self.data_registry)
                inspector = GraphInspector(
                    DEFAULT_POLICIES,
                    audit_mode=self.audit_mode,
                    allow_transfer_execution=self.allow_transfer_execution,
                )

                pdg = constructor.construct(check_trace)
                annotator.annotate(pdg)
                decision = inspector.inspect(pdg)
                self._total_count += 1

                # Only block if the violation is on the CURRENT tool call.
                # The PDG includes all previous tool calls, and re-analysis
                # might flag old nodes as unauthorized retroactively. We only
                # care about the tool call we're checking right now.
                current_violations = [
                    v for v in decision.violations
                    if v.node_label == tool_name
                ]
                if not current_violations:
                    # Violations are on old/previous tool calls, not this one.
                    # Allow the current call to proceed.
                    self._trace.add_assistant(thought, tool_name, tool_args)
                    continue

                if decision.action == "block":
                    self._blocked_count += 1
                    tool_name_str = tool_name or "unknown"
                    # Build a focused error message for the current call
                    error_msg = (
                        f"AGENTARMOR BLOCKED tool call '{tool_name_str}': "
                        + "; ".join(v.message for v in current_violations)
                    )
                    logger.log(f"[AGENTARMOR] {error_msg}")

                    # Remove the blocked tool call from the assistant message.
                    # This achieves two things:
                    # 1. AgentDojo's security metric (functions_stack_trace_from_messages)
                    #    won't count the blocked call as a successful attack.
                    # 2. ToolsExecutor won't try to execute it.
                    # We do NOT add an error result, because that would require
                    # the tool_call to still be present (API validation).
                    last_msg["tool_calls"] = [
                        tc for tc in last_msg["tool_calls"]
                        if tc is not tool_call
                    ]

                    # Do NOT add blocked call to _trace -- it would carry the
                    # LOW integrity forward to future turns.

                    # For any remaining sibling calls in this message, record
                    # them in _trace so they execute normally.
                    for tc in last_msg["tool_calls"]:
                        self._trace.add_assistant(thought, tc.function,
                                                  dict(tc.args) if tc.args else {})

                    return query, runtime, env, messages, extra_args
                else:
                    # Allowed — record in trace for future calls
                    self._trace.add_assistant(thought, tool_name, tool_args)

            except Exception as e:
                logger.log(f"[AGENTARMOR] Error during inspection: {e}")
                # On error, allow the tool call
                self._trace.add_assistant(thought, tool_name, tool_args)

        return query, runtime, env, messages, extra_args

    def _extract_thought(self, msg: ChatMessage) -> str:
        """Extract thought text from an assistant message."""
        content = msg.get("content")
        if content is None:
            return "(no thought)"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("content", "") for block in content
                if block.get("type") == "text"
            )
        return str(content)

    def _sync_tool_results_from_messages(self, messages: list[ChatMessage]) -> None:
        """Record tool results from message history that are not yet in _trace."""
        if self._trace is None:
            return
        for msg in messages:
            if msg["role"] != "tool":
                continue
            # Skip blocked/error tool results — these are not real observations
            if msg.get("error"):
                continue
            tc = msg.get("tool_call")
            if tc is None:
                continue
            tid = msg.get("tool_call_id")
            if tid and tid in self._recorded_tool_ids:
                continue
            obs = self._extract_content_str(msg)
            self._trace.add_tool(tc.function, dict(tc.args) if tc.args else {}, obs)
            if tid:
                self._recorded_tool_ids.add(tid)

    def _extract_content_str(self, msg: ChatMessage) -> str:
        """Extract string content from any message type."""
        content = msg.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("content", "") for block in content
                if block.get("type") == "text"
            )
        return str(content)

    def _is_fresh_conversation(self, messages: list[ChatMessage]) -> bool:
        """Return True if messages represent the start of a new task.

        At the start of a new task, the pipeline.query() is called with
        empty messages. By the time our defense sees them, there are at
        most system + user + assistant messages (no tool results yet).
        A continuing conversation always has prior tool messages.
        """
        tool_msg_count = sum(1 for m in messages if m["role"] == "tool")
        return tool_msg_count == 0

    def _build_system_from_messages(self, messages: list[ChatMessage]) -> str:
        """Extract system message text from message history."""
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        block.get("content", "") for block in content
                        if block.get("type") == "text"
                    )
                return str(content)
        return ""

    @property
    def stats(self) -> dict:
        return {"blocked": self._blocked_count, "total": self._total_count}


# =============================================================================
# Auto-build ToolRegistry from AgentDojo's FunctionRuntime
# =============================================================================

def build_tool_registry_from_runtime(runtime: FunctionsRuntime) -> ToolRegistry:
    """
    Create an AGENTARMOR ToolRegistry from AgentDojo's FunctionsRuntime.

    Analyzes each tool's parameters and assigns default security policies.
    High-risk tools (send, transfer, delete, execute, write) get
    MEDIUM-required integrity by default.
    """
    registry = ToolRegistry()

    HIGH_RISK_PATTERNS = [
        "send", "transfer", "delete", "remove", "execute", "run",
        "create_trans", "create_user", "write", "modify", "update",
        "share", "post", "publish", "grant", "revoke", "approve",
    ]

    for func_name, func in runtime.functions.items():
        params = []
        param_names = []
        if func.parameters:
            try:
                param_names = list(func.parameters.model_fields.keys())
            except Exception:
                param_names = []

        side_effects = [f"{func_name}_result"]
        output_data = f"{func_name}_output"

        is_high_risk = any(p in func_name.lower() for p in HIGH_RISK_PATTERNS)

        meta = ToolMeta(
            name=func_name,
            params=param_names,
            side_effects=side_effects,
            output_data=output_data,
            toolname_policy=(
                (RuleType.FORBID, IntegrityLevel.MEDIUM)
                if is_high_risk else None
            ),
            param_policies={
                p: (RuleType.FORBID, IntegrityLevel.MEDIUM)
                for p in param_names
            } if is_high_risk else {},
        )
        registry.register(meta)

    return registry


# =============================================================================
# Custom Pipeline Builder with AGENTARMOR
# =============================================================================

def build_agentarmor_pipeline(
    model: str = "gpt-4o-mini",
    system_message: str | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
    allow_transfer_execution: bool = True,
    audit_mode: bool = False,
) -> tuple[AgentPipeline, "AgentArmorDefense", ToolRegistry]:
    """
    Build an AgentDojo pipeline with AGENTARMOR defense.

    Uses laozhang.ai (OpenAI-compatible) API so we don't depend on
    AgentDojo's model/provider registry.

    The pipeline chain is:
        SystemMessage → InitQuery → LLM → AgentArmorDefense → ToolsExecutor

    The ToolsExecutionLoop wraps [AgentArmorDefense, ToolsExecutor, LLM]
    so that every tool-call round-trip passes through AGENTARMOR inspection.
    """
    import openai as openai_mod

    api_key = os.environ.get("LAOZHANG_API_KEY", "sk-VYJUD0uVKS2rcYwzD9198eC385674e039023E8777d79A14d")
    base_url = os.environ.get("LAOZHANG_BASE_URL", "https://api.laozhang.ai/v1")

    # Create OpenAI-compatible client pointed at laozhang.ai
    oai_client = openai_mod.OpenAI(api_key=api_key, base_url=base_url)
    llm = OpenAILLM(oai_client, model)
    llm_name = model

    if system_message is None:
        system_message = load_system_message(None)

    system_msg_component = SystemMessage(system_message)
    init_query_component = InitQuery()

    if tool_output_format == "json":
        tool_fmt = lambda r: tool_result_to_str(r, dump_fn=json.dumps)
    else:
        tool_fmt = tool_result_to_str

    tool_reg = ToolRegistry()
    data_reg = DataRegistry().default_agent_registry()
    llm_client = LLMClient(api_key=api_key, default_model=model)

    defense = AgentArmorDefense(
        tool_registry=tool_reg,
        data_registry=data_reg,
        llm_client=llm_client,
        allow_transfer_execution=allow_transfer_execution,
        audit_mode=audit_mode,
    )

    # Order: defense inspects → executor runs allowed calls → LLM produces next message
    tools_loop = ToolsExecutionLoop([defense, ToolsExecutor(tool_fmt), llm])

    pipeline = AgentPipeline([
        system_msg_component, init_query_component, llm, tools_loop,
    ])
    # AgentDojo attacks parse the pipeline name for model identification.
    # Ensure the pipeline name contains a known model key (e.g. "gpt-4o-mini-2024-07-18")
    # while the actual API call still uses the user-provided model name.
    pipeline.name = f"gpt-4o-mini-2024-07-18-agentarmor"
    return pipeline, defense, tool_reg


# =============================================================================
# Benchmark Runner
# =============================================================================

def _select_tasks(
    all_tasks: list[str],
    specified: tuple[str, ...],
    max_count: int | None,
) -> list[str] | None:
    """Resolve task selection: specific IDs > max count > all tasks."""
    if specified:
        return list(specified)
    if max_count is not None and max_count > 0:
        return list(all_tasks)[:max_count]
    return None  # None means "all tasks" for benchmark functions


def run_agentdojo_benchmark(
    suite_name: str = "banking",
    model: str = "gpt-4o-mini",
    attack: str | None = None,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    benchmark_version: str = "v1.2.2",
    allow_transfer_execution: bool = True,
    audit_mode: bool = False,
    max_user_tasks: int | None = None,
    max_injection_tasks: int | None = None,
):
    """
    Run AgentDojo benchmark with AGENTARMOR defense.

    Args:
        suite_name: One of 'banking', 'workspace', 'slack', 'travel'.
        model: LLM model name.
        attack: Attack name (None = utility only, no attacks).
        user_tasks: Specific user tasks to run (empty = all).
        injection_tasks: Specific injection tasks (empty = all).
        benchmark_version: AgentDojo benchmark version.
        allow_transfer_execution: AGENTARMOR config.
        audit_mode: Report but don't block.
        max_user_tasks: Limit number of user tasks.
        max_injection_tasks: Limit number of injection tasks.
    """
    load_dotenv(".env")

    print(f"\n{'='*60}")
    print(f"  AgentDojo + AGENTARMOR Benchmark")
    print(f"{'='*60}")
    print(f"  Suite:        {suite_name}")
    print(f"  Model:        {model}")
    print(f"  Attack:       {attack or 'None (utility test)'}")
    print(f"  Defense:      AGENTARMOR")
    print(f"  Allow TF:     {allow_transfer_execution}")
    print(f"  Audit Mode:   {audit_mode}")
    print(f"{'='*60}\n")

    suite = get_suite(benchmark_version, suite_name)

    # Build pipeline with proper tool registry for this suite
    pipeline, defense, tool_reg = build_agentarmor_pipeline(
        model=model,
        allow_transfer_execution=allow_transfer_execution,
        audit_mode=audit_mode,
    )

    # Build tool registry from the suite's tools
    from agentdojo.functions_runtime import FunctionsRuntime
    runtime = FunctionsRuntime(suite.tools)
    auto_reg = build_tool_registry_from_runtime(runtime)
    defense.tool_registry = auto_reg
    defense.data_registry = DataRegistry().default_agent_registry()
    defense._trace = None  # Reset trace for new benchmark
    defense._last_query = ""
    defense._recorded_tool_ids = set()
    defense._blocked_count = 0
    defense._total_count = 0

    logdir = Path(f"./runs/agentarmor_{suite_name}_{model.replace('.', '_')}")
    logdir.mkdir(parents=True, exist_ok=True)

    # Resolve which user / injection tasks to run
    run_user_tasks = _select_tasks(suite.user_tasks.keys(), user_tasks, max_user_tasks)
    run_injection_tasks = _select_tasks(suite.injection_tasks.keys(), injection_tasks, max_injection_tasks)

    from agentdojo.logging import OutputLogger
    with OutputLogger(str(logdir)):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline, suite,
                user_tasks=run_user_tasks,
                logdir=logdir, force_rerun=True,
                benchmark_version=benchmark_version,
            )
        else:
            attacker = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline, suite, attacker,
                user_tasks=run_user_tasks,
                injection_tasks=run_injection_tasks,
                logdir=logdir, force_rerun=True,
                benchmark_version=benchmark_version,
            )

    # Compute metrics
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results) if utility_results else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Utility (UAR_no_atk):          {avg_utility:.1%}")
    if attack:
        security_results = results.get("security_results", {}).values()
        avg_security = sum(security_results) / len(security_results) if security_results else 0
        asr = 1.0 - avg_security
        print(f"  Security (1-ASR):              {avg_security:.1%}")
        print(f"  Attack Success Rate (ASR):     {asr:.1%}")

    print(f"  AGENTARMOR Blocked/Total:      {defense._blocked_count}/{defense._total_count}")
    if defense._total_count > 0:
        print(f"  AGENTARMOR Block Rate:         {defense._blocked_count/defense._total_count:.1%}")
    print(f"{'='*60}\n")

    return results, defense


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AgentDojo + AGENTARMOR Benchmark")
    parser.add_argument("--suite", "-s", default="banking",
                        choices=["banking", "workspace", "slack", "travel"],
                        help="Task suite to benchmark")
    parser.add_argument("--model", "-m", default="gpt-4o-mini",
                        help="LLM model")
    parser.add_argument("--attack", "-a", default=None,
                        help="Attack to use (None = utility only)")
    parser.add_argument("--all-attacks", action="store_true",
                        help="Run all available attacks")
    parser.add_argument("--allow-transfer", action="store_true", default=True,
                        help="Allow TransferExecution pattern")
    parser.add_argument("--no-transfer", action="store_true",
                        help="Disallow TransferExecution pattern")
    parser.add_argument("--audit", action="store_true",
                        help="Audit mode (report, don't block)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of user tasks")
    parser.add_argument("--benchmark-version", default="v1.2.2",
                        help="AgentDojo benchmark version")
    parser.add_argument("--suite-only", action="store_true",
                        help="Show available suites and exit")

    args = parser.parse_args()

    if args.suite_only:
        suites = ["banking", "workspace", "slack", "travel"]
        print("Available suites:", suites)
        from agentdojo.attacks.attack_registry import ATTACKS
        print("Available attacks:", ATTACKS)
        sys.exit(0)

    allow_tf = not args.no_transfer

    results, defense = run_agentdojo_benchmark(
        suite_name=args.suite,
        model=args.model,
        attack=args.attack,
        allow_transfer_execution=allow_tf,
        audit_mode=args.audit,
        max_user_tasks=args.max_tasks,
        benchmark_version=args.benchmark_version,
    )
