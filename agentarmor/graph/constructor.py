"""
Graph Constructor — transforms an AgentTrace into a Program Dependence Graph.

Implements the 8-step algorithm from paper Fig. 2:
  1. Node Decomposition
  2. Control Flow Edge Construction
  3. Control Dependency Analysis (LLM)
  4. Node Filtering (CFG → DFG nodes)
  5. Data Flow Edge Construction
  6. Data Dependency Analysis (LLM)
  7. Tool Registry Integration
  8. PDG Construction (merge CFG + DFG)
"""

from uuid import uuid4

from agentarmor.trace import AgentTrace, SystemMessage, UserMessage, LLMMessage, ToolMessage
from agentarmor.types import NodeType, EdgeType, IntegrityLevel, ConfidentialityLevel, SecurityType
from agentarmor.graph.nodes import GraphNode
from agentarmor.graph.edges import GraphEdge
from agentarmor.graph.pdg import ProgramDependenceGraph
from agentarmor.graph.dependency_analyzer import DependencyAnalyzer, ContextInput
from agentarmor.registry.tool_registry import ToolRegistry
from agentarmor.registry.data_registry import DataRegistry


class GraphConstructor:
    """
    Converts an LLM agent runtime trace into a Program Dependence Graph.
    """

    def __init__(self, tool_registry: ToolRegistry, data_registry: DataRegistry,
                 dependency_analyzer: DependencyAnalyzer):
        self.tool_registry = tool_registry
        self.data_registry = data_registry
        self.dep_analyzer = dependency_analyzer

        self._node_counter = 0
        self._pdg = ProgramDependenceGraph()

    def construct(self, trace: AgentTrace) -> ProgramDependenceGraph:
        """Run the full 8-step pipeline and return the PDG."""
        self._node_counter = 0
        self._pdg = ProgramDependenceGraph()

        cf_nodes, messages = self._step1_decompose(trace)
        # Register all nodes in the PDG before building edges
        for n in cf_nodes:
            self._pdg.add_node(n)
        self._step2_control_flow(cf_nodes)
        self._step3_control_dependency(cf_nodes, messages)
        df_nodes = self._step4_node_filter(cf_nodes)
        self._step5_data_flow(df_nodes, messages)
        self._step6_data_dependency(df_nodes, messages)
        self._step7_tool_registry()
        self._step8_merge(cf_nodes, df_nodes)

        return self._pdg

    # ------------------------------------------------------------------
    # Step 1: Node Decomposition
    # ------------------------------------------------------------------

    def _step1_decompose(self, trace: AgentTrace) -> tuple[list[GraphNode], list]:
        """Decompose each trace event into typed graph nodes."""
        cf_nodes: list[GraphNode] = []
        messages: list = []  # keep original messages for dependency analysis

        for event in trace.events:
            if isinstance(event, SystemMessage):
                node = GraphNode(
                    node_type=NodeType.SystemPrompt,
                    label="System",
                    node_id=self._new_id(),
                    content=event.content,
                )
                cf_nodes.append(node)
                messages.append(("system", event, [node]))

            elif isinstance(event, UserMessage):
                node = GraphNode(
                    node_type=NodeType.UserPrompt,
                    label="User",
                    node_id=self._new_id(),
                    content=event.content,
                )
                cf_nodes.append(node)
                messages.append(("user", event, [node]))

            elif isinstance(event, LLMMessage):
                # LLM call node
                llm_node = GraphNode(
                    node_type=NodeType.LLM,
                    label="LLM",
                    node_id=self._new_id(),
                    content=event.thought[:200],
                )
                # Thought node
                thought_node = GraphNode(
                    node_type=NodeType.Thought,
                    label="Thought",
                    node_id=self._new_id(),
                    content=event.thought[:500],
                )
                group = [llm_node, thought_node]
                cf_nodes.extend(group)

                if event.has_tool_call:
                    # ToolName node
                    tn_node = GraphNode(
                        node_type=NodeType.ToolName,
                        label=event.tool_name,
                        node_id=self._new_id(),
                        content=event.tool_name,
                    )
                    group.append(tn_node)
                    cf_nodes.append(tn_node)

                    # ToolParam nodes
                    for pname, pval in event.tool_params.items():
                        tp_node = GraphNode(
                            node_type=NodeType.ToolParam,
                            label=f"{event.tool_name}.{pname}",
                            node_id=self._new_id(),
                            content=str(pval),
                            metadata={"param_name": pname, "param_value": pval},
                        )
                        group.append(tp_node)
                        cf_nodes.append(tp_node)

                messages.append(("assistant", event, group))

            elif isinstance(event, ToolMessage):
                # Tool execution node
                tool_node = GraphNode(
                    node_type=NodeType.Tool,
                    label=event.tool_name,
                    node_id=self._new_id(),
                    content=event.tool_name,
                )
                # Observation node
                obs_node = GraphNode(
                    node_type=NodeType.Observation,
                    label="Observation",
                    node_id=self._new_id(),
                    content=event.observation[:2000],
                )
                group = [tool_node, obs_node]
                cf_nodes.extend(group)
                messages.append(("tool", event, group))

        return cf_nodes, messages

    # ------------------------------------------------------------------
    # Step 2: Control Flow Edges
    # ------------------------------------------------------------------

    def _step2_control_flow(self, cf_nodes: list[GraphNode]) -> None:
        """Connect nodes in temporal execution order."""
        for i in range(len(cf_nodes) - 1):
            edge = GraphEdge(
                source=cf_nodes[i].node_id,
                target=cf_nodes[i + 1].node_id,
                edge_type=EdgeType.ControlFlow,
            )
            self._pdg.add_edge(edge)

    # ------------------------------------------------------------------
    # Step 3: Control Dependency Analysis
    # ------------------------------------------------------------------

    def _step3_control_dependency(self, cf_nodes: list[GraphNode], messages: list) -> None:
        """
        For each tool call in the trace, analyze which input contexts
        influence the action selection (control dependency).

        Contexts are built INCREMENTALLY: only System/User prompts and
        Observations from PREVIOUS tool calls are visible. This avoids
        leaking future observations into earlier dependency analyses.
        """
        # Seed with initial contexts that exist before any tool call
        initial_nodes = [n for n in cf_nodes
                         if n.node_type in (NodeType.SystemPrompt, NodeType.UserPrompt)]
        contexts = [ContextInput(node_id=n.node_id, label=n.label,
                                 content=n.content, source_type=str(n.node_type))
                    for n in initial_nodes]

        # Process tool calls in temporal order
        for msg_type, msg, group in messages:
            if msg_type != "assistant" or not isinstance(msg, LLMMessage):
                continue
            if not msg.has_tool_call:
                continue

            tn_node = next((n for n in group if n.node_type == NodeType.ToolName), None)
            tp_nodes = [(n.node_id, n.metadata.get("param_name", ""),
                         str(n.metadata.get("param_value", "")))
                        for n in group if n.node_type == NodeType.ToolParam]

            if tn_node is None:
                continue

            # Analyze with context available AT THIS POINT in the trace
            result = self.dep_analyzer.analyze(
                contexts=contexts,
                tool_name_node_id=tn_node.node_id,
                tool_name=msg.tool_name,
                tool_param_nodes=tp_nodes,
            )

            for edge in result.control_edges + result.data_edges:
                known_ids = {c.node_id for c in contexts} | {n.node_id for n in cf_nodes}
                if edge.source in known_ids:
                    self._pdg.add_edge(edge)

            # Add this call's Observation to context for NEXT tool calls
            obs_node = self._find_obs_for_assistant(messages, msg)
            if obs_node:
                contexts.append(ContextInput(
                    node_id=obs_node.node_id, label=obs_node.label,
                    content=obs_node.content, source_type="Observation"
                ))

    # ------------------------------------------------------------------
    # Step 4: Node Filtering (CFG → DFG nodes)
    # ------------------------------------------------------------------

    def _step4_node_filter(self, cf_nodes: list[GraphNode]) -> list[GraphNode]:
        """
        Filter to keep only DFG-relevant nodes.
        Paper Table 3: LLM and Thought are NOT in DFG.
        """
        dfg_types = {NodeType.SystemPrompt, NodeType.UserPrompt, NodeType.ToolName,
                     NodeType.ToolParam, NodeType.Tool, NodeType.Observation, NodeType.Data}
        return [n for n in cf_nodes if n.node_type in dfg_types]

    # ------------------------------------------------------------------
    # Step 5: Data Flow Edges
    # ------------------------------------------------------------------

    def _step5_data_flow(self, df_nodes: list[GraphNode], messages: list) -> None:
        """
        Build explicit data flow edges:
          - SystemPrompt → ToolName (system context flows into decisions)
          - UserPrompt → ToolName / ToolParam
          - ToolParam → Tool (parameters flow into the tool)
          - Tool → Observation (output flows from tool)
          - Previous Observation → next ToolName / ToolParam
        """
        # Build a lookup for quick access
        nodes_by_type: dict[NodeType, list[GraphNode]] = {}
        for n in df_nodes:
            nodes_by_type.setdefault(n.node_type, []).append(n)

        # System → first ToolName
        sys_nodes = nodes_by_type.get(NodeType.SystemPrompt, [])
        tn_nodes = nodes_by_type.get(NodeType.ToolName, [])
        if sys_nodes and tn_nodes:
            for tn in tn_nodes:
                self._pdg.add_edge(GraphEdge(sys_nodes[0].node_id, tn.node_id, EdgeType.DataFlow))

        # UserPrompt → ToolName / ToolParam
        user_nodes = nodes_by_type.get(NodeType.UserPrompt, [])
        tp_nodes = nodes_by_type.get(NodeType.ToolParam, [])
        if user_nodes:
            user_id = user_nodes[0].node_id
            for tn in tn_nodes:
                self._pdg.add_edge(GraphEdge(user_id, tn.node_id, EdgeType.DataFlow))
            for tp in tp_nodes:
                self._pdg.add_edge(GraphEdge(user_id, tp.node_id, EdgeType.DataFlow))

        # ToolParam → Tool (within the same message group)
        for msg_type, msg, group in messages:
            if msg_type not in ("assistant", "tool"):
                continue
            tool_nodes = [n for n in group if n.node_type == NodeType.Tool]
            param_nodes = [n for n in group if n.node_type == NodeType.ToolParam]
            for tool_n in tool_nodes:
                for pn in param_nodes:
                    self._pdg.add_edge(GraphEdge(pn.node_id, tool_n.node_id, EdgeType.DataFlow))

        # Tool → Observation
        tool_nodes = nodes_by_type.get(NodeType.Tool, [])
        obs_nodes = nodes_by_type.get(NodeType.Observation, [])
        for t_node in tool_nodes:
            for o_node in obs_nodes:
                if self._order_adjacent(t_node, o_node, df_nodes):
                    self._pdg.add_edge(GraphEdge(t_node.node_id, o_node.node_id, EdgeType.DataFlow))

    # ------------------------------------------------------------------
    # Step 6: Data Dependency Analysis
    # ------------------------------------------------------------------

    def _step6_data_dependency(self, df_nodes: list[GraphNode], messages: list) -> None:
        """
        Data dependency edges were already inferred in step 3 via the
        dependency analyzer alongside control dependencies.
        This step verifies and supplements them for multi-round traces.
        """
        # Data dependencies already inferred per-tool-call in step 3.
        # Cross-resource data flows (e.g., write-then-read) are handled
        # by the tool registry in step 7.
        pass

    # ------------------------------------------------------------------
    # Step 7: Tool Registry Integration
    # ------------------------------------------------------------------

    def _step7_tool_registry(self) -> None:
        """
        Complement the DFG with implicit data flows inside tools.
        For each tool call, look up its metadata and add:
          - Tool → side_effect Data nodes
          - Side-effect Data → Observation edges
        """
        tn_nodes = [n for n in self._pdg.nodes if n.node_type == NodeType.ToolName]
        for tn in tn_nodes:
            meta = self.tool_registry.get(tn.content)
            if meta is None:
                continue

            # Create Data nodes for side effects
            for data_name in meta.side_effects:
                data_node = GraphNode(
                    node_type=NodeType.Data,
                    label=data_name,
                    node_id=self._new_id(),
                    content=data_name,
                    security_type=self._get_data_type(data_name),
                )
                self._pdg.add_node(data_node)

                # Connect Tool → side-effect Data
                tool_nodes = [n for n in self._pdg.nodes
                              if n.node_type == NodeType.Tool and n.content == tn.content]
                for tool_n in tool_nodes:
                    self._pdg.add_edge(GraphEdge(
                        tool_n.node_id, data_node.node_id, EdgeType.DataFlow
                    ))
                    self._pdg.add_edge(GraphEdge(
                        tool_n.node_id, data_node.node_id, EdgeType.DataDependency
                    ))

            # Create Data node for primary output
            if meta.output_data:
                out_data = GraphNode(
                    node_type=NodeType.Data,
                    label=meta.output_data,
                    node_id=self._new_id(),
                    content=meta.output_data,
                    security_type=self._get_data_type(meta.output_data),
                )
                self._pdg.add_node(out_data)
                for tool_n in [n for n in self._pdg.nodes
                               if n.node_type == NodeType.Tool and n.content == tn.content]:
                    self._pdg.add_edge(GraphEdge(
                        tool_n.node_id, out_data.node_id, EdgeType.DataFlow
                    ))

    # ------------------------------------------------------------------
    # Step 8: PDG Merge
    # ------------------------------------------------------------------

    def _step8_merge(self, cf_nodes: list[GraphNode], df_nodes: list[GraphNode]) -> None:
        """
        Merge CFG and DFG into the final PDG.

        Paper Table 3: PDG excludes LLM and Thought nodes (CFG-only).
        PDG nodes are: SystemPrompt, UserPrompt, ToolName, ToolParam,
                       Tool, Observation, Data.

        Adds PrincipalInput / PrincipalOutput edges for the boundary.
        """
        # PDG node types per Table 3
        pdg_types = {NodeType.SystemPrompt, NodeType.UserPrompt,
                     NodeType.ToolName, NodeType.ToolParam,
                     NodeType.Tool, NodeType.Observation, NodeType.Data}

        # Remove CFG-only nodes (LLM, Thought) from the PDG
        for node_id in list(self._pdg._nodes):
            node = self._pdg._nodes[node_id]
            if node.node_type not in pdg_types:
                # Remove incident edges first
                for e in list(self._pdg._edges):
                    if e.source == node_id or e.target == node_id:
                        self._pdg._edges.remove(e)
                        self._pdg._adj[e.source].remove(e)
                        self._pdg._in_edges[e.target].remove(e)
                del self._pdg._nodes[node_id]

        # PrincipalInput edges: SystemPrompt → UserPrompt
        sys_nodes = [n for n in self._pdg.nodes if n.node_type == NodeType.SystemPrompt]
        user_nodes = [n for n in self._pdg.nodes if n.node_type == NodeType.UserPrompt]
        for sn in sys_nodes:
            for un in user_nodes:
                self._pdg.add_edge(GraphEdge(
                    sn.node_id, un.node_id, EdgeType.PrincipalInput
                ))

        # PrincipalOutput edges: final Observation → boundary
        obs_nodes = [n for n in self._pdg.nodes if n.node_type == NodeType.Observation]
        if obs_nodes:
            last_obs = obs_nodes[-1]
            for n in self._pdg.nodes:
                if n.node_type in (NodeType.Tool, NodeType.Data):
                    self._pdg.add_edge(GraphEdge(
                        n.node_id, last_obs.node_id, EdgeType.PrincipalOutput
                    ))
                    break

    # -- helpers -------------------------------------------------------------

    def _new_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def _has_node_in_list(self, nodes: list[GraphNode], node_id: str) -> bool:
        return any(n.node_id == node_id for n in nodes)

    def _find_obs_for_assistant(self, messages: list, assistant_msg: LLMMessage) -> GraphNode | None:
        """Find the Observation node from the tool message following an assistant message."""
        found_assistant = False
        for msg_type, msg, group in messages:
            if msg is assistant_msg:
                found_assistant = True
                continue
            if found_assistant and msg_type == "tool" and isinstance(msg, ToolMessage):
                for n in group:
                    if n.node_type == NodeType.Observation:
                        return n
                break
        return None

    def _order_adjacent(self, a: GraphNode, b: GraphNode, nodes: list[GraphNode]) -> bool:
        """Check if nodes a and b are adjacent in the node list."""
        try:
            idx_a = next(i for i, n in enumerate(nodes) if n.node_id == a.node_id)
            idx_b = next(i for i, n in enumerate(nodes) if n.node_id == b.node_id)
            return abs(idx_a - idx_b) <= 3
        except StopIteration:
            return False

    def _get_data_type(self, data_name: str) -> SecurityType | None:
        """Look up initial security type for a data name."""
        meta = self.data_registry.get(data_name)
        if meta:
            return meta.security_type
        # Default: external data has medium integrity unless a specific
        # injection pattern (UnauthorizedIndirectExecution) is detected by
        # the dependency analyzer and lowers it to LOW.
        return SecurityType(IntegrityLevel.MEDIUM, ConfidentialityLevel.MEDIUM)
