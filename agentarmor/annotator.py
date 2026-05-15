"""
Graph Annotator — assigns security semantics to PDG nodes.

Two-phase process (paper §5.2):
  Phase 1 — Type Assign: Look up predefined types from the Property Registry.
  Phase 2 — Type Infer: Propagate types through the graph using
              single-source propagation and multi-source lattice joins.
"""

from agentarmor.types import NodeType, SecurityType, IntegrityLevel, ConfidentialityLevel
from agentarmor.graph.pdg import ProgramDependenceGraph
from agentarmor.graph.nodes import GraphNode
from agentarmor.registry.tool_registry import ToolRegistry
from agentarmor.registry.data_registry import DataRegistry


class GraphAnnotator:
    """
    Annotates each node in the PDG with a SecurityType by combining
    registry lookups and structural type inference.
    """

    def __init__(self, tool_registry: ToolRegistry, data_registry: DataRegistry):
        self.tool_registry = tool_registry
        self.data_registry = data_registry

    def annotate(self, pdg: ProgramDependenceGraph) -> ProgramDependenceGraph:
        """
        Run both phases and return the annotated PDG.

        Returns the same PDG instance with nodes' security_type set.
        """
        self._type_assign(pdg)
        self._type_infer(pdg)
        return pdg

    # ------------------------------------------------------------------
    # Phase 1: Type Assign
    # ------------------------------------------------------------------

    def _type_assign(self, pdg: ProgramDependenceGraph) -> None:
        """
        Assign initial security types AND rule types from the registries.

        Paper §5.2 defines Type := {security type, rule type} per node.
        Phase 1 populates both from the property registry.

        - SystemPrompt / UserPrompt → DataRegistry (security type only, no rule)
        - Tool node → ToolRegistry (tool_type)
        - ToolName node → ToolRegistry (toolname_policy)
        - ToolParam node → ToolRegistry (param_policies)
        - Known Data nodes → DataRegistry
        """
        for node in pdg.nodes:
            if node.node_type == NodeType.SystemPrompt:
                meta = self.data_registry.get("system_prompt")
                if meta:
                    node.security_type = meta.security_type

            elif node.node_type == NodeType.UserPrompt:
                meta = self.data_registry.get("user_prompt")
                if meta:
                    node.security_type = meta.security_type

            elif node.node_type == NodeType.Tool:
                meta = self.tool_registry.get(node.content)
                if meta:
                    node.security_type = meta.tool_type

            elif node.node_type == NodeType.ToolName:
                meta = self.tool_registry.get(node.content)
                if meta is not None and meta.toolname_policy is not None:
                    node.rule = meta.toolname_policy

            elif node.node_type == NodeType.ToolParam:
                param_name = node.metadata.get("param_name", "")
                if param_name:
                    meta = self.tool_registry.get(node.label.split(".")[0]
                                                   if "." in node.label else "")
                    if meta and param_name in meta.param_policies:
                        node.rule = meta.param_policies[param_name]

            elif node.node_type == NodeType.Data:
                meta = self.data_registry.get(node.content)
                if meta:
                    node.security_type = meta.security_type
                else:
                    # Unknown data defaults to untrusted
                    if node.security_type is None:
                        node.security_type = SecurityType(
                            IntegrityLevel.LOW, ConfidentialityLevel.MEDIUM
                        )

    # ------------------------------------------------------------------
    # Phase 2: Type Infer
    # ------------------------------------------------------------------

    def _type_infer(self, pdg: ProgramDependenceGraph) -> None:
        """
        Propagate security types along DEPENDENCY edges only.

        Types are propagated only through DataDependency and ControlDependency
        edges — not through ControlFlow or DataFlow, which represent
        structural/temporal relationships rather than semantic causation.

        Algorithm:
          1. Assign default types to all unseeded nodes.
          2. Fixed-point iteration: propagate types along dependency edges.
             - Single-source → inherit; Multi-source → lattice join.
        """
        from agentarmor.types import EdgeType

        DEPENDENCY_EDGES = {EdgeType.DataDependency, EdgeType.ControlDependency}

        # Step 1: Assign default types as seeds for propagation
        for node in pdg.nodes:
            if node.security_type is None:
                if node.node_type == NodeType.Observation:
                    node.security_type = SecurityType(
                        IntegrityLevel.MEDIUM, ConfidentialityLevel.MEDIUM
                    )
                else:
                    node.security_type = SecurityType(
                        IntegrityLevel.MEDIUM, ConfidentialityLevel.MEDIUM
                    )

        # Step 2: Fixed-point propagation
        changed = True
        while changed:
            changed = False
            for node in pdg.nodes:
                if self._is_trusted_seed(node):
                    continue

                in_edges = pdg.in_edges(node.node_id)
                dep_predecessors = [e.source for e in in_edges
                                    if e.edge_type in DEPENDENCY_EDGES]
                if not dep_predecessors:
                    continue

                pred_types = []
                has_unauthorized_source = False
                for pred_id in dep_predecessors:
                    pred_node = pdg.get_node(pred_id)
                    if pred_node and pred_node.security_type is not None:
                        pred_types.append(pred_node.security_type)
                    # Check if any dependency edge from this predecessor has an
                    # UnauthorizedIndirectExecution pattern — if so, the
                    # predecessor's content is untrusted injection.
                    # IMPORTANT: UserPrompt and SystemPrompt are ALWAYS
                    # authorized sources. UnauthorizedIndirectExecution can
                    # only originate from Observation or Data nodes.
                    if pred_node and pred_node.node_type in (
                        NodeType.UserPrompt, NodeType.SystemPrompt
                    ):
                        continue
                    for e in in_edges:
                        if e.source == pred_id and e.edge_type in DEPENDENCY_EDGES:
                            if e.pattern and e.pattern.value == "UnauthorizedIndirectExecution":
                                has_unauthorized_source = True
                                break

                if not pred_types:
                    continue

                if len(pred_types) == 1:
                    new_type = pred_types[0]
                else:
                    new_type = pred_types[0]
                    for t in pred_types[1:]:
                        new_type = new_type.join(t)

                # If an UnauthorizedIndirectExecution edge exists, lower
                # integrity to LOW regardless of the predecessor's type.
                if has_unauthorized_source:
                    new_type = SecurityType(IntegrityLevel.LOW, new_type.confidentiality)

                if node.security_type != new_type:
                    node.security_type = new_type
                    changed = True

    def _is_trusted_seed(self, node: GraphNode) -> bool:
        """Check if this node's type comes from a trusted source (registry)."""
        if node.node_type in (NodeType.SystemPrompt, NodeType.UserPrompt):
            return True
        if node.node_type == NodeType.Tool:
            return self.tool_registry.get(node.content) is not None
        if node.node_type == NodeType.Data:
            return self.data_registry.get(node.content) is not None
        return False
