"""
Program Dependence Graph — the central data structure.

Combines Control Flow Graph (CFG) and Data Flow Graph (DFG)
into a unified PDG that captures both control and data dependencies.
"""

from collections import defaultdict

from agentarmor.graph.nodes import GraphNode
from agentarmor.graph.edges import GraphEdge
from agentarmor.types import EdgeType


class ProgramDependenceGraph:
    """
    The PDG represents the agent's execution trace as a graph where:
      - Nodes are typed execution events (ToolName, ToolParam, Observation, etc.)
      - Edges are control-flow, control-dependency, data-flow, and data-dependency

    It is the output of the Graph Constructor and the input to the
    Graph Annotator / Inspector.
    """

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adj: dict[str, list[GraphEdge]] = defaultdict(list)
        self._in_edges: dict[str, list[GraphEdge]] = defaultdict(list)

    # -- nodes ---------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    # -- edges ---------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)
        self._adj[edge.source].append(edge)
        self._in_edges[edge.target].append(edge)

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    def out_edges(self, node_id: str) -> list[GraphEdge]:
        return list(self._adj.get(node_id, []))

    def in_edges(self, node_id: str) -> list[GraphEdge]:
        return list(self._in_edges.get(node_id, []))

    def predecessors(self, node_id: str) -> list[str]:
        return [e.source for e in self._in_edges.get(node_id, [])]

    def successors(self, node_id: str) -> list[str]:
        return [e.target for e in self._adj.get(node_id, [])]

    # -- filtered queries ----------------------------------------------------

    def control_dependency_edges(self) -> list[GraphEdge]:
        return [e for e in self._edges if e.edge_type == EdgeType.ControlDependency]

    def data_dependency_edges(self) -> list[GraphEdge]:
        return [e for e in self._edges if e.edge_type == EdgeType.DataDependency]

    def data_flow_edges(self) -> list[GraphEdge]:
        return [e for e in self._edges if e.edge_type == EdgeType.DataFlow]

    # -- stats ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)
