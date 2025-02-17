import networkx as nx

from builtins import frozenset as fset
from typing import Generic, Iterable, Collection, Mapping, TypeAlias, TypeVar

from toolz.dicttoolz import dissoc, valmap
from toolz.itertoolz import groupby, mapcat


N = TypeVar('N')


class Graph(Generic[N]):
    def __init__(self, edges: Iterable[tuple[N, N]]):
        self.graph = nx.DiGraph(edges)
        self.dom_helper = DomHelper(self.graph)

    def get_nodes(self) -> Collection[N]:
        return self.graph.nodes

    def get_edges(self) -> Collection[tuple[N, N]]:
        return self.graph.edges

    def get_preds(self, n: N) -> Iterable[N]:
        return list(self.graph.predecessors(n))

    def get_succs(self, n: N) -> Iterable[N]:
        return list(self.graph.successors(n))

    def get_dom_helper(self):
        return self.dom_helper


def get_start_node(g: nx.DiGraph):
    assert isinstance(g, nx.DiGraph)
    start, = [n for n in g.nodes if g.in_degree[n] == 0]
    return start


def _immediate_dominators(g: nx.DiGraph):
    assert isinstance(g, nx.DiGraph)
    start = get_start_node(g)
    return dissoc(nx.immediate_dominators(g, start), start)


def _dominance_frontier(g: nx.DiGraph):
    assert isinstance(g, nx.DiGraph)
    start = get_start_node(g)
    return nx.dominance_frontiers(g, start)


class DomHelper:
    def __init__(self, graph: nx.DiGraph):
        self.start = get_start_node(graph)
        self.imm_dom = _immediate_dominators(graph)
        self.dom_frontier = _dominance_frontier(graph)

    def dom_tree_preorder(self):
        dom_tree = nx.DiGraph(self.imm_dom.items()).reverse()
        return nx.traversal.dfs_preorder_nodes(dom_tree, self.start)

    def dominates(self, b1, b2):
        if b1 == b2:
            return True
        elif b2 == self.start:
            return False
        else:
            return self.dominates(b1, self.imm_dom[b2])
