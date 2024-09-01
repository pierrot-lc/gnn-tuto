import jax.experimental.sparse as jsparse
import networkx as nx
import pytest
from src.dataset import GraphData


def random_graph(n_nodes: int, directed: bool) -> nx.DiGraph:
    graph = nx.gaussian_random_partition_graph(
        n_nodes, 10, 10, 0.25, 0.1, directed=directed
    )
    if not nx.is_directed(graph):
        graph = nx.to_directed(graph)

    nx.set_node_attributes(graph, 1, "scores")
    return graph


@pytest.mark.parametrize(
    "graph",
    [
        GraphData.from_networkx(random_graph(100, True)),
        GraphData.from_networkx(random_graph(100, False)),
    ],
)
def test_adjacency_orientation(graph: GraphData):
    adjacency = jsparse.todense(graph.adjacency)
    for e1, e2 in graph.edges:
        assert adjacency[e2, e1] == 1
