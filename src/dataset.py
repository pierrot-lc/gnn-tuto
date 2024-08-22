from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from jaxtyping import PRNGKeyArray, jaxtyped, Array, Float, Int


class Graph(eqx.Module):
    adjacency: Int[Array, "n_nodes n_nodes"]
    edges: Int[Array, "n_nodes 2"]

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> "Graph":
        n_nodes = len(graph)

        edges = nx.edges(graph)
        edges = jnp.array(edges, dtype=jnp.int32)

        data = jnp.ones(len(edges), dtype=jnp.int32)
        adjacency = jsparse.BCOO((data, edges), shape=(n_nodes, n_nodes))
        return Graph(adjacency, edges)


class Dataset:
    graphs: list[nx.Graph]

    def __init__(self, graphs: list[Graph]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Graph:
        return self.graphs[index]

    @classmethod
    def from_files(
        cls,
        adj_file: Path,
        graph_id_file: Path,
        graph_labels_file: Path,
        node_labels_file: Path,
    ) -> "Dataset":
        graph = nx.read_adjlist(adj_file, create_using=nx.DiGraph, delimiter=",")

        # Relabel all nodes from 0 to N-1
        graph = nx.relabel_nodes(
            graph,
            {old_id: new_id for new_id, old_id in enumerate(graph.nodes)},
        )

        # Read the labels.
        with open(node_labels_file, "r") as file:
            node_labels = [int(line.strip()) for line in file.readlines()]
        node_labels = {node_id: label for node_id, label in enumerate(node_labels)}
        nx.set_node_attributes(graph, node_labels, "label")

        # Split the graph into the multiple subgraphs.
        with open(graph_id_file, "r") as file:
            graph_ids = [int(line.strip()) for line in file.readlines()]

        assert list(sorted(graph_ids)) == graph_ids

        graphs = []
        current_node_id = 1
        for graph_id in sorted(set(graph_ids)):
            node_ids = []
            while (
                current_node_id < len(graph_ids)
                and graph_ids[current_node_id] == graph_id
            ):
                node_ids.append(current_node_id)
                current_node_id += 1

            subgraph = nx.subgraph(graph, node_ids).copy()
            subgraph = nx.relabel_nodes(
                subgraph,
                {old_id: new_id for new_id, old_id in enumerate(subgraph.nodes)},
            )
            graphs.append(subgraph)

        # Remove some weird graphs.
        graphs = [g for g in graphs if g.number_of_nodes() > 1]
        graphs = [g for g in graphs if g.number_of_edges() > 0]

        # To Graph.
        graphs = [Graph.from_networkx(g) for g in graphs]
        return cls(graphs)
