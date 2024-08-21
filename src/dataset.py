from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from jaxtyping import PRNGKeyArray, jaxtyped


class Dataset:
    graphs: list[nx.DiGraph]

    def __init__(self, graphs: list[nx.DiGraph]):
        self.graphs = graphs

    @classmethod
    def from_files(
        cls,
        adj_file: Path,
        graph_id_file: Path,
        graph_labels_file: Path,
        node_labels_file: Path,
    ) -> "Dataset":
        graph = nx.read_adjlist(adj_file, create_using=nx.DiGraph, delimiter=",")

        with open(node_labels_file, "r") as file:
            node_labels = [int(line.strip()) for line in file.readlines()]
        node_labels = {node_id: label for node_id, label in enumerate(node_labels)}
        nx.set_node_attributes(graph, node_labels, "label")

        # Split the graph into the multiple subgraphs.
        with open(graph_id_file, "r") as file:
            graph_ids = [int(line.strip()) for line in file.readlines()]

        nodes_ids = [
            [
                node_id
                for node_id, graph_id in zip(range(len(graph)), graph_ids)
                if graph_id == current_id
            ]
            for current_id in set(graph_ids)
        ]
        graphs = [graph.subgraph(nodes) for nodes in nodes_ids]
        return cls(graphs)
