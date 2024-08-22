from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from beartype import beartype
from jaxtyping import Array, Bool, Int, PRNGKeyArray, jaxtyped
from tqdm import tqdm


class GraphData(eqx.Module):
    adjacency: Int[
        jsparse.BCOO, "n_nodes n_nodes"
    ]  # a[i, j] = 1 <=> node j is link to node i.
    edges: Int[Array, "n_edges 2"]  # e[n] = (j, i) <=> node j is linked to node i.
    nodes: Int[Array, " n_nodes"]
    mask: Bool[Array, " n_nodes"]
    label: Int[Array, ""]

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph, label: int) -> "GraphData":
        n_nodes = len(graph)
        edges = jnp.array(nx.edges(graph), dtype=jnp.int32)
        adjacency = jsparse.BCOO(
            (
                jnp.ones(len(edges), dtype=jnp.int32),
                jnp.flip(
                    edges, axis=1
                ),  # Recall that a[i, j] = 1 <=> node j is linked to node i.
            ),
            shape=(n_nodes, n_nodes),
        )
        nodes = jnp.array(
            [graph.nodes[n]["atom_id"] for n in range(n_nodes)], dtype=jnp.int32
        )
        mask = jnp.ones(n_nodes, dtype=jnp.bool_)
        label = jnp.array(label, dtype=jnp.int32)
        return cls(adjacency, edges, nodes, mask, label)

    @classmethod
    def pad(cls, graph: "GraphData", max_nodes: int, max_edges: int) -> "GraphData":
        n_nodes, n_edges = len(graph.adjacency), len(graph.edges)
        edges = jnp.pad(
            graph.edges,
            ((0, max_edges - n_edges), (0, 0)),
            mode="constant",
            constant_values=max_nodes - 1,
        )
        adjacency = jsparse.BCOO(
            (jnp.ones(max_edges, dtype=jnp.int32), jnp.flip(edges, axis=1)),
            shape=(max_nodes, max_nodes),
        )
        nodes = jnp.pad(graph.nodes, (0, max_nodes - n_nodes))
        mask = jnp.pad(
            graph.mask, (0, max_nodes - n_nodes), mode="constant", constant_values=False
        )
        return cls(adjacency, edges, nodes, mask, graph.label)

    @classmethod
    def stack(cls, graphs: list["GraphData"]) -> "GraphData":
        return cls(
            adjacency=jsparse.sparsify(jnp.stack)([g.adjacency for g in graphs]),
            edges=jnp.stack([g.edges for g in graphs]),
            nodes=jnp.stack([g.nodes for g in graphs]),
            mask=jnp.stack([g.mask for g in graphs]),
            label=jnp.stack([g.nodes for g in graphs]),
        )


class Dataset:
    graphs: list[GraphData]

    def __init__(self, graphs: list[GraphData]):
        self.graphs = graphs

    @jaxtyped(typechecker=beartype)
    def iter(
        self, batch_size: int, total_iters: int, key: PRNGKeyArray
    ) -> Iterator[GraphData]:
        for sk in jr.split(key, total_iters):
            batch_ids = jr.choice(sk, len(self), (batch_size,))
            samples = [self[i] for i in batch_ids]
            yield GraphData.stack(samples)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> GraphData:
        return self.graphs[index]

    @property
    def n_atoms(self) -> int:
        return max(jnp.max(g.nodes) for g in self.graphs) + 1

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
        nx.set_node_attributes(graph, node_labels, "atom_id")

        # Split the graph into the multiple subgraphs.
        with open(graph_id_file, "r") as file:
            graph_ids = [int(line.strip()) for line in file.readlines()]

        assert list(sorted(graph_ids)) == graph_ids

        graphs = []
        current_node_id = 1
        for graph_id in tqdm(
            sorted(set(graph_ids)), desc="Splitting into subgraphs", leave=False
        ):
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

        # Graph labels.
        with open(graph_labels_file, "r") as file:
            graph_labels = [int(line.strip()) for line in file.readlines()]
        assert len(graphs) == len(graph_labels)

        # Remove some weird graphs.
        graphs = [g for g in graphs if g.number_of_nodes() > 1]
        graphs = [g for g in graphs if g.number_of_edges() > 0]

        # To Graph.
        graphs = [
            GraphData.from_networkx(g, l)
            for g, l in tqdm(
                zip(graphs, graph_labels),
                desc="To jax array",
                total=len(graphs),
                leave=False,
            )
        ]

        # Pad the graphs.
        max_nodes = max(len(g.adjacency) for g in graphs)
        max_edges = max(len(g.edges) for g in graphs)
        graphs = [
            GraphData.pad(g, max_nodes, max_edges)
            for g in tqdm(graphs, desc="Padding graphs", leave=False)
        ]

        return cls(graphs)
