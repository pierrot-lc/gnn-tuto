from collections.abc import Iterator

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped
from tqdm import tqdm


class GraphData(eqx.Module):
    adjacency: Int[
        jsparse.BCOO, "n_nodes n_nodes"
    ]  # a[i, j] = 1 <=> node j is link to node i.
    edges: Int[Array, "n_edges 2"]  # e[n] = (j, i) <=> node j is linked to node i.
    scores: Float[Array, " n_nodes"]
    mask: Bool[Array, " n_nodes"]

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> "GraphData":
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
        scores = jnp.array([graph.nodes[i]["betweenness"] for i in range(len(graph))])
        mask = jnp.ones(n_nodes, dtype=jnp.bool_)
        return cls(adjacency, edges, scores, mask)

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
        scores = jnp.pad(graph.scores, (0, max_nodes - n_nodes))
        mask = jnp.pad(
            graph.mask, (0, max_nodes - n_nodes), mode="constant", constant_values=False
        )
        return cls(adjacency, edges, scores, mask)

    @classmethod
    def stack(cls, graphs: list["GraphData"]) -> "GraphData":
        return cls(
            adjacency=jsparse.sparsify(jnp.stack)([g.adjacency for g in graphs]),
            edges=jnp.stack([g.edges for g in graphs]),
            scores=jnp.stack([g.scores for g in graphs]),
            mask=jnp.stack([g.mask for g in graphs]),
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

    @classmethod
    def split(
        cls, dataset: "Dataset", split: float, *, key: PRNGKeyArray
    ) -> tuple["Dataset", "Dataset"]:
        assert 0 <= split <= 1

        training_size = int(len(dataset) * split)
        perm = jr.permutation(key, len(dataset))
        training_graphs = [dataset.graphs[i] for i in perm[:training_size]]
        val_graphs = [dataset.graphs[i] for i in perm[training_size:]]
        return cls(training_graphs), cls(val_graphs)

    @classmethod
    def generate(cls, n_nodes: int, n_graphs: int, key: PRNGKeyArray) -> "Dataset":
        graphs = []
        seeds = [int(jr.key_data(sk)[1]) for sk in jr.split(key, n_graphs)]
        for seed in tqdm(seeds, desc="Generating graphs", leave=False):
            graph = nx.erdos_renyi_graph(n_nodes, seed=seed, p=0.05, directed=True)

            # Keep the largest connected component.
            nodes = max(nx.weakly_connected_components(graph), key=len)
            graph = graph.subgraph(nodes).copy()
            graph = nx.relabel_nodes(
                graph, {old_id: new_id for new_id, old_id in enumerate(graph.nodes)}
            )  # Relabel nodes from 0 to N.

            # Score the nodes.
            scores = nx.betweenness_centrality(
                graph, k=None, normalized=False, weight=None
            )
            nx.set_node_attributes(graph, scores, "betweenness")

            graphs.append(graph)

        # To Graph.
        graphs = [
            GraphData.from_networkx(graph)
            for graph in tqdm(
                graphs,
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
