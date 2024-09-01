from functools import partial

import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped


class GConvLayer(eqx.Module):
    """Classical graph convolutional layer.

    The aggregation function is the sum.
    """

    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
        _: any,
    ) -> Float[Array, "n_nodes hidden_dim"]:
        """Apply the standard graph conv layer.

        ---
        Parameters:
            x: Node embeddings.
            a: Adjacency matrix where a[i, j] = 1 => edge from j to i.
            _: Ignored (for compatibility with the GATLayer).

        ---
        Returns:
            The updated node embeddings.
        """
        m = jax.vmap(self.linear)(x)
        m = jax.nn.relu(m)

        m = a @ m

        x = jax.vmap(self.norm)(x + m)
        return x


class GATLayer(eqx.Module):
    """Graph attention layer from the Graph Attention Networks paper.

    This implementation is a single-head layer with an added norm and residual
    connection.

    Paper: https://arxiv.org/abs/1710.10903.
    """

    linear: nn.Linear
    attention: nn.Sequential
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        keys = iter(jr.split(key, 2))
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=next(keys))
        self.attention = nn.Sequential(
            [
                nn.Linear(
                    2 * hidden_dim, out_features=1, use_bias=False, key=next(keys)
                ),
                nn.Lambda(partial(jax.nn.leaky_relu, negative_slope=0.2)),
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        _: any,
        e: Int[Array, "n_edges 2"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        """Apply the GAT layer.

        ---
        Parameters:
            x: Node embeddings.
            _: Ignored (for compatibility with the GConvLayer).
            e: Edge relations as a list of (source, destination) node ids.

        ---
        Returns:
            The updated node embeddings.
        """
        # Information from neighbours.
        m = x[e[:, 0]]
        m = jax.vmap(self.linear)(m)

        # Information from destination node.
        s = x[e[:, 1]]
        s = jax.vmap(self.linear)(s)

        a = jnp.concat((m, s), axis=1)  # Shape of [n_edges, 2 * hidden_dim].
        a = jax.vmap(self.attention)(a)  # Shape of [n_edges, 1].

        # Apply the softmax combination in two steps:
        # 1. Unormalized combination of the features.
        # 2. Apply the normalization factor.
        a = jax.lax.exp(a)
        m = jax.ops.segment_sum(a * m, e[:, 1], len(x))
        n = jax.ops.segment_sum(a, e[:, 1], len(x))
        m = x / jnp.where(n == 0.0, 1.0, n)  # Avoid division by 0.

        m = jax.nn.relu(m)
        x = jax.vmap(self.norm)(m + x)
        return x


class HiddenLayer(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(self, x: Float[Array, " hidden_dim"]) -> Float[Array, " hidden_dim"]:
        y = self.linear(x)
        y = jax.nn.relu(y)
        return self.norm(x + y)


class GNN(eqx.Module):
    """Sequentially apply a list of graph layers and simple hidden layers."""

    convs: GConvLayer | GATLayer
    hiddens: HiddenLayer

    def __init__(
        self, hidden_dim: int, n_layers: int, conv_type: str, *, key: PRNGKeyArray
    ):
        keys = iter(jr.split(key, 2))
        make_hidden = lambda k: HiddenLayer(hidden_dim, key=k)
        match conv_type:
            case "sum":
                make_conv = lambda k: GConvLayer(hidden_dim, key=k)
            case "gat":
                make_conv = lambda k: GATLayer(hidden_dim, key=k)
            case _:
                raise ValueError(f"Unknown conv type: {conv_type}")

        self.convs = eqx.filter_vmap(make_conv)(jr.split(next(keys), n_layers))
        self.hiddens = eqx.filter_vmap(make_hidden)(jr.split(next(keys), n_layers))

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
        e: Int[Array, "n_edges 2"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        """Apply all GNN layers.

        ---
        Parameters:
            x: Node embeddings.
            a: Adjacency matrix where a[i, j] = 1 => edge from j to i.
                Used if conv_type == "sum".
            e: Edge relations as a list of (source, destination) node ids.
                Used if conv_type == "gat".

        ---
        Returns:
            The updated node embeddings.
        """
        layers = (self.convs, self.hiddens)
        dynamic, static = eqx.partition(layers, eqx.is_array)

        def scan_fn(x, layer):
            conv, hidden = eqx.combine(layer, static)
            x = conv(x, a, e)
            x = jax.vmap(hidden)(x)
            return x, None

        x, _ = jax.lax.scan(scan_fn, x, dynamic)
        return x


class RankingModel(eqx.Module):
    gnn: GNN
    predict: nn.Linear
    hidden_dim: int

    def __init__(
        self, hidden_dim: int, n_layers: int, conv_type: str, *, key: PRNGKeyArray
    ):
        super().__init__()
        sk = iter(jr.split(key, 2))
        self.gnn = GNN(hidden_dim, n_layers, conv_type, key=next(sk))
        self.predict = nn.Linear(hidden_dim, "scalar", key=next(sk))
        self.hidden_dim = hidden_dim

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
        e: Int[Array, "n_edges 2"],
    ) -> Float[Array, " n_nodes"]:
        """Predict the score of all nodes in the graph.

        ---
        Parameters:
            a: Adjacency matrix where a[i, j] = 1 => edge from j to i.
                Used if conv_type == "sum".
            e: Edge relations as a list of (source, destination) node ids.
                Used if conv_type == "gat".

        ---
        Returns:
            The node's scores.
        """
        x = jnp.zeros((len(a), self.hidden_dim), dtype=jnp.float32)
        x = self.gnn(x, a, e)
        x = jax.vmap(self.predict)(x)
        return x
