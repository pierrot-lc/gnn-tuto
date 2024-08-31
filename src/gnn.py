import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped


class GConvLayerAdj(eqx.Module):
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
        m = jax.vmap(self.linear)(x)
        m = jax.nn.relu(m)

        m = a @ m

        x = jax.vmap(self.norm)(x + m)
        return x


class GConvLayerEdges(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)
        self.norm = nn.RMSNorm(hidden_dim)

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        _: any,
        e: Int[Array, "n_edges 2"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        m = x[e[:, 0]]
        m = jax.vmap(self.linear)(m)
        m = jax.nn.relu(m)

        d = jax.ops.segment_sum(
            jnp.ones_like(m), segment_ids=e[:, 1], num_segments=len(x)
        )
        m = jax.ops.segment_max(m, segment_ids=e[:, 1], num_segments=len(x))
        m = jnp.where(d == 0, 0.0, m)

        x = jax.vmap(self.norm)(x + m)
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
    convs: GConvLayerAdj | GConvLayerEdges
    hiddens: HiddenLayer

    def __init__(
        self, hidden_dim: int, n_layers: int, conv_type: str, *, key: PRNGKeyArray
    ):
        keys = iter(jr.split(key, 2))
        make_hidden = lambda k: HiddenLayer(hidden_dim, key=k)
        match conv_type:
            case "edges":
                make_conv = lambda k: GConvLayerEdges(hidden_dim, key=k)
            case "adj":
                make_conv = lambda k: GConvLayerAdj(hidden_dim, key=k)
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
        x = jnp.zeros((len(a), self.hidden_dim), dtype=jnp.float32)
        x = self.gnn(x, a, e)
        x = jax.vmap(self.predict)(x)
        return x
