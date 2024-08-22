import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped


class GConvLayer(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=key)
        self.norm = nn.RMSNorm(hidden_dim, use_bias=False)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        m = jax.vmap(self.linear)(x)
        m = jax.nn.relu(m)
        m = a @ m
        x = jax.vmap(self.norm)(x + m)
        return x


class GNN(eqx.Module):
    convs: GConvLayer

    def __init__(self, hidden_dim: int, n_layers: int, *, key: PRNGKeyArray):
        super().__init__()
        make_conv = lambda k: GConvLayer(hidden_dim, key=k)
        self.convs = eqx.filter_vmap(make_conv)(jr.split(key, n_layers))

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        dynamic, static = eqx.partition(self.convs, eqx.is_array)

        def scan_fn(x, layer):
            conv: GConvLayer = eqx.combine(layer, static)
            x = conv(x, a)
            return x, None

        x, _ = jax.lax.scan(scan_fn, x, dynamic)
        return x


class GNNClassifier(eqx.Module):
    embed: nn.Embedding
    gnn: GNN
    predict: nn.Linear

    def __init__(
        self, hidden_dim: int, n_embeddings: int, n_layers: int, *, key: PRNGKeyArray
    ):
        super().__init__()
        sk = iter(jr.split(key, 3))
        self.embed = nn.Embedding(n_embeddings, hidden_dim, key=next(sk))
        self.gnn = GNN(hidden_dim, n_layers, key=next(sk))
        self.predict = nn.Linear(hidden_dim, 1, use_bias=True, key=next(sk))

    def __call__(
        self, x: Int[Array, " n_nodes"], a: Int[jsparse.BCOO, "n_nodes n_nodes"]
    ) -> Float[Array, ""]:
        x = jax.vmap(self.embed)(x)
        x = self.gnn(x, a)
        x = jnp.mean(x, axis=0)
        x = self.predict(x)
        return x[0]
