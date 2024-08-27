import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Scalar, jaxtyped


class GConvLayer(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
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


class HiddenLayer(eqx.Module):
    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=key)
        self.norm = nn.RMSNorm(hidden_dim, use_bias=False)

    def __call__(self, x: Float[Array, " hidden_dim"]) -> Float[Array, " hidden_dim"]:
        y = self.linear(x)
        y = jax.nn.relu(y)
        return self.norm(x + y)


class GNN(eqx.Module):
    convs: GConvLayer
    hiddens: HiddenLayer

    def __init__(self, hidden_dim: int, n_layers: int, *, key: PRNGKeyArray):
        keys = iter(jr.split(key, 2))
        make_conv = lambda k: GConvLayer(hidden_dim, key=k)
        make_hidden = lambda k: HiddenLayer(hidden_dim, key=k)
        self.convs = eqx.filter_vmap(make_conv)(jr.split(next(keys), n_layers))
        self.hiddens = eqx.filter_vmap(make_hidden)(jr.split(next(keys), n_layers))

    def __call__(
        self,
        x: Float[Array, "n_nodes hidden_dim"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        layers = (self.convs, self.hiddens)
        dynamic, static = eqx.partition(layers, eqx.is_array)

        def scan_fn(x, layer):
            conv, hidden = eqx.combine(layer, static)
            x = conv(x, a)
            x = jax.vmap(hidden)(x)
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
        self.predict = nn.Linear(hidden_dim, "scalar", use_bias=True, key=next(sk))

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Int[Array, " n_nodes"],
        a: Int[jsparse.BCOO, "n_nodes n_nodes"],
        mask: Bool[Array, " n_nodes"],
    ) -> Scalar:
        x = jax.vmap(self.embed)(x)
        x = self.gnn(x, a)

        # The embedding of the graph is the average of the nodes embeddings.
        mask = jnp.expand_dims(mask, axis=1)
        x = jnp.where(mask, x, 0.0)
        x = jnp.sum(x, axis=0) / jnp.sum(mask)

        x = self.predict(x)
        return x
