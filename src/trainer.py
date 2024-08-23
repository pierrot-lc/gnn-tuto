from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray
from wandb.wandb_run import Run

from .dataset import Dataset, GraphData
from .gnn import GNNClassifier


def key_source(key: PRNGKeyArray) -> Iterator[PRNGKeyArray]:
    while True:
        key, sk = jr.split(key)
        yield sk


def count_params(model: eqx.Module) -> int:
    """Count the number of parameters of the given equinox module."""
    # Replace the params of the PE module by None to filter them out.
    model = jax.tree_util.tree_map_with_path(
        lambda p, v: None if "positional_encoding" in jax.tree_util.keystr(p) else v,
        model,
    )
    # jax.tree_util.tree_map_with_path(lambda p, _: print(p), model)

    params = eqx.filter(model, eqx.is_array)

    n_params = jax.tree.map(lambda p: jnp.prod(jnp.array(p.shape)), params)
    n_params = jnp.array(jax.tree.leaves(n_params))
    n_params = jnp.sum(n_params)
    return int(n_params)


def train(
    model: GNNClassifier,
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    logger: Run,
    *,
    key: PRNGKeyArray,
):
    n_params = count_params(model)
    logger.summary["n_params"] = n_params

    for batch in train_dataset.iter(batch_size, train_iter, key):
        pass
