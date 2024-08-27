from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from beartype import beartype
from jaxtyping import Array, PRNGKeyArray, Scalar, jaxtyped
from tqdm import tqdm
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


@jaxtyped(typechecker=beartype)
def batch_loss(model: GNNClassifier, batch: GraphData) -> Scalar:
    logits = jax.vmap(model)(batch.nodes, batch.adjacency, batch.mask)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, batch.label)
    return jnp.mean(loss)


@eqx.filter_jit
def batch_metrics(model: GNNClassifier, batch: GraphData) -> dict[str, Array]:
    logits = jax.vmap(model)(batch.nodes, batch.adjacency, batch.mask)
    y_pred = logits > 0
    y_true = batch.label == 1

    metrics = dict()
    metrics["loss"] = optax.losses.sigmoid_binary_cross_entropy(logits, batch.label)
    metrics["accuracy"] = y_pred == y_true
    return metrics


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def batch_update(
    model: GNNClassifier,
    batch: GraphData,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> GNNClassifier:
    grads = eqx.filter_grad(batch_loss)(model, batch)
    params, static = eqx.partition(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    model = eqx.combine(params, static)
    return model


def eval(
    model: GNNClassifier,
    dataset: Dataset,
    batch_size: int,
    eval_iter: int,
    *,
    key: PRNGKeyArray,
) -> dict[str, float]:
    model = eqx.tree_inference(model, value=True)
    metrics = [
        batch_metrics(model, batch)
        for batch in tqdm(
            dataset.iter(batch_size, eval_iter, key),
            desc="Eval",
            total=eval_iter,
            leave=False,
        )
    ]
    metrics = jax.tree.map(
        lambda *xs: jnp.concat(xs), *metrics
    )  # Concat each array of metrics.
    metrics = jax.tree.map(jnp.mean, metrics)
    metrics = jax.tree.map(float, metrics)
    return metrics


def train(
    model: GNNClassifier,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    train_iter: int,
    eval_iter: int,
    eval_freq: int,
    logger: Run,
    *,
    key: PRNGKeyArray,
):
    n_params = count_params(model)
    logger.summary["n_params"] = n_params
    keys = key_source(key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for batch_id, batch in tqdm(
        enumerate(train_dataset.iter(batch_size, train_iter, next(keys))),
        desc="Train",
        total=train_iter,
        leave=True,
    ):
        model = batch_update(model, batch, optimizer, opt_state)

        if batch_id % eval_freq == 0:
            metrics = eval(model, train_dataset, batch_size, eval_iter, key=next(keys))
            logger.log({"train": metrics})

            metrics = eval(model, val_dataset, batch_size, eval_iter, key=next(keys))
            logger.log({"val": metrics})
