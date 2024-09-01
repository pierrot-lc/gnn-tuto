from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from beartype import beartype
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar, jaxtyped
from scipy.stats import kendalltau
from tqdm import tqdm
from wandb.wandb_run import Run

from .dataset import Dataset, GraphData
from .gnn import RankingModel


def key_source(key: PRNGKeyArray) -> Iterator[PRNGKeyArray]:
    """Infinite source of random keys."""
    while True:
        key, sk = jr.split(key)
        yield sk


def count_params(model: eqx.Module) -> int:
    """Count the number of parameters of the given equinox module."""
    params = eqx.filter(model, eqx.is_array)
    n_params = jax.tree.map(lambda p: jnp.prod(jnp.array(p.shape)), params)
    n_params = jnp.array(jax.tree.leaves(n_params))
    n_params = jnp.sum(n_params)
    return int(n_params)


@eqx.filter_jit
def margin_ranking_loss(
    pred_scores: Float[Array, " n_nodes"],
    true_scores: Float[Array, " n_nodes"],
    mask: Bool[Array, " n_nodes"],
    key: PRNGKeyArray,
    margin: float = 1.0,
    sampling_factor: float = 1.0,
) -> Scalar:
    """See https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html."""
    n_nodes = len(pred_scores)
    total_sampling = int(n_nodes * sampling_factor)
    perms = jr.choice(
        key,
        jnp.arange(n_nodes),
        (2, total_sampling),
        replace=True,
        p=mask / mask.sum(),  # Ignore masked nodes.
    )

    argsort = jnp.argsort(true_scores, descending=False)
    pred_scores = pred_scores[argsort]
    true_scores = true_scores[argsort]
    ranks = jnp.arange(n_nodes)

    x1_rank = ranks[perms[0]]
    x2_rank = ranks[perms[1]]
    x1_pred = pred_scores[perms[0]]
    x2_pred = pred_scores[perms[1]]

    y = jnp.sign(x1_rank - x2_rank)
    y = jax.lax.stop_gradient(y)
    loss = jnp.clip(-y * (x1_pred - x2_pred) + margin, min=0)

    # Do not penalize if the true scores are equal.
    x1_true = true_scores[perms[0]]
    x2_true = true_scores[perms[1]]
    loss = jnp.where(x1_true == x2_true, 0.0, loss)

    return jnp.mean(loss)


@jaxtyped(typechecker=beartype)
def batch_loss(model: RankingModel, batch: GraphData, key: PRNGKeyArray) -> Scalar:
    batch_size = batch.scores.shape[0]
    keys = jr.split(key, batch_size)

    pred_scores = jax.vmap(model)(batch.adjacency, batch.edges)
    losses = jax.vmap(margin_ranking_loss)(pred_scores, batch.scores, batch.mask, keys)
    return jnp.mean(losses)


def batch_metrics(
    model: RankingModel, batch: GraphData, key: PRNGKeyArray
) -> dict[str, Array]:
    metrics = dict()
    batch_size = batch.scores.shape[0]
    keys = jr.split(key, batch_size)

    pred_scores = jax.vmap(model)(batch.adjacency, batch.edges)
    metrics["loss"] = jax.vmap(margin_ranking_loss)(
        pred_scores, batch.scores, batch.mask, keys
    )

    kt_scores = jnp.array(
        [
            kendalltau(
                pred_[mask], true_[mask], nan_policy="raise", method="auto"
            ).statistic
            for pred_, true_, mask in zip(pred_scores, batch.scores, batch.mask)
        ]
    )
    # Score can be inf if all nodes have the same score.
    kt_scores = jnp.where(jnp.isfinite(kt_scores), kt_scores, 0.0)
    metrics["KT score"] = kt_scores
    return metrics


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def batch_update(
    model: RankingModel,
    batch: GraphData,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
) -> RankingModel:
    grads = eqx.filter_grad(batch_loss)(model, batch, key)
    params, static = eqx.partition(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    model = eqx.combine(params, static)
    return model


def eval(
    model: RankingModel,
    dataset: Dataset,
    batch_size: int,
    eval_iter: int,
    *,
    key: PRNGKeyArray,
) -> dict[str, float]:
    model = eqx.tree_inference(model, value=True)
    keys = key_source(key)
    metrics = [
        batch_metrics(model, batch, next(keys))
        for batch in tqdm(
            dataset.iter(batch_size, eval_iter, next(keys)),
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
    model: RankingModel,
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
    keys = key_source(key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    logger.summary["n_params"] = count_params(model)
    logger.summary["training_size"] = len(train_dataset)
    logger.summary["validation_size"] = len(val_dataset)
    logger.summary["n_nodes"] = train_dataset.graphs[0].scores.shape[0]
    logger.summary["n_edges"] = train_dataset.graphs[0].edges.shape[0]

    for batch_id, batch in tqdm(
        enumerate(train_dataset.iter(batch_size, train_iter, next(keys))),
        desc="Train",
        total=train_iter,
        leave=True,
    ):
        model = batch_update(model, batch, optimizer, opt_state, next(keys))

        if batch_id % eval_freq == 0:
            metrics = eval(model, train_dataset, batch_size, eval_iter, key=next(keys))
            logger.log({"train": metrics}, step=batch_id)

            metrics = eval(model, val_dataset, batch_size, eval_iter, key=next(keys))
            logger.log({"val": metrics}, step=batch_id)
