from dataclasses import dataclass
from pathlib import Path

import jax.random as jr
from hydra.utils import to_absolute_path
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    n_nodes: int
    n_graphs: int
    generation_key: PRNGKeyArray
    split_key: PRNGKeyArray

    def __post_init__(self):
        self.generation_key = jr.key(self.generation_key)
        self.split_key = jr.key(self.split_key)


@dataclass
class ModelConfig:
    hidden_dim: int
    n_layers: int
    conv_type: str
    key: PRNGKeyArray

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class TrainerConfig:
    learning_rate: float
    batch_size: int
    train_iter: int
    eval_iter: int
    eval_freq: int
    key: PRNGKeyArray

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class WandbConfig:
    entity: str
    group: str | None
    mode: str

    def __post_init__(self):
        if self.group.lower() == "none":
            self.group = None


@dataclass
class MainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    wandb: WandbConfig

    @classmethod
    def from_dict(cls, config: DictConfig) -> "MainConfig":
        return cls(
            dataset=DatasetConfig(**config.dataset),
            model=ModelConfig(**config.model),
            trainer=TrainerConfig(**config.trainer),
            wandb=WandbConfig(**config.wandb),
        )
