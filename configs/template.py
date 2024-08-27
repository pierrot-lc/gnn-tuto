from dataclasses import dataclass
from pathlib import Path

import jax.random as jr
from hydra.utils import to_absolute_path
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    root: Path
    prefix: str
    key: PRNGKeyArray

    def __post_init__(self):
        self.root = Path(to_absolute_path(self.root))
        self.key = jr.key(self.key)

    @property
    def adjacency_file(self) -> Path:
        return self.root / f"{self.prefix}_A.txt"

    @property
    def graph_id_file(self) -> Path:
        return self.root / f"{self.prefix}_graph_indicator.txt"

    @property
    def graph_labels_file(self) -> Path:
        return self.root / f"{self.prefix}_graph_labels.txt"

    @property
    def node_labels_file(self) -> Path:
        return self.root / f"{self.prefix}_node_labels.txt"


@dataclass
class ModelConfig:
    hidden_dim: int
    n_layers: int
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
