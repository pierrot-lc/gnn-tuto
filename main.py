import hydra
import optax
import wandb
from configs.template import MainConfig
from omegaconf import DictConfig, OmegaConf
from src.dataset import Dataset
from src.gnn import GNNClassifier
from src.trainer import train


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    config = MainConfig.from_dict(dict_config)

    dataset = Dataset.from_files(
        config.dataset.adjacency_file,
        config.dataset.graph_id_file,
        config.dataset.graph_labels_file,
        config.dataset.node_labels_file,
    )
    train_dataset, val_dataset = Dataset.split(
        dataset, split=0.8, key=config.dataset.key
    )

    model = GNNClassifier(
        config.model.hidden_dim,
        dataset.n_atoms,
        config.model.n_layers,
        key=config.model.key,
    )

    optimizer = optax.adamw(learning_rate=config.trainer.learning_rate)

    with wandb.init(
        project="gnn-tuto",
        group=config.wandb.group,
        config=OmegaConf.to_container(dict_config),
        entity=config.wandb.entity,
        mode=config.wandb.mode,
    ) as run:
        train(
            model,
            train_dataset,
            val_dataset,
            optimizer,
            config.trainer.batch_size,
            config.trainer.train_iter,
            config.trainer.eval_iter,
            config.trainer.eval_freq,
            logger=run,
            key=config.trainer.key,
        )


if __name__ == "__main__":
    main()
