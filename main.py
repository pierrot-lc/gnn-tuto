from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import jax.random as jr
import hydra
from src.dataset import Dataset
from src.gnn import GNNClassifier
from configs.template import MainConfig


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    config = MainConfig.from_dict(dict_config)

    dataset = Dataset.from_files(
        config.dataset.adjacency_file,
        config.dataset.graph_id_file,
        config.dataset.graph_labels_file,
        config.dataset.node_labels_file,
    )

    model = GNNClassifier(
        config.model.hidden_dim,
        dataset.n_atoms,
        config.model.n_layers,
        key=config.model.key,
    )

if __name__ == "__main__":
    main()
