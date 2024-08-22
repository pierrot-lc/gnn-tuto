from pathlib import Path
from src.dataset import Dataset

root = Path("./data/NCI1")
dataset = Dataset.from_files(
    root / "NCI1_A.txt",
    root / "NCI1_graph_indicator.txt",
    root / "NCI1_graph_labels.txt",
    root / "NCI1_node_labels.txt",
)
g = dataset[0]
