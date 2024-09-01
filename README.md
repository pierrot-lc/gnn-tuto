# Graph Neural Networks from Scratch with JAX

This repo is an example of an implementation of GNNs using JAX. It contains two
implementations:

1. The standard GCN using the sum aggregation implementated with the adjacency
   matrix.
2. A more modern [GAT][gat-paper] implemented using the list of edges.

Those two implementations should give the reader the keys to implement its own
GNN.

## Training objective

The code trains the GNN to rank the nodes of randomly generated graphs using
the [nx.clustering][node-clustering] score. The [margin-ranking
loss][margin-ranking-loss] is used. The [Kendall rank][kendall-rank] is used to
judge the correlation between the produced rank from the model and the actual
rank from the scores.

## Usage

This repo needs python 3.12 and the requirements from `requirements.txt`. You
can specify the hyperparameters by editing the `./configs/default.yaml` file.
Then, simply launch the training with:

```sh
python3 main.py
```

The training is logged to WandB. Have a look to the [Hydra][hydra] (for HPs
config) and [WandB][wandb] (for logging) docs for more.

You can have a look at my trainings [here][wandb-space].

## Sources

- [GNN introduction from distill.pub][gnn-intro]
- [GAT paper][gat-paper]

[gat-paper]:            https://arxiv.org/abs/1710.10903
[gnn-intro]:            https://distill.pub/2021/gnn-intro/
[hydra]:                https://hydra.cc/
[kendall-rank]:         https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
[margin-ranking-loss]:  https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
[node-clustering]:      https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
[wandb]:                https://wandb.ai/
[wandb-space]:          https://wandb.ai/pierrotlc/gnn-tuto
