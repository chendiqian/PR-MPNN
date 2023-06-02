# Unofficial readme to remind myself

## Environment setup
simply run `env_install.sh`

## Datasets
TBA

## Rewire method
We provide rewiring options as following:

- Add edges / remove edges

- Directed / undirected: meaning adding or deleting edges in a directed way or not. If not, will add _and_ remove undirected edges.

- Separated / merged: if separated, will sample 2 graphs, one with edges added and the other with edges removed. If merged, will merge the 2 graphs as one.

- In-place / not in-place: if in-place, will add the edges based on the original edges, otherwise will return a graph with _only_ the added edges.

- Candidate range options: from selected set or N ^ 2 possible edges. The upstream models will be GNN + MLP or transformer respectively.

- Shared-weights downstream models: we use distinct sets of weights for each rewired graph and the original graph. If you like the weights to be shared, set the model as e.g. `gin_duo_shared`.  

## Samplers that we use
- SIMPLE, [code](https://github.com/UCLA-StarAI/SIMPLE), [paper](https://arxiv.org/abs/2210.01941)
- I-MLE, [code](https://github.com/uclnlp/torch-imle), [paper](https://arxiv.org/abs/2106.01798)
- Gumbel softmax for [subset sampling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html)

## To replicate experiments
We provide yaml files under `configs`.

For fixed hyperparams, run e.g. 
`python main_fixconf.py with configs/zinc/global/topk20_1_random.yaml`

For a sweep, run e.g.
```
wandb sweep configs/zinc/global/sweep_20_1_simple.yaml
wandb agent $ID
```