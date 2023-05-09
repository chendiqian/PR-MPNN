# Unofficial readme to remind myself

## Environment setup
simply run `env_install.sh`

## Datasets
TBA

## Rewire method
We provide the rewire methods as follows:

### Transformer as upstream model:

- Global directed rewire

- Global undirected rewire

- Global edge addition

### Linear models as upstream model

- Local undirected rewire from edge candidates

- _TBA_

The methods can either be in-place, i.e., added to the original graph, or treated as a new separate graph.

The original graphs can be included in a graph batch, if so, we have options for weight sharing.

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