# Probabilistically Rewired Message-Passing Neural Networks

<img src="https://github.com/chendiqian/PR-MPNN/blob/rewire/main-figure.png" alt="drawing" width="800"/>
<p align="center">
</p>

Reference implementation of our rewiring method as proposed in 

[**Probabilistically Rewired Message-Passing Neural Networks
**](https://arxiv.org/abs/2310.02156)  
Chendi Qian*, Andrei Manolache*, Kareem Ahmed, Zhe Zeng, Guy Van den Broeck, Mathias Niepert<sup>†</sup>, Christopher Morris<sup>†</sup>

*These authors contributed equally.  
<sup>†</sup>Co-senior authorship.

## Environment setup
```
conda create -y -n prmpnn python=3.10
conda activate prmpnn
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

pip install torch_geometric
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp310-cp310-linux_x86_64.whl

conda install -y openbabel fsspec rdkit -c conda-forge
pip install ogb
pip install ml-collections
pip install numba
pip install sacred
pip install PyYAML
pip install wandb
pip install matplotlib
pip install seaborn
pip install GraphRicciCurvature
pip install gdown
```

## Datasets
TBA

## Rewire options
We provide rewiring options as following:

- Add edges / remove edges

- Directed / undirected: meaning adding or deleting edges in a directed way or not. If not, will add _and_ remove undirected edges.

- Separated / merged: if separated, will sample 2 graphs, one with edges added and the other with edges removed. If merged, will merge the 2 graphs as one.

- In-place / not in-place: if in-place, will add the edges based on the original edges, otherwise will return a graph with _only_ the added edges.

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