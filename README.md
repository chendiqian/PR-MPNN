# Probabilistically Rewired Message-Passing Neural Networks

<img src="https://github.com/chendiqian/PR-MPNN/blob/main/main-figure.png" alt="drawing" width="800"/>
<p align="center">
</p>

Reference implementation of our rewiring method as proposed in 

[Probabilistically Rewired Message-Passing Neural Networks](https://arxiv.org/abs/2310.02156)  
Chendi Qian*, Andrei Manolache*, Kareem Ahmed, Zhe Zeng, Guy Van den Broeck, Mathias Niepert<sup>†</sup>, Christopher Morris<sup>†</sup>

*These authors contributed equally.  
<sup>†</sup>Co-senior authorship.

## Environment setup
```
conda create -y -n prmpnn python=3.10
conda activate prmpnn

conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric==2.4.0  # maybe latest also works
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp310-cp310-linux_x86_64.whl

pip install ogb
pip install ml-collections
pip install sacred
pip install wandb
pip install gdown

# maybe need to downgrade numpy
pip install numpy=1.26.4
```

## Datasets
We empirically evaluate our rewiring method on multiple datasets.

### Real-world datasets
TUDatasets: [PyG class](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html?highlight=tudatasets), [paper](https://arxiv.org/pdf/2007.08663.pdf)
- ZINC
- Alchemy
- MUTAG
- PRC_MR
- PROTEINS
- NCI1
- NCI109
- IMDB-B
- IMDB-M

OGB: [website](https://ogb.stanford.edu/), [paper](https://arxiv.org/abs/2005.00687)
- ogbg-molhiv

WebKB: [PyG class](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html?highlight=webkb)
- Cornell
- Texas
- Wisconsin

LRGB: [code](https://github.com/vijaydwivedi75/lrgb), [paper](https://arxiv.org/abs/2206.08164)
- peptides-func
- peptides-struct

QM9 used in [DRew](https://github.com/BenGutteridge/DRew) and [SP-MPNN](https://github.com/radoslav11/SP-MPNN). Note there are different versions of QM9, e.g., [PPGN](https://github.com/hadarser/ProvablyPowerfulGraphNetworks)

### Synthetic datasets

EXP: [code](https://github.com/ralphabb/GNN-RNI), [paper](https://arxiv.org/pdf/2010.01179.pdf)

CSL: [code](https://github.com/PurdueMINDS/RelationalPooling), [paper](https://proceedings.mlr.press/v97/murphy19a/murphy19a.pdf)

Trees-NeighborsMatch: [code](https://github.com/tech-srl/bottleneck/), [paper](https://arxiv.org/abs/2006.05205)

Trees-LeafColor: Our own :star: :star: :star:

## Rewire options
We provide rewiring options as following:

- Add edges / remove edges

- Directed / undirected: meaning adding or deleting edges in a directed way or not. If not, will add _and_ remove undirected edges.

- Separated / merged: if separated, will sample 2 graphs, one with edges added and the other with edges removed. If merged, will merge the 2 graphs as one.

## Sampler candidates
- SIMPLE, [code](https://github.com/UCLA-StarAI/SIMPLE), [paper](https://arxiv.org/abs/2210.01941)
- I-MLE, [code](https://github.com/uclnlp/torch-imle), [paper](https://arxiv.org/abs/2106.01798)
- Gumbel softmax for [subset sampling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html)

## To replicate experiments
We provide yaml files under `configs`, run e.g. 
`python run.py with PATH_TO_CONFIG`

Note that this repo provides a taste of how PR-MPNN works, with examples given by GIN network. For replicating the results in our paper, please see to the `backup` branch, or contact [chendi.qian@log.rwth-aachen.de](mailto:chendi.qian@log.rwth-aachen.de)

## Known issue
If you are using a different version of PyTorch, you might have some error in the gradients produced by SIMPLE gradient estimator. You might want to check if SIMPLE gives nonzero gradients.
