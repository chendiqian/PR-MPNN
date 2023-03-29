#!/bin/bash

echo y | conda create -n nightly python=3.10
#conda activate diffstruct
source ${CONDA_PREFIX}/bin/activate nightly

conda install pytorch  pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
pip install cmake
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_geometric

pip install ogb
pip install ml-collections
#pip install numba
pip install sacred
pip install PyYAML
pip install wandb
pip install matplotlib
pip install seaborn