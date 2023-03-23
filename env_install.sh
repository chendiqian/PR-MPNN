#!/bin/bash

echo y | conda create -n diffstruct python=3.10
#conda activate diffstruct
source ${CONDA_PREFIX}/bin/activate diffstruct

pip install ml-collections
#pip install numba
pip install sacred
pip install PyYAML
pip install wandb
pip install matplotlib

conda install -y pytorch=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/pyg-team/pyg-lib.git
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install torch-geometric

pip install ogb