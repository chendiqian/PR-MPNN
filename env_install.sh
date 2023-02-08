#!/bin/bash

echo y | conda create -n diffstruct python=3.8
#conda activate diffstruct
source ${CONDA_PREFIX}/bin/activate diffstruct

pip install ml-collections
pip install numba
pip install tqdm
pip install ortools
pip install sacred
pip install PyYAML
pip install setuptools==59.5.0

echo y | conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
pip install torch-geometric

pip install tensorboard
pip install ogb