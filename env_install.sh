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

conda install -y pytorch=1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/pyg_lib-0.1.0%2Bpt113cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_sparse-0.6.17%2Bpt113cu116-cp38-cp38-linux_x86_64.whl
pip install torch-geometric

pip install ogb