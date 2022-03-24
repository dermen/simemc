#!/bin/bash

# update the CUDA install for your system
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py

python bootstrap.py hot update base \
    --builder=dials --nproc=32 --use-conda  --python=38

source ./conda_base/etc/profile.d/conda.sh
conda activate ./conda_base

CC=$(which gcc) CXX=$(which g++) python bootstrap.py build \
    --builder=dials --nproc=32 --use-conda=./conda_base --python=38 \
    --config-flags='--enable_cxx11' --config-flags='-- enable_openmp_if_possible=True' \
    --config-flags="--compiler=conda" --config-flags="--enable_cuda"
