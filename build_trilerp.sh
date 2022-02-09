# This builds script works for a V100

# If you have a dev build of CCTBX built with CONDA, then point to the conda installation:
export CPRE=/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalX/conda_base

# Download cub package for block reduction  https://nvlabs.github.io/cub/
export CUB=./CUB/cub-1.8.0/cub
# these arch flags work for v100 and cuda 11
export NVCCFLAGS='--generate-code=arch=compute_70,code=sm_70'
nvcc -c cuda_trilerp.cu -I$CUB -I$CPRE/include -I$CPRE/include/python3.8 -I$MODZ/eigen  -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -l python3.8 -lboost_python38 -lboost_system -lboost_numpy38  --compiler-options=-lstdc++,-fPIC,-O3 -o cuda_trilerp.o --expt-relaxed-constexpr $NVCCFLAGS

nvcc -c trilerp_ext.cpp -I$CUB -I$CPRE/include -I$CPRE/include/python3.8  -I$MODZ/eigen -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8 -lboost_python38 -lboost_system  -lboost_numpy38 --compiler-options=-lstdc++,-fPIC,-O3 -o trilerp.o --expt-relaxed-constexpr

# build the shared library
g++ -shared trilerp.o cuda_trilerp.o -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -L$CUDA_HOME/lib64 -lpython3.8 -lboost_python38  -lboost_numpy38 -lcudart -o trilerp.so

# NOTE: import with python:  "from trilerp import lerpy"
# (just ensure the folder containing trilperp.so is in PYTHONPATH)
