# This builds script works Perlmutter (NERSC)

# ENSURE your CCTBX environment is activated, as well as the CONDA env!

# Only run this script from the CCTBX modules/simemc folder!

# Install some remaining dependencies
libtbx.python -m pip install sympy pytest-mpi pytest IPython

# Install mpi4py on NERSC using this command (this will work with NERSCs slurm srun):
MPICC="$(which cc) -shared -target-accel=nvidia80 -lmpi -lgdrapi" libtbx.python -m pip install --no-binary mpi4py --no-cache-dir mpi4py mpi4py

# Install the reborn dependency
./build_reborn.sh

export CPRE=$CONDA_PREFIX
export CCTBX_MOD=$PWD/..
# Download cub package for block reduction  https://nvlabs.github.io/cub/
git clone https://github.com/NVIDIA/cub.git
export CUB=./cub/cub

# Flags for the A100 GPUs on Perlmutter:
export NVCCFLAGS='--generate-code=arch=compute_80,code=sm_80'

# MPI compile flags
# On Perlmutter I had to run `CC --cray-print-opts=cflags` to get these flags:
export MPI_INC="-I/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/include -I/opt/cray/pe/libsci/23.02.1.1/GNU/9.1/x86_64/include -I/opt/cray/pe/dsmml/0.2.2/dsmml//include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/nvvm/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/extras/CUPTI/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/extras/Debugger/include -I/opt/cray/xpmem/2.5.2-2.4_3.50__gd0f7936.shasta/include"

export MPI4PY_INC=$(python -c "import mpi4py;print(f'-I{mpi4py.get_include()}')")
export MPIFLAG=""
export PYMAJ=$(python -c "import sys;print(sys.version_info[0])")
export PYMIN=$(python -c "import sys;print(sys.version_info[1])")
export PYNUM=$(echo $PYMAJ$PYMIN)
export PY=$(echo python$PYMAJ.$PYMIN)

nvcc -c general.cu -I$CUB -I$CPRE/include -I$CPRE/include/$PY \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG \
   --expt-relaxed-constexpr $NVCCFLAGS  -o general2.o

nvcc -c emc_internal.cu -I$CUB -I$CPRE/include -I$CPRE/include/$PY \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib \
  -lboost_python$PYNUM -lboost_system -lboost_numpy$PYNUM  \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr $NVCCFLAGS \
  -o emc_internal2.o


nvcc -c orient_match.cu -I$CPRE/include -I$CPRE/include/$PY \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib \
  -lboost_python$PYNUM -lboost_system -lboost_numpy$PYNUM  \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr $NVCCFLAGS \
  -o orient_match2.o

nvcc -c emc_ext.cpp -I$CUB -I$CPRE/include -I$CPRE/include/$PY  \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib \
  -lboost_python$PYNUM -lboost_system  -lboost_numpy$PYNUM \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr \
  -o emc_ext2.o

# build the shared library using the CC compiler wrapper (for perlmutter only)
CC -shared emc_ext2.o emc_internal2.o orient_match2.o general2.o \
  -L$CPRE/lib \
  -L$CUDA_HOME/lib64 -lboost_python$PYNUM  \
  -lboost_numpy$PYNUM -lcudart -lmpi \
  -o emc.so

# NOTE: import with python:  "from simemc.emc import lerpy"
# (just ensure the folder containing emc.so is in PYTHONPATH)

