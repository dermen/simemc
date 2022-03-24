# this will be the prime install area
export CCTBXROOT=$HOME/xtal_gpu3
mkdir $CCTBXROOT
cd $CCTBXROOT
# Once this script is finished, inside of CCTBX_ROOT will be the folders
# modules, conda_base, and build
# When re-logging in, you will want to create a CCTBX env script which contains the CUDA env vars (below)
# as well as run: source CCTBX_ROOT/build/conda_setpaths.sh

# update the CUDA install for your system
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

wget \
https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py

python bootstrap.py hot update base \
    --builder=dials --nproc=32 --use-conda  --python=38

source ./conda_base/etc/profile.d/conda.sh
conda activate ./conda_base
# Install mpi4py and openmpi. we uninstall h5py because installing openmpi will lilkely change the hdf5 lib
conda uninstall -y h5py 
conda install -y openmpi mpi4py numba h5py
# Note, if you already have mpi installed, install mpi4py using pip using. This is whats done on supercomputers where the mpi installations are carefully tuned and loaded via modules
# Optional mpi4py install using pip:
# MPICC=mpicc python -m pip install -v --no-binary mpi4py mpi4py

CC=$(which gcc) CXX=$(which g++) python bootstrap.py build \
    --builder=dials --nproc=32 --use-conda=./conda_base --python=38 \
    --config-flags='--enable_cxx11' --config-flags='--enable_openmp_if_possible=True' \
    --config-flags="--compiler=conda" --config-flags="--enable_cuda"
