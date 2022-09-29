# This builds script works for a V100

# If you have a dev build of CCTBX built with CONDA,
# then point to the conda installation:
export CPRE=/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalX/conda_base
export CCTBX_MOD=/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules
# Download cub package for block reduction  https://nvlabs.github.io/cub/
export CUB=./CUB/cub-1.8.0/cub
# these arch flags work for v100 and cuda 11
export NVCCFLAGS='--generate-code=arch=compute_70,code=sm_70'

# MPI compile flags

#run the command `mpic++ -showme:compile` (ignore pthread) and libtbx.python -c "import mpi4py;print(mpi4py.get_include())"
export MPI_INC="-I/usr/common/software/sles15_cgpu/openmpi/4.0.3/gcc/include -I/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalX/conda_base/lib/python3.8/site-packages/mpi4py/include"
MPI4PY_INCL=$(libtbx.python -c "import mpi4py;print('-I'+mpi4py.get_include())")
export MPIFLAG="-march=skylake-avx512"

export MPI_LINK='-Xcompiler="-Wl,-rpath,/usr/common/software/sles15_cgpu/openmpi/4.0.3/gcc/lib,--enable-new-dtags" -L/usr/common/software/sles15_cgpu/openmpi/4.0.3/gcc/lib -lmpi'

#run the command `mpic++ -showme:link` , ignore pthread

nvcc -c general.cu -I$CUB -I$CPRE/include -I$CPRE/include/python3.8 \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG \
   --expt-relaxed-constexpr $NVCCFLAGS  -o general.o

nvcc -c emc_internal.cu -I$CUB -I$CPRE/include -I$CPRE/include/python3.8 \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ \
  -l python3.8 -lboost_python38 -lboost_system -lboost_numpy38  \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr $NVCCFLAGS \
  -o emc_internal.o


nvcc -c orient_match.cu -I$CPRE/include -I$CPRE/include/python3.8 \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ \
  -l python3.8 -lboost_python38 -lboost_system -lboost_numpy38  \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr $NVCCFLAGS \
  -o orient_match.o

nvcc -c emc_ext.cpp -I$CUB -I$CPRE/include -I$CPRE/include/python3.8  \
  -I$CCTBX_MOD/eigen $MPI4PY_INC $MPI_INC -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ \
  -lpython3.8 -lboost_python38 -lboost_system  -lboost_numpy38 \
  --compiler-options=-lstdc++,-fPIC,-O3,$MPIFLAG --expt-relaxed-constexpr \
  -o emc_ext.o

# build the shared library
mpic++ -shared emc_ext.o emc_internal.o orient_match.o general.o \
  -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ \
  -L$CUDA_HOME/lib64 -lpython3.8 -lboost_python38  \
  -lboost_numpy38 -lcudart \
  -o emc.so

# NOTE: import with python:  "from simemc.emc import lerpy"
# (just ensure the folder containing emc.so is in PYTHONPATH)
