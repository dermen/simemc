#!/bin/bash -l

#SBATCH -q special
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J lyso_sim
#SBATCH -C gpu 
#SBATCH -c 80
#SBATCH --gres=gpu:8 
#SBATCH -A m1759
#SBATCH -o job%j.out
#SBATCH -e job%j.err

# NERSC environment script:
module purge
module load cgpu gcc openmpi cuda

# Source to setpaths.sh , this file is created by CCTBX dev build script
source /global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/gpubuild3/setpaths.sh
export DIFFBRAGG_USE_CUDA=1
time srun -N1 --tasks-per-node=16 -c2 \
  libtbx.python sim_lyso.py 1600 1600sim_wBG --sparse
