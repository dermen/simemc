#!/bin/bash -l

#SBATCH -q special
#SBATCH -N 4 
#SBATCH -t 03:00:00
#SBATCH -J lyso_sim
#SBATCH -C gpu 
#SBATCH -c 80
#SBATCH --gres=gpu:8 
#SBATCH -A m1759
#SBATCH -o job%j.out
#SBATCH -e job%j.err

source  ~/stable.cuda.Z.sh
DIFFBRAGG_USE_CUDA=1 srun -N4 --tasks-per-node=24 -c2 libtbx.python sim_lyso.py  shots_2um

for n in {1..9};do srun -N4 --tasks-per-node=40 -c2  dials.stills_process proc.phil  shots_2um/rank${n}*/*cbf output.output_dir=shots_2um/proc composite_output=False  mp.method=mpi output.integration_pickle=None output.refined_experiments_filename=None; done
