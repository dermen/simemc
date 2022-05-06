# simemc
diffraction simulations and EMC

## Dependencies

* Developer build of [CCTBX](https://github.com/cctbx/cctbx_project) with [mpi4py](https://mpi4py.readthedocs.io/en/stable/#) support
* [reborn](https://kirianlab.gitlab.io/reborn/)
* [sympy](https://www.sympy.org/en/index.html)
* [pytest-mpi](https://pypi.org/project/pytest-mpi/) (optional)
* [kern line profiler](https://github.com/rkern/line_profiler.git)
* [CUB](https://nvlabs.github.io/cub/) (for doing block reduction in CUDA, see the example `build_trilerp.sh` script)
* Possibly some other python dependencies, run the tests and install any missing dependencies using PIP

## Building

1) Clone simemc (if not using ssh keys for github, change URL)

```
git clone git@github.com:dermen/simemc.git
```

2) Install dev CCTBX (edit and run the script `build_cctbx_dev_gpu_mpi.sh`) . Note what you set as the value of CCTBXROOT. This might take some tinkering with (DIALS/CCTBX mailining lists are active and responsive)

```
cd simemc
./build_cctbx_dev_gpu_mpi.sh
```

3) Activate the cctbx environment

```
source CCTBXROOT/build/conda_setpaths.sh
# Note, to deactivate run CCTBXROOT/build/conda_unsetpaths.sh
```

4) Install reborn. Edit the value of CCTBXROOT in `build_reborn.sh` and run the script

```
./build_reborn.sh
```

5) Get the other python dependencies

```
libtbx.pip install pytest-mpi line_profiler sympy
```

6) Build the `simemc` extension module from the root of the simemc repository (copy the `build_trilerp.sh` example script, and edit it accordingly)

```
cd simemc
./build_trilperp.sh
```

7) Softlink `simemc` to the cctbx modules folder

```
ln -s /full/path/to/simemc $CCTBXROOT/modules
```

8) Make a startup script that can be sourced at *each new login*. Example:

```
#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CCTBXROOT=~/xtal_gpu  # whatever this was set as when CCTBX was built
source $CCTBXROOT/build/conda_setpaths.sh
```



## Testing
From the repository root run `libtbx.pytest`. Then, optionally test the `mpi4py` installation using e.g. `mpirun -n 2 libtbx.pytest --with-mpi --no-summary -q` (provided `pytest-mpi` is installed).

The first time you run the tests, first run `iotbx.fetch_pdb 4bs7`. That command will download the PDB file used in the simulations. 

## Run the pipeline

Here we simulate 999 shots on a machine with 24 processors and 1 v100 GPU. We need to first downloaded the PDB file `4bs7` using `iotbx.fetch_pdb 4bs7`. We then create a quaternion file containing the orientation samples.

```
# prep (from the simemc repository root):
iotbx.fetch_pdb 4bs7
cd quat
gcc make-quaternion.c -O3 -lm -o quat
./quat -bin 70
cd ../
```

The script then runs them through EMC for a set number of iterations 

```
DIFFBRAGG_USE_CUDA=1 mpirun -n 4 libtbx.python tests/emc_iteration.py  1 250 water_sims --niter 100 --phil proc.phil  --minpred 3 --hcut 0.1  --xtalsize 0.0025 --densityUpdater lbfgs
```

Process the images using the standard stills process framework as a comparison:

```
mpirun  -n 24 dials.stills_process \
  proc.phil  water_sims/cbfs/*.cbf filter.min_spot_size=2 \
  output.output_dir=water_sims/proc mp.method=mpi
```

```
mpirun -n 24 cctbx.xfel.merge merge.phil \
  input.path=water_sims/proc output.output_dir=water_sims/merge
```

Because we ran a simulation, we know the ground truth structure factors. We can plot the correlation between the EMC-determined structure factors with the ground truth. We can do the same for `stills_process` / `xfel.merge` determined structure factors. See the script `make_corr_plot.py`:

```
libtbx.python make_corr_plot.py water_sims/Witer10.h5  --mtz water_sims/merge/iobs_all.mtz
```

Note the EMC-determined structure factors correlate much better for these simulated data with low spot counts. 
