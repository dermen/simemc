# simemc
diffraction simulations and EMC

## Dependencies

* Developer build of CCTBX with mpi4py support
* [reborn](https://kirianlab.gitlab.io/reborn/)
* [sympy](https://www.sympy.org/en/index.html)
* [pytest-mpi](https://pypi.org/project/pytest-mpi/) (optional)
* [kern line profiler](https://github.com/rkern/line_profiler.git)
* [CUB](https://nvlabs.github.io/cub/) (for doing block reduction in CUDA, see the example `build_trilerp.sh` script)
* possible some other python dependencies, Try running the tests and install missing dependencies using PIP

## Building


## Testing
From the repository root run `libtbx.pytest`. Then, optionally test the `mpi4py` installation using e.g. `mpirun -n 2 libtbx.pytest --with-mpi --no-summary -q` (provided `pytest-mpi` is installed).

## Run the pipeline

Here we simulate 999 shots on a machine with 24 processors and 1 v100 GPU. The script then runs them through EMC for a set number of iterations 

```
DIFFBRAGG_USE_CUDA=1 mpirun -n 3 libtbx.python tests/test_emc_iteration.py  \
  1 333  water_sims/1um --water --phil proc.phil  --minpred 4 \
  --hcut=0.04 --cbfdir water_sims/1um-cbfs --xtalsize 0.00125 --niter 15
```

Process the images using the standard stills process framework as a comparison:

```
mpirun  -n 24 dials.stills_process proc.phil  water_sims/1um-cbfs/*.cbf output.output_dir=water_sims/1um-proc mp.method=mpi
```

```
mpirun -n 24 cctbx.xfel.merge merge.phil  input.path=water_sims/1um-proc output.output_dir=water_sims/1um-xfelmerge
```

Because we ran a simulation, we know the ground truth structure factors. We can plot the correlation between the EMC-determined structure factors with the ground truth. We can do the same for `stills_process` / `xfel.merge` determined structure factors. See the script `make_corr_plot.py`:

```
libtbx.python make_corr_plot.py water_sims/1um/Witer11.h5  --mtz water_sims/1um-test8/xfel_merge/iobs_all.mtz
```

Note the EMC-determined structure factors correlate much better for these simulated data with low spot counts. 
