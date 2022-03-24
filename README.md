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

## Testing
From the repository root run `pytest`. Then, if pytest-mpi is installed,  run e.g. `mpirun -n 2 pytest --with-mpi --no-summary -q`.


