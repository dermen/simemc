# simemc
diffraction simulations and EMC

## Dependencies

* Developer build of CCTBX with mpi4py support
* [reborn](https://kirianlab.gitlab.io/reborn/)
* [pytest-mpi](https://pypi.org/project/pytest-mpi/) (optional)

## Testing
From the repository root run `pytest`. Then, if pytest-mpi is installed,  run `mpirun -n2 pytest --with-mpi`.


