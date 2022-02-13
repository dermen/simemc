from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os


def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)


def printR(*args, **kwargs):
    print("rank%d"%COMM.rank,*args, **kwargs)


def make_dir(dirname):
    if COMM.rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    COMM.barrier()
