from mpi4py import MPI
COMM = MPI.COMM_WORLD


def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)


def printR(*args, **kwargs):
    print("rank%d"%COMM.rank,*args, **kwargs)
