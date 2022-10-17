
from mpi4py import MPI
import pytest
import numpy as np

from simemc.mpi_utils import bcast_large, reduce_large, reduce_large_3d, print0

COMM= MPI.COMM_WORLD


@pytest.mark.mpi(min_size=2)
def test_large_reduce():
    """
    tests methods for reducing and broadcasting large numpy arrays
    :return:
    """
    v = np.random.random((200,200,200))

    print0("reduce whole", flush=True)
    v_all = COMM.reduce(v)
    v_all = bcast_large(v_all)
    #v_all = COMM.bcast(COMM.reduce(v))
    print0("Reduce chunked", flush=True)
    v_all2 = reduce_large(v, verbose=True, sz=64**3, broadcast=False)

    print0("done.", flush=True)
    if COMM.rank==0:
        assert np.allclose(v_all2, v_all)

        print("OK")

    print0("reduce chunked (no broadcast)")
    v_all3 = reduce_large(v, broadcast=False, sz=int(v.size/10), verbose=True)
    if COMM.rank ==0:
        assert np.allclose(v_all3, v_all)
    else:
        assert v_all3 is None

    COMM.barrier()
    if COMM.rank==0:
        print("OK")

    # try an even bigger array
    print0("Creating even bigger array", flush=True)
    v = np.ones((650,650,650))
    v_all4 = reduce_large(v, verbose=True, sz=32**3, broadcast=True, buffers=True)
    #v_all4 = bcast_large(v_all4, verbose=True, comm=COMM)
    assert v_all4.sum() == v.size*COMM.size
    print0("Again", flush=True)
    v_all5 = reduce_large_3d(v, verbose=True, buffers=True)
    v_all5 = bcast_large(v_all5, verbose=True, comm=COMM)
    assert v_all5.sum() == v.size*COMM.size


if __name__=="__main__":
    test_large_reduce()
