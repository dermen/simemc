from scipy.spatial.transform import Rotation
from mpi4py import MPI
import numpy as np
import pytest


from simemc.emc import probable_orients
from simemc import mpi_utils

COMM = MPI.COMM_WORLD

@pytest.mark.mpi(min_size=2)
def test(ndev=1):
    """
    This method tests the cuda IPC protocol implemented in the probable_orients class

    The IPC use-case here is specifically for when multiple processes share a single GPU

    The idea is that, if the number of rotation matrices is large, we should only
    allocate space for it once, and then share that memory with other processes
    running on the same GPU.
    :param ndev: number of devices
    """

    dev_id = COMM.rank % ndev
    O = probable_orients()

    # this is a communicator for processes sharing a GPU
    DEVICE_COMM = mpi_utils.get_host_dev_comm(dev_id)

    # unit cell matrix used in probable_orients:
    Bmat = .01, 0, 0, 0, .01, 0, 0, 0, .05

    # generate some qvecs
    max_q = 100  # maximum number of strong spots e.g.
    qvecs = np.random.uniform(-0.25,0.25,(100,3)).ravel()

    num_rot_mats = 100000
    np.random.seed(0)
    if DEVICE_COMM.rank == 0:  #only the root process per GPU gets the actual memory
        rotMats = Rotation.random(num_rot_mats).as_matrix().ravel()
    else:
        rotMats = np.empty([])

    # run the IPC protocol, memory only allocated once per GPU
    O.allocate_orientations_IPC(dev_id, rotMats, max_q, num_rot_mats, DEVICE_COMM)
    O.Bmatrix = Bmat
    prob_rot_IPC = O.orient_peaks(qvecs, 0.1, 3, False)
    O.free_device()

    # run the non-IPC protocol, memory allocated once per process running on a GPU (bad if rotMats is large)
    rotMats = mpi_utils.bcast_large(rotMats, comm=DEVICE_COMM)
    O = probable_orients()
    O.allocate_orientations(dev_id, rotMats, max_q)
    O.Bmatrix = Bmat
    prob_rot = O.orient_peaks(qvecs, 0.1, 3, False)

    assert len(prob_rot)==len(prob_rot_IPC)
    assert np.allclose(prob_rot, prob_rot_IPC)
    COMM.barrier()
    if COMM.rank==0:
        print("OK")


if __name__=="__main__":
    import sys
    try:
        ndev = int(sys.argv[1])
    except IndexError:
        ndev = 1
    test(ndev)
