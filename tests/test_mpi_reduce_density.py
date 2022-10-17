from mpi4py import MPI
COMM = MPI.COMM_WORLD

from simemc import mpi_utils
import time
from simemc.emc import lerpy
from simemc.dragon_geom import DET, BEAM
from scipy.spatial.transform import Rotation
from simemc import utils
import numpy as np
import pytest


@pytest.mark.mpi(min_size=2)
def test():
    _test()


def _test():
    dens_dim=551
    max_q=0.5
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    dev_id = 0
    rotMats = Rotation.random(100, random_state=0).as_matrix()

    n = dens_dim
    X_MIN = L.xmin
    X_MAX = L.xmax
    c,d = utils.corners_and_deltas(L.dens_sh, X_MIN, X_MAX)

    fdim,sdim = DET[0].get_image_size()
    img_sh = len(DET), sdim, fdim
    npix = int(np.product(img_sh))

    qx,qy,qz = map(lambda x: x.ravel(), utils.calc_qmap(DET, BEAM))

    qcoords = np.vstack((qx,qy,qz)).T
    assert len(qcoords) == npix
    L.allocate_lerpy(
        dev_id, rotMats, npix,
        c,d, qcoords,
        rotMats.shape[0], npix)

    Wstart = None
    if COMM.rank==0:
        Wstart = np.random.random((n,n,n))
    # this should copy Wstart to gpu.densities on rank=0 and them broadcast to all other ranks
    L.mpi_set_starting_densities(Wstart, COMM)

    # copy Wstart to all other ranks, and verify gpu.densities was set properly by mpi_set_starting_densities
    Wstart = mpi_utils.bcast_large(Wstart, verbose=True, comm=COMM)
    assert np.allclose(L.densities(), Wstart.ravel())

    L.toggle_insert()
    assert np.allclose(L.wts(), 0)
    assert np.allclose(L.densities(), 0)

    vals = np.ones(img_sh)

    L.trilinear_insertion(0, vals)

    W = L.densities()
    wts = L.wts()

    W = W.reshape(L.dens_sh)
    wts = wts.reshape(L.dens_sh)
    t = time.time()
    Wred_external = mpi_utils.reduce_large_3d(W, verbose=True, buffers=True)
    wtsred_external = mpi_utils.reduce_large_3d(wts, verbose=True, buffers=True)
    text = time.time()-t

    t = time.time()
    L.reduce_densities(COMM)
    L.reduce_weights(COMM)
    tint = time.time()-t
    Wred_internal = L.densities()
    wtsred_internal = L.densities()
    if COMM.rank==0:
        Wred_external = Wred_external.ravel()
        wtsred_external = wtsred_external.ravel()
        print("internal reduce took %f sec, external reduce took %f sec" % (tint, text))
        assert np.allclose(Wred_external, Wred_internal)
        assert np.allclose(wtsred_external, wtsred_internal)
        sig_red = Wred_internal[Wred_internal>0].mean()
        sig = W[W>0].mean()
        print("Mean sig: %f, sig_red: %f" % (sig, sig_red))
        assert np.allclose(sig*COMM.size, sig_red, atol=1e-1)
        print("Ok")


if __name__=="__main__":
    test()
