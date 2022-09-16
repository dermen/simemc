
from simemc.emc import lerpy
from simemc.dragon_geom import DET, BEAM
from scipy.spatial.transform import Rotation
from simemc import utils
import numpy as np
import pytest


@pytest.mark.mpi_skip()
def test():
    _test()

@pytest.mark.mpi_skip()
def test_highRes():
    _test(True)

def _test(highRes=False):
    if highRes:
        dens_dim=512
        max_q=0.5
    else:
        dens_dim=256
        max_q=0.25
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    dev_id = 0
    rotMats = Rotation.random(100, random_state=0).as_matrix()

    n = dens_dim
    W = np.random.random((n,n,n))
    X_MIN = L.xmin
    X_MAX = L.xmax
    c,d = utils.corners_and_deltas(W.shape, X_MIN, X_MAX)

    fdim,sdim = DET[0].get_image_size()
    img_sh = len(DET), sdim, fdim
    npix = int(np.product(img_sh))

    qx,qy,qz = map(lambda x: x.ravel(), utils.calc_qmap(DET, BEAM))

    qcoords = np.vstack( (qx,qy,qz)).T
    assert len(qcoords) == npix
    L.allocate_lerpy(
        dev_id, rotMats, W, npix,
        c,d, qcoords,
        rotMats.shape[0], npix)

    assert np.allclose(L.densities(), W.ravel())

    try:
        wts = L.wts()
        raise RuntimeError("Wts hasnt been allocated yet, so an error should have been thrown")
    except TypeError:
        pass
    L.toggle_insert()
    assert np.allclose(L.wts(), 0)
    assert np.allclose(L.densities(), 0)

    vals = np.ones(img_sh)

    L.trilinear_insertion(0, vals)
    W1  = L.densities()
    wts1 = L.wts()

    L.trilinear_insertion(0, vals)
    W2  = L.densities()
    wts2 = L.wts()
    assert np.allclose(W1*2, W2)
    assert np.allclose(wts1*2, wts2)

    W2 = utils.errdiv(W2, wts2)
    L.update_density(W2)
    W_rt = L.trilinear_interpolation(0)
    assert np.allclose(W_rt, 1)
    try:
        from reborn.misc.interpolate import trilinear_insertion, trilinear_interpolation

        qcoords_rot = np.dot( rotMats[0].T, qcoords.T).T
        is_inbounds = utils.qs_inbounds(qcoords_rot, W.shape, X_MIN, X_MAX)
        A = np.zeros(W.shape)
        B = np.zeros(W.shape)
        trilinear_insertion(
            A,B,
            vectors=qcoords_rot[is_inbounds],
            insert_vals=vals.ravel()[is_inbounds],
            x_min=X_MIN, x_max=X_MAX)
        A1 = A.copy()
        B1 = B.copy()
        trilinear_insertion(
            A,B,
            vectors=qcoords_rot[is_inbounds],
            insert_vals=vals.ravel()[is_inbounds],
            x_min=X_MIN, x_max=X_MAX)

        assert np.allclose(A1*2, A)
        assert np.allclose(B1*2, B)

        A = utils.errdiv(A,B)

        W_rt = trilinear_interpolation(
            A, qcoords_rot[is_inbounds],
            x_min=X_MIN, x_max=X_MAX)

        assert np.allclose( W_rt, 1)

        W_rt_from_GPUdensity = trilinear_interpolation(
            W2.reshape((n,n,n)).astype(np.float64), qcoords_rot[is_inbounds],
            x_min=X_MIN, x_max=X_MAX)
        assert np.allclose( W_rt_from_GPUdensity, 1)

        L.toggle_insert()
        assert np.allclose(L.densities(), 0)
        assert np.allclose(L.wts(), 0)

        vals1 = np.ones(img_sh)
        vals2 = np.ones(img_sh)*2
        L.copy_image_data(vals1)
        L.copy_image_data(vals2)
        assert np.all(vals1*2==vals2)
    except ImportError:
        pass

    print("OK")


if __name__=="__main__":
    import sys
    highRes = int(sys.argv[1])
    _test(highRes)
