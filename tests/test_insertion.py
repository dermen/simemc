
from simemc.emc import lerpy
from simemc.sim_geom import DET, BEAM
from simemc import const
from scipy.spatial.transform import Rotation

def test():
    L = lerpy()
    dev_id = 0
    rotMats = Rotation.random(100, random_state=0).as_matrix()
    import numpy as np
    # make identity
    #rotMats[0] = np.eye(3)

    n = const.NBINS
    from simemc import utils
    import numpy as np
    W = np.random.random((n,n,n))
    c,d = utils.corners_and_deltas(W.shape, const.X_MIN_DRAGON, const.X_MAX_DRAGON)

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

    from reborn.misc.interpolate import trilinear_insertion, trilinear_interpolation
    qcoords_rot = np.dot( rotMats[0].T, qcoords.T).T
    is_inbounds = utils.qs_inbounds(qcoords_rot, W.shape, const.X_MIN_DRAGON, const.X_MAX_DRAGON)
    A = np.zeros(W.shape)
    B = np.zeros(W.shape)
    trilinear_insertion(
        A,B,
        vectors=qcoords_rot[is_inbounds],
        insert_vals=vals.ravel()[is_inbounds],
        x_min=const.X_MIN_DRAGON, x_max=const.X_MAX_DRAGON)
    A1 = A.copy()
    B1 = B.copy()
    trilinear_insertion(
        A,B,
        vectors=qcoords_rot[is_inbounds],
        insert_vals=vals.ravel()[is_inbounds],
        x_min=const.X_MIN_DRAGON, x_max=const.X_MAX_DRAGON)

    assert np.allclose(A1*2, A)
    assert np.allclose(B1*2, B)

    A = utils.errdiv(A,B)

    W_rt = trilinear_interpolation(
        A, qcoords_rot[is_inbounds],
        x_min=const.X_MIN_DRAGON, x_max=const.X_MAX_DRAGON)

    assert np.allclose( W_rt, 1)

    W_rt_from_GPUdensity = trilinear_interpolation(
        W2.reshape((n,n,n)).astype(np.float64), qcoords_rot[is_inbounds],
        x_min=const.X_MIN_DRAGON, x_max=const.X_MAX_DRAGON)
    assert np.allclose( W_rt_from_GPUdensity, 1)

    L.toggle_insert()
    assert np.allclose(L.densities(), 0)
    assert np.allclose(L.wts(), 0)

    #probs = 0, 0.50, 0.50
    #for i, p in enumerate(probs):
    #    if p > 0:
    #        vals = np.ones(img_sh) * (i+1)*p
    #        L.trilinear_insertion(i,vals)
    #W = utils.errdiv(L.densities(), L.wts())

    #L.toggle_insert()
    #for i, p in enumerate(probs):
    #    vals = np.ones(img_sh) * (i+1)*p
    #    L.trilinear_insertion(i,vals)

    #W2 = utils.errdiv(L.densities(), L.wts())

    vals1 = np.ones(img_sh)
    vals2 = np.ones(img_sh)*2
    L.copy_image_data(vals1)
    L.copy_image_data(vals2)
    assert np.all(vals1*2==vals2)

    print("OK")

if __name__=="__main__":
    test()
