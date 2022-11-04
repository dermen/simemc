
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

from reborn.misc.interpolate import trilinear_interpolation, trilinear_insertion
from simemc import utils, sim_const, sim_utils
from simemc.emc import lerpy
import pytest

# first simulate an image, and insert it into the 3D density

@pytest.mark.mpi_skip()
def test():
    _test()

@pytest.mark.mpi_skip()
def test_sparse():
    _test(sparse=True)

@pytest.mark.mpi_skip()
def test_highRes():
    _test(True)

def _test(highRes=False, sparse=False):

    # simulate
    np.random.seed(0)
    C = sim_utils.random_crystal()
    SIM = sim_utils.get_noise_sim(0)
    Famp = sim_utils.get_famp()
    img = sim_utils.synthesize_cbf(
        SIM, C, Famp,
        dev_id=0,
        xtal_size=0.002, outfile=None, background=0)

    # nominal qmap
    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx, qy, qz = map(lambda x: x.ravel(), qmap)
    qcoords = np.vstack((qx, qy, qz)).T

    # rotated qmap
    Umat = np.reshape(C.get_U(), (3, 3))
    qcoords_rot = np.dot(Umat.T, qcoords.T).T

    # insert the image into the 3d density
    if highRes:
        max_q =0.5
        dens_dim=512
    else:
        max_q=0.25
        dens_dim=256
    qbins = np.linspace(-max_q, max_q, dens_dim+1)
    W = utils.insert_slice(img.ravel(), qcoords_rot, qbins)

    peak_mask = utils.whole_punch_W(dens_dim, max_q)

    if sparse:
        W*= peak_mask

    # lengthscale parameters of the density array
    dens_shape = dens_dim, dens_dim, dens_dim
    X_MIN, X_MAX = utils.get_xmin_xmax(max_q, dens_dim) 
    corner, deltas = utils.corners_and_deltas(dens_shape, X_MIN, X_MAX)
    qs = corner[0] + np.arange(dens_dim) * deltas[0]

    # make a nearest neighbor and a linear interpolator using scipy methods
    rgi_line = RegularGridInterpolator((qs, qs, qs), W, method='linear', fill_value=0,bounds_error=False)
    #rgi_near = RegularGridInterpolator((qs, qs, qs), W, method='nearest', fill_value=0,bounds_error=False)

    Wr_line = rgi_line(qcoords_rot).reshape(img.shape)
    Wr_reborn = trilinear_interpolation(np.ascontiguousarray(W), np.ascontiguousarray(qcoords_rot),
                                      x_min=X_MIN,
                                      x_max=X_MAX).reshape(img.shape)

    # now try with the CUDA interpolator
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    dev_id=0
    L.allocate_lerpy(dev_id,
                     np.array([np.reshape(C.get_U(),(3,3))]),
                     len(qcoords),
                     tuple(corner), tuple(deltas), qcoords,
                     1, len(qcoords),
                     peak_mask=peak_mask if sparse else None)
    if sparse:
        L.update_density(W[peak_mask])
    else:
        L.update_density(W)
    Wr_simemc = L.trilinear_interpolation(0).reshape(img.shape)

    print("mean of scipy slice=%f" % Wr_line.mean())
    print("mean of simemc slice=%f" % Wr_simemc.mean())
    print("mean of reborn slice=%f" % Wr_reborn.mean())

    inbounds = utils.qs_inbounds(qcoords, dens_shape, X_MIN, X_MAX).reshape(img.shape)
    #assert pearsonr(Wr_reborn[inbounds], Wr_simemc[inbounds])[0] >.999
    assert pearsonr(Wr_line[inbounds], Wr_simemc[inbounds])[0] >.999
    assert pearsonr(Wr_reborn[inbounds], Wr_simemc[inbounds])[0] >.999
    L.free()

    print("OK!")


if __name__=="__main__":
    import sys
    highRes, sparse = map(int, sys.argv[1:3])
    _test(highRes, sparse)
