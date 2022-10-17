
from simemc.emc import lerpy
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr
from simemc import sim_const
from simemc import utils
import numpy as np
import pytest

DET = sim_const.DETECTOR
BEAM= sim_const.BEAM

@pytest.mark.mpi_skip()
def test_friedel():
    _test(1,1)

@pytest.mark.mpi_skip()
def test_friedel_centered():
    _test(1,1,1)

@pytest.mark.mpi_skip()
def test():
    _test(1,0)

@pytest.mark.mpi_skip()
def test_centered():
    _test(1,0,1)

def _test(on_dev=1, friedel=1, centered=0):
    """

    :param on_dev: 1=do symmetrization on device (the efficient way), 0= do symmetrization using old utils.symmetrize method
    :param friedel: 1=test applying friedel symmetry on top of crystal symmetry, 0=do not test
    :param centered: 1=test using a centered space group (C2221), else if 0,  P43212
    :return:
    """
    dens_dim=301
    max_q=0.25
    ucell = 79,79,38,90,90,90
    sym = "P43212"
    if centered:
        ucell = 40,180,142,90,90,90
        sym = "C2221"

    n = dens_dim
    W = np.random.random((n,n,n))


    # setup the lerpy instance
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    X_MIN = L.xmin
    X_MAX = L.xmax
    c,d = utils.corners_and_deltas(W.shape, X_MIN, X_MAX)

    # sensible values for a detector
    fdim,sdim = DET[0].get_image_size()
    img_sh = len(DET), sdim, fdim
    npix = int(np.product(img_sh))
    qx,qy,qz = map(lambda x: x.ravel(), utils.calc_qmap(DET, BEAM))
    qcoords = np.vstack( (qx,qy,qz)).T
    assert len(qcoords) == npix

    # make some dummie rot mats
    rotMats = Rotation.random(100, random_state=0).as_matrix()

    dev_id = 0
    L.allocate_lerpy(
        dev_id, rotMats, npix,
        c,d, qcoords,
        rotMats.shape[0], npix)
    L.update_density(W)

    assert L.max_num_rots == 100  # quick test of the property

    Wtest = L.densities()
    assert np.allclose(Wtest, W.ravel())

    L.toggle_insert()
    L.update_density(Wtest)

    print("integrate asymmetric density")
    # now first, integrate the density W, and then average symmetry pairs
    ma = utils.integrate_W(W, dens_dim, max_q, ucell, sym, kernel_iters=1, conn=1).as_amplitude_array().resolution_filter(d_min=1/max_q)
    if friedel:
        ma = ma.average_bijvoet_mates()  # symmetrize the density
    else:
        ma = ma.merge_equivalents().array() #average_bijvoet_mates()  # symmetrize the density
    if friedel:
        print("Applying friedel symmetry")
    else:
        print("Not applying friedel symmetry")

    if on_dev:
        # apply the actual symmetry operators to the density on the GPU
        # and integrate once again
        print("symmetrize density using LERPY")
        L.set_sym_ops(ucell, sym)
        L.symmetrize()
        if friedel:
            L.apply_friedel_symmetry()
        Wsym = L.densities().reshape(L.dens_sh)
    else:
        print("symmetrize density using utils")
        Wsym = utils.symmetrize(W, dens_dim, max_q, sym, uc=ucell, friedel=friedel)
    print("integrate symmetric density")
    ma2 = utils.integrate_W(Wsym, dens_dim, max_q, ucell, sym, kernel_iters=1, conn=1).as_amplitude_array().resolution_filter(d_min=1/max_q)

    mamap2 = {h: v for h, v in zip(ma2.indices(), ma2.data())}
    mamap = {h: v for h, v in zip(ma.indices(), ma.data())}
    vals = []
    for h in mamap:
        if h not in mamap2:
            continue
        vals.append((mamap[h], mamap2[h]))
    a, b = zip(*vals)
    print("Number of meas=%d" % len(a))
    pear = pearsonr(a,b)[0]
    print("pearson", pear)
    assert pear > 0.99
    print("OK")


if __name__=="__main__":
    import sys
    try:
        on_dev = int(sys.argv[1])
        friedel = int(sys.argv[2])
        centered = int(sys.argv[3])
    except IndexError:
        on_dev = friedel = 1
        centered = 0
    _test(on_dev, friedel, centered)
