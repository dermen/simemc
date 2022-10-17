
from simemc import utils, sim_utils, const, sim_const
from simemc.emc import lerpy
from simtbx.diffBragg import utils as db_utils
from scipy.stats import pearsonr
import numpy as np

import pytest

@pytest.mark.mpi_skip()
def test_lowRes():
    main(256, 0.25)

@pytest.mark.mpi_skip()
def test_highRes():
    main(512, 0.5)

def main(dens_dim, max_q):
    """
    dens_dim: number of density bins along one edge (cubic density)
    max_q: maximum q at outer corner of voxel
    """
    gpu_device = 0
    maxRotInds = 10000

    print("loading qmap and image")
    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    np.random.seed(0)
    C = sim_utils.random_crystal()
    print(C.get_U())

    SIM = sim_utils.get_noise_sim(0)
    Famp = sim_utils.get_famp()
    img = sim_utils.synthesize_cbf(
        SIM, C, Famp,
        dev_id=0,
        xtal_size=0.1, outfile=None, background=0)

    qcoords = np.vstack((qx,qy,qz)).T
    Umat = np.reshape(C.get_U(), (3,3))

    dens_shape = dens_dim, dens_dim, dens_dim

    rotMats = np.array([Umat])

    maxNumQ = len(qcoords)
    Npix = img.size
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    corner,deltas = utils.corners_and_deltas(dens_shape, L.xmin, L.xmax)
    L.allocate_lerpy(gpu_device, rotMats, maxNumQ,
                     tuple(corner), tuple(deltas), qcoords,
                     maxRotInds, Npix)
    L.toggle_insert()
    L.trilinear_insertion(0, img)

    W = utils.errdiv(L.densities(), L.wts()).reshape(dens_shape)

    ucell, sym = utils.ucell_and_symbol(None, None)
    hkls,I = utils.integrate_W(W, dens_dim, max_q, ucell_p=ucell, symbol=sym)

    # integrate the image directly and compare, the results should correlate!
    qmags = np.linalg.norm(qcoords, axis=1)
    qmask = (qmags < max_q).reshape(img.shape)
    img *= qmask
    thresh = np.median(img[img>0])
    R = db_utils.refls_from_sims(np.array([img]), sim_const.DETECTOR, sim_const.BEAM, thresh=thresh)
    print("Detected %d spots" % len(R))

    sgi = C.get_space_group().info()
    cbo = sgi.change_of_basis_op_to_primitive_setting()
    C = C.change_basis(cbo)

    _=db_utils.refls_to_hkl(R, sim_const.DETECTOR, sim_const.BEAM, C, update_table=True)
    Imap1 = {h:v for h,v in zip(hkls, I)}
    Imap2 = {h:v for h,v in zip(R['miller_index'], R['intensity.sum.value'])}

    hcommon = set(Imap1.keys()).intersection(Imap2.keys())
    vals1 = [Imap1[h] for h in hcommon]
    vals2 = [Imap2[h] for h in hcommon]

    # the integrated intensities should be correlated with the structure factor intensities
    cc = pearsonr(vals1, vals2)[0]
    print("%d integrated refls show CC=%.3f with density" % (len(vals1), cc))
    assert cc > 0.9
    print("OK")

if __name__=="__main__":
    import sys
    switch = int(sys.argv[1])
    if switch:
        test_highRes()
    else:
        test_lowRes()
