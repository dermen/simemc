
from simemc import utils, sim_utils, const, sim_const
from simemc.emc import lerpy
from simtbx.diffBragg import utils as db_utils
from scipy.stats import pearsonr
import numpy as np

import pytest

@pytest.mark.mpi_skip()
def test():
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
        xtal_size=0.0025, outfile=None, background=0)

    qcoords = np.vstack((qx,qy,qz)).T
    Umat = np.reshape(C.get_U(), (3,3))

    dens_shape = const.DENSITY_SHAPE
    corner,deltas = utils.corners_and_deltas(dens_shape, const.X_MIN, const.X_MAX)

    rotMats = np.array([Umat])

    maxNumQ = len(qcoords)
    Npix = img.size
    L = lerpy()
    Winit = np.zeros(dens_shape)
    L.allocate_lerpy(gpu_device, rotMats, Winit, maxNumQ,
                     tuple(corner), tuple(deltas), qcoords,
                     maxRotInds, Npix)
    L.toggle_insert()
    L.trilinear_insertion(0, img)

    W = utils.errdiv(L.densities(), L.wts()).reshape(dens_shape)

    hkls,I = utils.integrate_W(W)

    # integrate the image directly and compare, the results should correlate!
    R = db_utils.refls_from_sims(np.array([img]), sim_const.DETECTOR, sim_const.BEAM)

    _=db_utils.refls_to_hkl(R, sim_const.DETECTOR, sim_const.BEAM, C, update_table=True)
    Imap1 = {h:v for h,v in zip(hkls, I)}
    Imap2 = {h:v for h,v in zip(R['miller_index'], R['intensity.sum.value'])}

    hcommon = set(Imap1.keys()).intersection(Imap2.keys())
    vals1 = [Imap1[h] for h in hcommon]
    vals2 = [Imap2[h] for h in hcommon]

    # the integrated intensities should be correlated with the structure factor intensities
    assert pearsonr(vals1, vals2)[0] > 0.9
    print("OK")

if __name__=="__main__":
    test()
