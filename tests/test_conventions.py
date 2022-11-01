
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr
from dials.array_family import flex
from dxtbx.model import ExperimentList, Experiment
from simtbx.diffBragg import utils as db_utils

from reborn.misc.interpolate import trilinear_insertion
from simemc import utils, sim_utils, sim_const
from simemc.emc import probable_orients, lerpy


@pytest.mark.mpi_skip()
def test_conventions_reborn_insert_highRes():
    print("lk")
    _test_conventions(False, True)

@pytest.mark.mpi_skip()
def test_conventions_reborn_insert_lowRes():
    print("lk")
    _test_conventions(False, False)

@pytest.mark.mpi_skip()
def test_conventions_hist_insert_lowRes():
    print("lk")
    _test_conventions(True, False)

@pytest.mark.mpi_skip()
def test_conventions_hist_insert_highRes():
    print("lk")
    _test_conventions(True, True)


def _test_conventions(use_hist_method=True, highRes=False):
    if highRes:
        dens_dim=512
        max_q=0.5
    else:
        dens_dim=256
        max_q=0.25

    gpu_device = 0
    num_rot_mats = 10000000
    #maxRotInds = 10000
    maxRotInds = 5
    max_num_strong_spots = 1000

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
        xtal_size=0.002, outfile=None, background=0)
    print("IMG MEAN=", img.mean())

    qcoords = np.vstack((qx,qy,qz)).T

    Umat = np.reshape(C.get_U(), (3,3))
    qcoords_rot = np.dot(Umat.T, qcoords.T).T

    maxNumQ = qcoords_rot.shape[0]
    dens_shape = dens_dim, dens_dim, dens_dim
    X_MIN, X_MAX = utils.get_xmin_xmax(max_q, dens_dim) 
    corner,deltas = utils.corners_and_deltas(dens_shape, X_MIN, X_MAX )
    inbounds = utils.qs_inbounds(qcoords_rot, dens_shape, X_MIN, X_MAX)
    if use_hist_method:
        qbins = np.linspace(-max_q, max_q, dens_dim+1)
        W = utils.insert_slice(img.ravel(), qcoords_rot, qbins)
    else:
        densities = np.zeros(dens_shape)
        weights = np.zeros(dens_shape)

        trilinear_insertion(
            densities, weights,
            vectors=np.ascontiguousarray(qcoords_rot[inbounds]),
            insert_vals=img.ravel()[inbounds],  x_min=X_MIN, x_max=X_MAX)

        W = utils.errdiv(densities, weights)

    rotMats = Rotation.random(num_rot_mats).as_matrix()
    rot_idx = 1
    rotMats[rot_idx] = Umat

    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    L.allocate_lerpy(gpu_device, rotMats, maxNumQ,
                     tuple(corner), tuple(deltas), qcoords,
                     maxRotInds, maxNumQ)
    L.update_density(W)
    all_Wr_simemc = []
    all_log_Rdr = []
    for i_rot in range(maxRotInds):

        Wr_simemc = L.trilinear_interpolation(i_rot).reshape(img.shape)
        all_Wr_simemc.append(Wr_simemc)

        assert np.all(Wr_simemc >-1e-6)
        log_Rdr_simemc = (np.log(Wr_simemc+1e-6)*img - Wr_simemc)[inbounds.reshape(img.shape)].sum()
        print("Rot%d, log_Rdr= %f "
              % (i_rot, log_Rdr_simemc,))
        print(Wr_simemc.min(), Wr_simemc.max(), Wr_simemc.mean())
        all_log_Rdr.append(log_Rdr_simemc)

    assert np.argmax(all_log_Rdr)==rot_idx
    print("Copy image data to lerpy")
    L.copy_image_data(img.ravel(), mask=inbounds.ravel())
    inds = np.arange(maxRotInds).astype(np.int32)
    print("Compute equation two for all orientations")
    L.equation_two(inds)
    L.free()
    log_Rdr = L.get_out()
    Pdr = utils.compute_P_dr_from_log_R_dr(log_Rdr)
    # the first rotation matrix is the ground truth:
    assert np.argmax(Pdr) == rot_idx

    # see if the strong spots are enough to determine the most probable orientation
    O = probable_orients()
    O.allocate_orientations(gpu_device, rotMats.ravel(), max_num_strong_spots)

    O.Bmatrix = C.get_B()
    # quickly test the Bmatrix property
    assert np.allclose(O.Bmatrix, C.get_B())

    # load the strong spot reflections
    R = db_utils.refls_from_sims(np.array([img]), sim_const.DETECTOR, sim_const.BEAM)
    R['id'] = flex.int(len(R), 0)
    minPred=3
    hcut=0.02
    El = ExperimentList()
    E = Experiment()
    E.detector= sim_const.DETECTOR
    E.beam = sim_const.BEAM
    El.append(E)
    R.centroid_px_to_mm(El)
    R.map_centroids_to_reciprocal_space(El)
    # resolution of each refl; only use refls within the resolution cutoff
    qvecs = R['rlp'].as_numpy_array()

    prob_rot = O.orient_peaks(qvecs.ravel(), hcut, minPred, True)
    # rotation index (ground truth) should definitely be in the prob_rot list
    assert rot_idx in prob_rot
    print("OK")


if __name__=="__main__":
    use_hist = int(sys.argv[1])
    highRes = int(sys.argv[2])
    _test_conventions(use_hist, highRes)
