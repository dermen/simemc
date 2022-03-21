
import time
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr
from dials.array_family import flex
from dxtbx.model import ExperimentList, Experiment

from reborn.misc.interpolate import trilinear_insertion
from simemc import utils, const, sim_utils, sim_const
from simemc.emc import probable_orients, lerpy


@pytest.mark.skip()
def test_conventions_reborn_insert():
    _test_conventions(False)


@pytest.mark.skip()
def test_conventions_hist_insert():
    _test_conventions(True)


def _test_conventions(use_hist_method=True):
    gpu_device = 0
    num_rot_mats = 10000000
    maxRotInds = 10000
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
        xtal_size=0.002, outfile=None, background=0, just_return_img=True )

    qcoords = np.vstack((qx,qy,qz)).T

    Umat = np.reshape(C.get_U(), (3,3))
    qcoords_rot = np.dot( Umat.T, qcoords.T).T
    qbins = const.QBINS
    maxNumQ = qcoords_rot.shape[0]
    dens_shape = const.DENSITY_SHAPE
    corner,deltas = utils.corners_and_deltas(dens_shape, const.X_MIN, const.X_MAX )
    if use_hist_method:
        W = utils.insert_slice(img.ravel(), qcoords_rot, qbins)
    else:
        good = utils.qs_inbounds(qcoords_rot, dens_shape, const.X_MIN, const.X_MAX)
        densities = np.zeros(dens_shape)
        weights = np.zeros(dens_shape)

        trilinear_insertion(
            densities, weights,
            vectors=np.ascontiguousarray(qcoords_rot[good]),
            insert_vals=img.ravel()[good],  x_min=const.X_MIN, x_max=const.X_MAX)

        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.nan_to_num(densities / weights)

    Npix = maxNumQ

    rotMats = Rotation.random(num_rot_mats).as_matrix()
    rot_idx = 1
    rotMats[rot_idx] = Umat

    L = lerpy()
    L.allocate_lerpy(gpu_device, rotMats, W, maxNumQ,
                     tuple(corner), tuple(deltas), qcoords,
                     maxRotInds, Npix)
    L.toggle_insert()
    L.trilinear_insertion(rot_idx, img)

    with np.errstate(divide='ignore', invalid='ignore'):
        W2 = np.nan_to_num(L.densities()/ L.wts())
    W2 = W2.reshape(W.shape)
    if not use_hist_method:
        # quantitative comparison should only be done if reborn insertion was used to get W
        assert np.sum(W> 0) == np.sum(W2>0)

        if L.size_of_cudareal==8:
            assert np.allclose(W, W2.reshape(W.shape))
        else:
            assert pearsonr(W[W> 0], W2[W2 > 0])[0] > 0.99

    L.update_density(W)


    print("Copy image data to lerpy")
    L.copy_image_data(img.ravel())
    inds = np.arange(maxRotInds).astype(np.int32)
    print("Compute equation two for all orientations")
    L.equation_two(inds)
    Rdr = L.get_out()
    # the first rotation matrix is the ground truth:
    assert np.argmax(Rdr)==rot_idx

    # see if the strong spots are enough to determine the most probable orientation
    O = probable_orients()
    O.allocate_orientations(gpu_device, rotMats.ravel(), max_num_strong_spots)

    O.Bmatrix = C.get_B()
    # quickly test the Bmatrix property
    assert np.allclose(O.Bmatrix, C.get_B())

    # load the strong spot reflections
    R = flex.reflection_table.from_file("../quick_sim/proc/strong_idx-shot0.refl")
    minPred=3
    hcut=0.05
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
    if int(sys.argv[1]):
        test_conventions_hist_insert()
    else:
        test_conventions_reborn_insert()
