
import dxtbx
import numpy as np
from dxtbx.model import Crystal
from scipy.spatial.transform import Rotation
from dials.array_family import flex
from dxtbx.model import ExperimentList, Experiment

from simemc import utils
from simemc.emc import probable_orients, lerpy
import pytest


@pytest.mark.skip(reason="test currently depends on hard-coded paths")
def test_conventions():
    gpu_device = 0
    num_rot_mats = 10000000
    maxRotInds = 10000
    max_num_strong_spots = 1000

    print("loading qmap and image")
    qx,qy,qz =utils.load_qmap("../qmap.npy")
    #img = loader.get_raw_data().as_numpy_array()
    loader = dxtbx.load("../quick_sim/rank0/shot0.cbf")
    img = loader.get_raw_data().as_numpy_array()
    qcoords = np.vstack((qx,qy,qz)).T
    Amat = np.load("../quick_sim/rank0/shot0.npz")["A"].reshape((3,3))
    real_a, real_b, real_c = map(tuple, np.linalg.inv(Amat) )
    cdict = {"real_space_a": real_a, "real_space_b": real_b, "real_space_c": real_c, "space_group_hall_symbol": "-P 4 2", "__id__": "crystal"}
    C = Crystal.from_dict(cdict)
    Umat = np.reshape(C.get_U(), (3,3))
    qcoords_rot = np.dot( Umat.T, qcoords.T).T
    qbins = np.linspace(-0.25, 0.25, 257)
    print("inserting slice")
    W = utils.insert_slice(img.ravel(), qcoords_rot, qbins)

    maxNumQ = qcoords_rot.shape[0]
    xmin = [qbins[0]] *3
    xmax = [qbins[-1]] *3
    Npix = maxNumQ
    corner,deltas = utils.corners_and_deltas(W.shape, xmin, xmax )

    rotMats = Rotation.random(num_rot_mats, random_state=0).as_matrix()
    rotMats[0] = Umat

    print("Allocate a lerpy")
    L = lerpy()
    L.allocate_lerpy(gpu_device, rotMats.ravel(), W.ravel(), maxNumQ,
                     tuple(corner), tuple(deltas), qcoords.ravel(),
                     maxRotInds, Npix)

    print("Copy image data to lerpy")
    L.copy_image_data(img.ravel())
    inds = np.arange(maxRotInds).astype(np.int32)
    print("Compute equation two for all orientations")
    L.equation_two(inds)
    Rdr = L.get_out()
    # the first rotation matrix is the ground truth:
    assert np.argmax(Rdr)==0

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
    Det = loader.get_detector()
    Beam = loader.get_beam()
    El = ExperimentList()
    E = Experiment()
    E.detector= Det
    E.beam = Beam
    El.append(E)
    R.centroid_px_to_mm(El)
    R.map_centroids_to_reciprocal_space(El)
    # resolution of each refl; only use refls within the resolution cutoff
    qvecs = R['rlp'].as_numpy_array()

    O.orient_peaks(qvecs.ravel(), hcut, minPred, True)
    prob_rot = O.get_probable_orients()
    # rotation index 0 (ground truth) should definitely be in the prob_rot list
    assert 0 in prob_rot
    print("OK")


if __name__=="__main__":
    test_conventions()
