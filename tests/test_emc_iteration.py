
import pytest
import numpy as np
import os
from simtbx.diffBragg import utils as db_utils
from simemc import utils
from simemc import sim_const, sim_utils
from simemc import mpi_utils
from simemc.emc import lerpy, probable_orients

import const

print0 = mpi_utils.print0f
printR = mpi_utils.printRf


@pytest.mark.mpi(min_size=1)
def test_emc_iteration(ndev):
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    np.random.seed(COMM.rank)
    quat_file = os.path.join(os.path.dirname(__file__), "../quatgrid/c-quaternion70.bin")
    if not os.path.exists(quat_file):
        raise OSError("Please generate the quaternion file %s  with `./quat -bin 70`" % quat_file)
    rots, wts = utils.load_quat_file(quat_file)

    # simulate 100 shots
    Famp = sim_utils.get_famp()

    SIM = sim_utils.get_noise_sim(0)

    dev_id = COMM.rank % ndev
    this_ranks_imgs = []
    this_ranks_rot_indices = []
    nshot = COMM.size*5
    print0("Simulating %d shots on %d ranks" % (nshot, COMM.size))
    rot_indices = np.random.permutation(rots.shape[0])[:nshot]
    for i_shot in range(nshot):
        if i_shot % COMM.size != COMM.rank:
            continue
        printR("Shot %d / %d on device %d" % (i_shot+1, nshot, dev_id))
        rot_idx = rot_indices[i_shot]

        C = sim_utils.random_crystal()
        Umat = rots[rot_idx]
        C.set_U(Umat.ravel())
        img = sim_utils.synthesize_cbf(
            SIM, C, Famp,
            dev_id=0,
            xtal_size=0.002, outfile=None, background=0, just_return_img=True )

        this_ranks_rot_indices.append(rot_idx)
        this_ranks_imgs.append(np.array([img], np.float32))


    # make an initial density on grid
    Winit = utils.get_W_init()

    rots = rots.astype(np.float32)
    O = probable_orients()
    max_num_strong_spots = 1000
    O.allocate_orientations(dev_id, rots.ravel(), max_num_strong_spots)
    O.Bmatrix = sim_const.CRYSTAL.get_B()

    min_pred=7
    hcut=0.03

    this_ranks_prob_rot =[]
    for rot_idx, img in zip(this_ranks_rot_indices, this_ranks_imgs):
        R = db_utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM)
        qvecs = db_utils.refls_to_q(R, sim_const.DETECTOR, sim_const.BEAM)
        qvecs = qvecs.astype(np.float32)
        prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, False)
        print0("%d probable rots on shot " % len(prob_rot))
        assert rot_idx in prob_rot
        this_ranks_prob_rot.append(prob_rot)
    O.free_device()

    # Now, assemble a starting point
    Wstart = np.zeros(const.DENSITY_SHAPE, np.float32)
    L = lerpy()
    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    corner, deltas = utils.corners_and_deltas(const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
    qcoords = np.vstack([qx,qy,qz]).T
    qcoords = qcoords.astype(np.float32)
    maxRotInds = 20000
    L.allocate_lerpy(dev_id, rots.ravel(), Wstart.ravel(), 2464*2527,
                     corner, deltas, qcoords.ravel(),
                     maxRotInds, 2463*2527)
    L.toggle_insert()
    guess_indices = np.random.choice(rots.shape[0], size=len(this_ranks_rot_indices)).astype(np.int32)
    for rot_idx, img in zip(guess_indices, this_ranks_imgs):
        print0(rot_idx)
        L.trilinear_insertion(rot_idx, img.ravel(), False)

    try:
        mpi_utils.do_emc(L, this_ranks_imgs, this_ranks_prob_rot)
    except NotImplementedError:
        pass

    print0("OK")

if __name__=="__main__":
    import sys
    ndev = sys.argv[1]
    test_emc_iteration(int(ndev))
