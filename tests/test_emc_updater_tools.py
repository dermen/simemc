from mpi4py import MPI
COMM = MPI.COMM_WORLD

from simemc import mpi_utils
from simemc.emc import lerpy
from simemc.dragon_geom import DET, BEAM
from scipy.spatial.transform import Rotation
from simemc import utils
import numpy as np
import pytest


@pytest.mark.mpi(min_size=2)
def test():
    _test()


def _test():
    dens_dim=151
    max_q=0.25
    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    dev_id = 0
    rotMats = Rotation.random(100, random_state=0).as_matrix()

    X_MIN = L.xmin
    X_MAX = L.xmax
    c,d = utils.corners_and_deltas(L.dens_sh, X_MIN, X_MAX)

    fdim,sdim = DET[0].get_image_size()
    img_sh = len(DET), sdim, fdim
    npix = int(np.product(img_sh))

    qx,qy,qz = map(lambda x: x.ravel(), utils.calc_qmap(DET, BEAM))

    qcoords = np.vstack((qx,qy,qz)).T
    assert len(qcoords) == npix
    L.allocate_lerpy(
        dev_id, rotMats.ravel(), npix,
        c,d, qcoords.ravel(),
        rotMats.shape[0], npix)
    dens = np.random.random(L.dens_dim**3)
    relp_mask = np.random.randint(0,2,dens.shape).astype(bool)
    if COMM.rank==0:
        L.copy_relp_mask_to_device(relp_mask)
    L.bcast_relp_mask(COMM)
    L.update_density(dens)
    L.bcast_densities(COMM)  # overrides density on ranks>1

    new_vals = None
    if COMM.rank==0:
        new_vals = np.ones(relp_mask.sum())
    new_vals = COMM.bcast(new_vals)

    L.update_masked_density(new_vals)

    mpi_utils.ensure_same(L.densities())

    finite_rot_inds = np.array(list(range(3))).astype(np.int32)
    finite_P_dr = np.array([1/3.]*3).astype(L.array_type)
    shot_mask = np.ones(npix, bool)
    nshots = 2
    shots = np.random.random((nshots,npix))*10
    shot_backgrounds = np.random.random((nshots,npix))*2

    # pattern 1 for computing gradients (all internal arrays)
    L.reset_density_derivs()
    for i_shot in range(nshots):
        L.copy_image_data(shots[i_shot], shot_mask, shot_backgrounds[i_shot])
        L.dens_deriv(finite_rot_inds, finite_P_dr, verbose=False, shot_scale_factor=1,
                     reset_derivs=False, return_grad=False)
    L.reduce_density_derivs(COMM)
    L.bcast_density_derivs(COMM)
    dens_deriv = L.densities_gradient()  # should be same on all ranks

    # pattern 2 for computing gradients (all internal arrays)
    dens_deriv2 = np.zeros_like(dens_deriv)
    for i_shot in range(nshots):
        L.copy_image_data(shots[i_shot], shot_mask, shot_backgrounds[i_shot])
        dens_deriv2 += L.dens_deriv(finite_rot_inds, finite_P_dr, verbose=False, shot_scale_factor=1,
                     reset_derivs=True, return_grad=True)
    dens_deriv2 = COMM.bcast(COMM.reduce(dens_deriv2))

    assert np.allclose(dens_deriv2, dens_deriv)

    if COMM.rank==0:
        print("Ok")


if __name__=="__main__":
    test()
