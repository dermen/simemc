
from scipy.spatial.transform import Rotation
import numpy as np
import pytest


from simemc import utils
from simemc.dragon_geom import DET, BEAM
from simemc.emc import lerpy


@pytest.mark.mpi_skip()
def test():
    main()

def main():
    dens_dim =151
    max_q =0.25
    L = lerpy()
    L.dens_dim =dens_dim
    L.max_q =max_q
    dev_id = 0
    rotMats = Rotation.random(100, random_state=0).as_matrix()

    X_MIN = L.xmin
    X_MAX = L.xmax
    c ,d = utils.corners_and_deltas(L.dens_sh, X_MIN, X_MAX)

    fdim ,sdim = DET[0].get_image_size()
    img_sh = len(DET), sdim, fdim
    npix = int(np.product(img_sh))

    qx ,qy ,qz = map(lambda x: x.ravel(), utils.calc_qmap(DET, BEAM))
    qcoords = np.vstack((qx, qy, qz)).T

    reparam_dens = np.random.random(L.dens_dim**3)
    peak_mask = np.random.randint(0, 2, reparam_dens.shape).astype(bool)

    L.allocate_lerpy(
        dev_id, rotMats.ravel(), npix,
        c, d, qcoords.ravel(),
        rotMats.shape[0], npix, peak_mask=peak_mask)

    reparam_dens = reparam_dens[peak_mask]

    dens = np.sqrt(reparam_dens**2+1)-1
    L.update_reparameterized_density(reparam_dens)
    assert np.allclose(L.densities(), dens)

    finite_rot_inds = np.array(list(range(3))).astype(np.int32)
    finite_P_dr = np.array([1 / 3.] * 3).astype(L.array_type)
    shot_mask = np.ones(npix, bool)
    nshots = 2
    shots = np.random.random((nshots, npix)) * 10
    shot_backgrounds = np.random.random((nshots, npix)) * 2

    L.reset_density_derivs()
    for i_shot in range(nshots):
        L.copy_image_data(shots[i_shot], shot_mask, shot_backgrounds[i_shot])
        L.dens_deriv(finite_rot_inds, finite_P_dr, verbose=False, shot_scale_factor=1,
                     reset_derivs=False, return_grad=False)
    dens_deriv = L.densities_gradient()
    dens_deriv_reparam = - dens_deriv * (dens_deriv / np.sqrt(dens_deriv**2 + 1))
    dens_deriv_reparam2 = L.reparameterized_densities_gradient()

    assert np.allclose(dens_deriv_reparam, dens_deriv_reparam2)
    print("OK")


if __name__=="__main__":
    main()