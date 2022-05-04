
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import time
from scipy.stats import pearsonr, linregress
import pytest

from reborn.misc.interpolate import trilinear_interpolation
from simemc.emc import lerpy
from simemc import utils
from simemc import sim_const


@pytest.mark.mpi_skip()
def test_equation_two():
    main()


@pytest.mark.mpi_skip()
def test_equation_two_scale_deriv():
    main(finite_diff=1)


@pytest.mark.mpi_skip()
def test_equation_two_density_deriv():
    main(finite_diff=2)


def main(maxRotInds=10, finite_diff=0):
    """

    :param maxRotInds: how many random orientations to test
    :param finite_diff: 0,1 or 2
        0- is no finite difference test
        1- test derivative of equation two w.r.t. scale factors
        2- '' '' w.r.t. the density model
    :return:
    """

    if finite_diff is not None:
        assert finite_diff in [0,1,2]
    fdim, sdim = sim_const.DETECTOR[0].get_image_size()
    N= fdim*sdim
    maxNumQ = fdim*sdim
    qmin = 1/40.
    qmax = 1/4.
    Umat = np.array([[-0.46239704,  0.87328348,  0.15350879],
                     [-0.19088188, -0.26711064,  0.94457187],
                     [ 0.86588284,  0.40746519,  0.29020516]])

    np.random.seed(789)
    data1 = np.random.random(N) * 100
    data1 = data1.astype(np.float32).astype(np.float64)

    background = np.random.random(data1.shape)*10

    # do interpolation
    qx = np.random.uniform(-0.25, 0.25, N)
    qy = np.random.uniform(-0.25, 0.25,N)
    qz = np.random.uniform(0, 0.25, N)
    qcoords = np.vstack((qx,qy,qz)).T
    qmags = np.linalg.norm(qcoords, axis=1)
    inbounds = np.logical_and(qmags > qmin, qmags < qmax)

    I = np.random.uniform(-0.01, 10, (256,256,256))
    qbins = np.linspace(-0.25, 0.25, 257)
    rotMats = Rotation.random(400000, random_state=0).as_matrix()
    rotMats[0] = Umat

    dens_sh = I.shape
    qcent = (qbins[:-1] + qbins[1:])*.5
    xmin = qcent[0], qcent[0], qcent[0]
    xmax = qcent[-1], qcent[-1], qcent[-1]
    c,d = utils.corners_and_deltas(dens_sh, xmin, xmax )

    talloc = time.time()
    L = lerpy()
    L.allocate_lerpy(0, rotMats, I, maxNumQ,
                     tuple(c), tuple(d), qcoords,
                     maxRotInds, N)
    talloc = time.time()-talloc
    print("Took %.4f sec to allocate device (this only ever happens once per EMC computation)" % talloc)

    tcopy = time.time()
    L.copy_image_data(data1, mask=inbounds, bg=background)
    tcopy = time.time()-tcopy
    print("Takes %.4f sec to copy data to GPU" % tcopy)

    inds = np.arange(maxRotInds)
    t2 = time.time()
    try:  # quick test of the auto type converter
        L.auto_convert_arrays = False
        L.equation_two(inds)
        raise RuntimeError("Auto type check failed")
    except TypeError:
        L.auto_convert_arrays = True
        L.equation_two(inds, shot_scale_factor=1)
    Rgpu = np.array(L.get_out())
    if finite_diff != 0:
        # in this case we only wish to test the derivative of equation 2 using finite differences
        if finite_diff == 1:
            L.equation_two(inds, verbose=False, shot_scale_factor=1, deriv=1)
            deriv_Rgpu = np.array(L.get_out())
            percs = [2**i * 0.00005 for i in range(8)]
            errors = []
            for perc in percs:
                scale_factor_shifted = 1 + perc
                L.equation_two(inds, verbose=False, shot_scale_factor=scale_factor_shifted)
                Rgpu_shifted = np.array(L.get_out())
                delta_Rgpu = Rgpu_shifted - Rgpu
                fdiff = delta_Rgpu / perc
                error = np.mean(np.abs(fdiff - deriv_Rgpu)/deriv_Rgpu)
                print("FINITE DIFF SCALING: ", perc, error)
                errors.append(error)
            l = linregress(percs, errors)
            assert l.slope > 0, l.slope
            assert l.rvalue > 0.999, l.rvalue
            print("ok")
            return

        if finite_diff == 2:
            Pdr_vals = utils.compute_P_dr_from_log_R_dr(Rgpu)
            Pdr_vals = np.array(Pdr_vals)
            Qdr = (Pdr_vals*Rgpu).sum()
            deriv_Qdr = L.dens_deriv(inds, Pdr_vals, verbose=True, shot_scale_factor=1)

            # check the gradient at 10 random voxels
            random_voxels = np.random.choice(np.where(deriv_Qdr != 0)[0], size=10)
            percs = [2**i * 0.005 for i in range(8)]
            for random_voxel in random_voxels:
                errors = []
                print("RANDOM VOXEL: %d" % random_voxel)
                delta_h_vals = np.array(percs)* (I.ravel()[random_voxel])
                for delta_h in delta_h_vals:
                    density_shifted = I.ravel().copy()
                    density_shifted[random_voxel] += delta_h
                    L.update_density(density_shifted.ravel())
                    L.equation_two(inds, verbose=False, shot_scale_factor=1)
                    Rgpu_shifted = np.array(L.get_out())
                    Pdr_vals_shifted = np.array(utils.compute_P_dr_from_log_R_dr(Rgpu_shifted))

                    Qdr_shifted = np.sum(Pdr_vals_shifted*Rgpu_shifted)

                    delta_Qdr = Qdr_shifted - Qdr
                    fdiff = delta_Qdr / delta_h
                    dd = deriv_Qdr[random_voxel]
                    error = np.abs(fdiff - dd)/dd
                    print("FINITE DIFF SCALING: ", delta_h, error)
                    errors.append(error)
                l = linregress(delta_h_vals, errors)
                assert l.slope > 0, l.slope
                assert l.rvalue > 0.999, l.rvalue
            print("ok")
            return

    t2 = time.time() - t2
    print("First 3 R_dr values:")
    print("GPU:",np.round(Rgpu[:3], 3), "(%.4f sec)" % t2)

    Rcpu = []
    Rcpu2 = []
    t = time.time()
    twaste = 0
    epsilon=1e-6 # this should match whats in the CUDA kernel EMC_equation_two , TODO: make epsilon an attribute of emc_ext ?
    for i_rot in range(maxRotInds):
        qcoords_rot = np.dot(qcoords, rotMats[i_rot])
        W = trilinear_interpolation(I, qcoords_rot, x_min=xmin, x_max=xmax)
        tt = time.time()
        kji = np.floor((qcoords_rot - c) / d)
        bad = np.logical_or(kji < 0, kji > I.shape[0]-2)
        good = ~np.any(bad, axis=1)
        sel = np.logical_and( inbounds, np.logical_and(good, W > 0))
        twaste += time.time()-tt
        Wsel = W[sel]
        r = np.sum(data1[sel]*np.log(background[sel] + Wsel+epsilon)-Wsel - background[sel])
        Rcpu.append(r)
        Wr = L.trilinear_interpolation(i_rot)
        r2 = np.sum(data1*np.log1p(background + Wr+epsilon) - Wr - background)
        Rcpu2.append(r2)  # TODO whats important about Wcpu2 ?
    t = time.time()-t - twaste
    print("CPU:",np.round(Rcpu[:3],3), "(%.3f sec)" % t)
    try:
        assert np.allclose( Rgpu, Rcpu)
        print("Results are identical!")
    except:
        print("WARNING: results are not all close, maybe due to compiling GPU code with CUDAREAL defined as float")
        l = linregress(Rgpu, Rcpu)
        c,p = pearsonr(Rgpu, Rcpu)
        assert c > 0.99
        assert p < 1e-5
        assert l.slope > 0.9
        print("... yet results are highly correlated: pearson=%f" % c )

    print("Took %.4f sec with CUDA" % t2)
    print("Took %.4f sec with fortran + openMP" % t)

    L.free()
    print("OK!")


if __name__=="__main__":
    max_rot_inds, finite_diff = int(sys.argv[1]), int(sys.argv[2])
    main(max_rot_inds, finite_diff)
