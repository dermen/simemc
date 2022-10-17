import lmfit
import numpy as np
from scipy import ndimage as ni
from joblib import Parallel, delayed

from cctbx import miller, crystal
from dials.array_family import flex

from simemc import utils


def _make_gauss(P, shape):
    """

    :param P:  lmfit parameters
    :param shape: shape of the peak profile (should be 3 dim, e.g. 5,5,5)
    :return: gaussian, np.ndarray with shape=shape
    """
    Z,Y,X = np.indices(shape)
    Gx = P['amp'].value * np.exp(-(X-2)**2/2/P['sigma_x'].value**2)
    Gy = P['amp'].value * np.exp(-(Y-2)**2/2/P['sigma_y'].value**2)
    Gz = P['amp'].value * np.exp(-(Z-2)**2/2/P['sigma_z'].value**2)
    return Gx*Gy*Gz


def _resid(P, peak):
    """

    :param P: lmfit parameters
    :param peak: pealk profile, 3d np.ndarray, shape usually around 5,5,5
    :return:
    """
    gauss = _make_gauss(P, peak.shape)
    resid = gauss-peak
    return resid.ravel()


def fit_peak_to_density(peak):
    """

    :param peak: peak profiles, 3d numpy array, shape ~ 5,5,5
    :return: 3-tuple (fitted amplitude, fit output object, fitted peak(shape=peak.shape))
    """
    assert len(peak.shape)==3
    sigma_x_init = .2*peak.shape[0]
    sigma_y_init = .2*peak.shape[1]
    sigma_z_init = .2*peak.shape[2]
    amp_init = np.percentile(peak,95)
    P = lmfit.Parameters()
    P.add( lmfit.Parameter("amp",amp_init , vary=True, min=1e-10))
    P.add( lmfit.Parameter("sigma_z", sigma_z_init, vary=True, min=1e-5))
    P.add( lmfit.Parameter("sigma_y", sigma_y_init, vary=True, min=1e-5))
    P.add( lmfit.Parameter("sigma_x", sigma_x_init, vary=True, min=1e-5))
    out = lmfit.minimize(_resid, P, args=(peak,))
    return out.params['amp'].value, out, _make_gauss(out.params,peak.shape )


def integrate_W(W, max_q, ucell_p, symbol, method='sum', kernel_iters=2, conn=2, nj=40):
    """

    :param W: 3D density, usually shape ~256,256,256 or more
    :param max_q: maximum q in density
    :param ucell_p: unit cell tuple (ang, ang, ang, deg, deg, deg)
    :param symbol: space group lookup symbol e.g. P43212
    :param method: fit or sum. fit will attempt to fit 3D gaussians to each Bragg reflection
    :param kernel_iters: number of kernel iterations (increase to increase peak profile shape)
    :param conn: connectivity, determines which voxels in the profile are summed (only matters for summation method)
    :param nj: number of jobs
    :return: flex.miller_index object
    """

    if method not in ['sum', 'fit']:
        raise NotImplementedError("no integration method=%s" %method)
    dens_dim = W.shape[0]

    BO = utils.get_BO_matrix(ucell_p, symbol)

    max_h, max_k, max_l = utils.get_hkl_max(max_q, BO)

    hvals = np.arange(-max_h + 1, max_h, 1)
    kvals = np.arange(-max_k + 1, max_k, 1)
    lvals = np.arange(-max_l + 1, max_l, 1)
    hkl_grid = np.meshgrid(hvals, kvals, lvals, indexing='ij')
    hkl_grid = np.array(hkl_grid)
    BOinv = np.round(np.linalg.inv(BO), 8)

    # find the q-values of the whole miller-indices
    qa_vals, qb_vals, qc_vals = np.dot(hkl_grid.T, BOinv.T).T

    QBINS = np.linspace(-max_q, max_q, dens_dim + 1)
    QCENT = (QBINS[:-1] + QBINS[1:]) * .5  # center of each voxel
    qcorner = np.sqrt(3) * max_q
    sel = (np.abs(qa_vals) < qcorner) * (np.abs(qb_vals) < qcorner) * (np.abs(qc_vals) < qcorner)

    all_vals = [qa_vals[sel], qb_vals[sel], qc_vals[sel]]
    all_pos = [np.searchsorted(QCENT, vals) for vals in all_vals]
    selA = (all_pos[0] < dens_dim) * (all_pos[1] < dens_dim) * (all_pos[2] < dens_dim)
    selB = (all_pos[0] > 0) * (all_pos[1] > 0) * (all_pos[2] > 0)
    selAB = selA * selB

    all_inds = []
    for pos, vals in zip(all_pos, all_vals):
        qvals = vals[selAB]
        qpos = pos[selAB]

        left = np.abs(QCENT[qpos - 1] - qvals)
        right = np.abs(QCENT[qpos] - qvals)
        left_or_right_choice = np.argmin(list(zip(left, right)), axis=1)
        inds = [qpos[i] - 1 if choice == 0 else qpos[i] for i, choice in enumerate(left_or_right_choice)]
        all_inds.append(inds)
    aidx, bidx, cidx = all_inds
    all_hvals, all_kvals, all_lvals = map(lambda x: x[sel][selAB], hkl_grid)

    # create a kernel for integrating each peak in the 3-d map
    base_kernel = ni.generate_binary_structure(3, conn)  # 3 dimensional kernel
    kernel = ni.iterate_structure(base_kernel, kernel_iters)  # enlargen kernel to bring in more neighboring voxels
    ksz = int(kernel.shape[0] / 2)  # kernel always has odd dimension

    # iterate over peaks and integrate using the kernel
    def main(jid):
        hkls = []
        data = []
        nfit_fail = 0
        for i_peak,(i1, i2, i3, h, k, l) in enumerate(zip(aidx, bidx, cidx, all_hvals, all_kvals, all_lvals)):
            if i_peak % nj != jid: continue
            i1_slc = slice(i1 - ksz, i1 + ksz + 1, 1)
            i2_slc = slice(i2 - ksz, i2 + ksz + 1, 1)
            i3_slc = slice(i3 - ksz, i3 + ksz + 1, 1)
            peakRegion = W[i1_slc, i2_slc, i3_slc]  # region around one peak, same shape as kernel

            if i_peak % 1000==0:
                print( "integrating %d / %d" %(i_peak+1, len(aidx)))

            if method=='fit':
                integrated_val, fit_out, fit_peak = fit_peak_to_density(peakRegion)
                if not fit_out.success:
                    integrated_val = peakRegion[kernel].sum()
                    nfit_fail+= 1

            else: # method=="sum":
                integrated_val = peakRegion[kernel].sum()

            data.append(integrated_val)
            hkls.append((int(h), int(k), int(l)))
        return hkls,data, nfit_fail

    all_results = Parallel(n_jobs=nj)(delayed(main)(jid) for jid in range(nj))
    hkls = flex.miller_index()
    data = flex.double()
    nfit_fail = 0
    for h, I, n in all_results:
        hkls.extend(flex.miller_index(h))
        data.extend(flex.double(I))
        nfit_fail += n

    p1_ucell = utils.get_p1_ucell(ucell_p, symbol)
    p1_sym = crystal.symmetry(p1_ucell, "P1")
    p1_mset = miller.set(p1_sym, hkls, True)
    # get the operator to gor from this space group defined by `symbol` to P1
    sym = crystal.symmetry(ucell_p, symbol)
    op_to_p1 = sym.change_of_basis_op_to_primitive_setting()
    # apply operator
    mset = p1_mset.change_basis(op_to_p1.inverse())
    mset = miller.set(sym, mset.indices(), True)
    ma = miller.array(mset, data)
    ma = ma.set_observation_type_xray_intensity()
    if method=="fit":
        print("Fit %d / %d peaks (the rest were summed)" % (len(data)-nfit_fail, len(data)))
    return ma
