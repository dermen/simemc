import numpy as np
import os
import sympy
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree, distance
from scipy import ndimage as nd
from simemc.compute_radials import RadPros
import time
from dials.command_line.stills_process import phil_scope
from dxtbx.model import ExperimentList
from simtbx.diffBragg import utils as db_utils
from dials.array_family import flex
from dials.algorithms.spot_finding.factory import SpotFinderFactory
from libtbx.phil import parse

from simemc import sim_const
try:
    from simemc.emc import lerpy
    from simemc.emc import probable_orients
except ImportError:
    lerpy = None
    probable_orients = None
from simemc import const
from reborn.misc.interpolate import trilinear_insertion, trilinear_interpolation
from cctbx import sgtbx
from scipy import ndimage as ni
from dials.algorithms.spot_finding.factory import FilterRunner
from dials.model.data import PixelListLabeller, PixelList
from dials.algorithms.spot_finding.finder import pixel_list_to_reflection_table



def voxel_resolution():
    qX,qY,qZ = np.meshgrid(*([const.QCENT]*3), indexing='ij')
    qmag =np.sqrt( qX**2 + qY**2 + qZ**2)
    dspace = 1/qmag
    return dspace 


def whole_punch_W(W, width=1, ucell_p=None):
    """
    set all values far from the Bragg peaks to 0
    :param W: input density
    :param width: increase the width to increase the size of the Bragg peaks
        The unit is arbitrary, 0 would is a single pixel at every Bragg peaks
        1 keeps 17(?) pixels at every Bragg peak
    :return: W with 0s in between the Bragg reflections
    """
    if ucell_p is not None:
        ucell_man = db_utils.manager_from_params(ucell_p)
        Bmat = np.reshape(ucell_man.B_realspace, (3,3))
        a1,a2,a3 = Bmat[:,0], Bmat[:,1], Bmat[:,2]
    else:
        a1,a2,a3 = sim_const.CRYSTAL.get_real_space_vectors()
    V = np.dot(a1, np.cross(a2,a3))
    b1 = np.cross(a2,a3)/V
    b2 = np.cross(a3,a1)/V
    b3 = np.cross(a1,a2)/V

    astar=np.linalg.norm(b1)
    bstar=np.linalg.norm(b2)
    cstar=np.linalg.norm(b3)

    qX,qY,qZ = np.meshgrid(*([const.QCENT]*3), indexing='ij')

    frac_h = qX / astar
    frac_k = qY/ bstar
    frac_l = qZ/ cstar

    H = np.ceil(frac_h-0.5)
    K = np.ceil(frac_k-0.5)
    L = np.ceil(frac_l-0.5)

    hvals = np.arange(-H.max()+1, H.max(),1)
    kvals = np.arange(-K.max()+1, K.max(),1)
    lvals = np.arange(-L.max()+1, L.max(),1)

    avals = hvals*astar
    bvals = kvals*bstar
    cvals = lvals*cstar

    aidx = [np.argmin(np.abs(const.QCENT-a)) for a in avals]
    bidx = [np.argmin(np.abs(const.QCENT-b)) for b in bvals]
    cidx = [np.argmin(np.abs(const.QCENT-c)) for c in cvals]

    Imap = np.zeros((256,256,256),bool)
    A,B,C = np.meshgrid(aidx, bidx, cidx, indexing='ij')
    Imap[A,B,C] = True
    Imap = ni.binary_dilation(Imap.astype(bool), iterations=width)
    Imap = Imap.reshape(W.shape)
    return W*Imap, Imap


def integrate_W(W, ucell_p=None):
    if ucell_p is not None:
        ucell_man = db_utils.manager_from_params(ucell_p)
        Bmat = np.reshape(ucell_man.B_realspace, (3,3))
        a1,a2,a3 = Bmat[:,0], Bmat[:,1], Bmat[:,2]
    else:
        a1,a2,a3 = sim_const.CRYSTAL.get_real_space_vectors()
    V = np.dot(a1, np.cross(a2,a3))
    b1 = np.cross(a2,a3)/V
    b2 = np.cross(a3,a1)/V
    b3 = np.cross(a1,a2)/V

    astar=np.linalg.norm(b1)
    bstar=np.linalg.norm(b2)
    cstar=np.linalg.norm(b3)

    qX,qY,qZ = np.meshgrid(*([const.QCENT]*3), indexing='ij')

    frac_h = qX / astar
    frac_k = qY/ bstar
    frac_l = qZ/ cstar

    H = np.ceil(frac_h-0.5)
    K = np.ceil(frac_k-0.5)
    L = np.ceil(frac_l-0.5)
    
    hvals = np.arange(-H.max()+1, H.max(),1)
    kvals = np.arange(-K.max()+1, K.max(),1)
    lvals = np.arange(-L.max()+1, L.max(),1)

    avals = hvals*astar
    bvals = kvals*bstar
    cvals = lvals*cstar

    aidx = [np.argmin(np.abs(const.QCENT-a)) for a in avals]
    bidx = [np.argmin(np.abs(const.QCENT-b)) for b in bvals]
    cidx = [np.argmin(np.abs(const.QCENT-c)) for c in cvals]

    hvals = hvals.astype(np.int32)
    kvals = kvals.astype(np.int32)
    lvals = lvals.astype(np.int32)

    Ivals = []
    hkl_idx = []
    for a,h in zip(aidx, hvals):
        for b,k in zip(bidx, kvals):
            for c,l in zip(cidx, lvals):
                hkl_idx.append( (h,k,l))
                val = W[a,b,c] + W[a-1,b,c] + W[a+1,b,c] + \
                    W[a,b-1,c] + W[a,b+1,c] + W[a,b,c-1] + W[a,b,c+1]
                Ivals.append(val)

    return hkl_idx, Ivals


def get_W_init(ndom=20, ucell_p=None):
    """
    Get an initial guess of the density
    :param ndom: number of unit cells along crystal a-axis
         Increase this parameter to make the gaussian falloff stronger.
         The other two crystal axes wil have ndom such that the overall
         shape is spherical.
    :return: Density estimate
    """
    if ucell_p is not None:
        ucell_man = db_utils.manager_from_params(ucell_p)
        Bmat = np.reshape(ucell_man.B_realspace, (3,3))
        a1,a2,a3 = Bmat[:,0], Bmat[:,1], Bmat[:,2]
    else:
        a1,a2,a3 = sim_const.CRYSTAL.get_real_space_vectors()
    V = np.dot(a1, np.cross(a2,a3))
    b1 = np.cross(a2,a3)/V
    b2 = np.cross(a3,a1)/V
    b3 = np.cross(a1,a2)/V

    astar=np.linalg.norm(b1)
    bstar=np.linalg.norm(b2)
    cstar=np.linalg.norm(b3)

    qbin_cent = (const.QBINS[1:] + const.QBINS[:-1])*.5
    qX,qY,qZ = np.meshgrid(qbin_cent,qbin_cent,qbin_cent)

    frac_h = qX / astar
    frac_k = qY/ bstar
    frac_l = qZ/ cstar

    H = np.ceil(frac_h-0.5)
    K = np.ceil(frac_k-0.5)
    L = np.ceil(frac_l-0.5)

    na = ndom
    nb = ndom *bstar/astar
    nc = ndom *cstar/astar
    del_h = H-frac_h
    del_k = K-frac_k
    del_l = L-frac_l

    hkl_rad_sq = na**2*del_h**2 + nb**2*del_k**2 + nc**2*del_l**2
    W_init = np.exp(-hkl_rad_sq*2/0.63)

    return W_init


def corners_and_deltas(shape, x_min, x_max):
    """
    :param shape:  shape of the densities array (3dim)
    :param x_min: vector to lower voxel center
    :param x_max: vector to upper voxel center
    :return:  corners and deltas to be passed to tri
    """
    shape = np.array(shape)
    x_min = np.atleast_1d(np.array(x_min))
    x_max = np.atleast_1d(np.array(x_max))
    if len(x_min) == 1:
        x_min = np.squeeze(np.array([x_min, x_min, x_min]))
    if len(x_max) == 1:
        x_max = np.squeeze(np.array([x_max, x_max, x_max]))
    deltas = (x_max - x_min) / (shape - 1)
    corners = x_min
    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)
    return corners, deltas

def load_quat_file(quat_file):
    """
    Load the data file written by Ti-Yen's quaternion grid sampler
    """
    try:
        quat_data = np.loadtxt(quat_file, skiprows=1)
    except UnicodeDecodeError:
        num_quat = np.fromfile(quat_file, np.int32, 1)[0]
        quat_data = np.fromfile(quat_file, np.float64, offset=4)
        quat_data = quat_data.reshape((num_quat, 5))

    # Convert these quats to rotation matrices using scipy
    rotMats = Rotation.from_quat(quat_data[:, :4]).as_matrix()
    weights = quat_data[:,4]

    return rotMats, weights



def load_qmap(qmap_file, as_1d=True):
    """
    :param qmap_file: path to a qmap.npy file created by save_qmap
    :param as_1d: bool, return as 1d arrays if True, else return as 2D arrays (same shape as detector)
    : returns: numpy arrays specifying the rlp of every pixel
    """
    if as_1d:
        Qx, Qy, Qz = map( lambda x: x.ravel(), np.load(qmap_file) ) 
    else:
        Qx, Qy, Qz = np.load(qmap_file)
        
    return Qx, Qy, Qz


def get_data_with_bg_removed(expt, phil_file, radProMaker=None, renorm=None, return_radProMaker=False):
    """
    subtract an azimuthally isotropic background from a detector image

    :param expt: dxtbx experiment object 
    :param phil_file: path to strong spots phil file
    :param radProMaker: instance of radial profile maker, will be created if None
    :param renorm: float, if True, then renormalize the data after correcting for solid angle and polarization. For the simemc data in particular, set this to 100 for sensible units
    :returns: the background subtracted , polarization corrected data, optionally with the RadPros instance
    """

    # make if None
    if radProMaker is None:
        refGeom = {"D": expt.detector, "B": expt.beam}
        radProMaker = RadPros(refGeom)
        radProMaker.polarization_correction()
        radProMaker.solidAngle_correction()


    # load the strong spot phil, for finding spots and removing them 
    # prior to fitting the radial profile
    phil_str = open(phil_file, "r").read()
    user_phil = parse(phil_str)
    phil_sources = [user_phil]
    working_phil, unused = phil_scope.fetch(sources=phil_sources, track_unused_definitions=True)
    params = working_phil.extract()

    # load the data from the experiment
    data = expt.imageset.get_raw_data(0)[0].as_numpy_array()
    data *= (radProMaker.POLAR[0] * radProMaker.OMEGA[0])
    if renorm is not None:
        data /= data.max()
        data *= renorm 

    # estimate the radial profile
    radialProfile = radProMaker.makeRadPro(
            data_pixels=np.array([data]), 
            strong_params=params,
            apply_corrections=False)

    # get the azimuthally symetric background image
    BGdata = radProMaker.expand_radPro(radialProfile)[0]
    # subtract the background
    data = data-BGdata
    if return_radProMaker:
        return data, radProMaker
    else:
        return data 


def insert_slice(K_t, qvecs, qbins):
    """
    :param K_t: ndarray, shape (N,), data pixels (2D slice of 3d volume)
    :param qvecs: ndarray shape (N,3) qvectors
    :param qbins: qbins , shape (M,), bin edges defining 3d reciprocal space
    :return: 3d intensity with K_t inserted as a slice, shape (M,M,M)
    """
    assert qvecs.shape[1]==3
    assert K_t.shape[0] == qvecs.shape[0]
    qsampX, qsampY, qsampZ = qvecs.T
    qsamples = qsampX, qsampY, qsampZ
    counts = np.histogramdd( qsamples, bins=[qbins, qbins, qbins])[0]
    vals = np.histogramdd( qsamples, bins=[qbins, qbins, qbins], weights=K_t)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.nan_to_num(vals / counts)
    return result


def save_qmap(output_file, Det, Beam):
    qmap = calc_qmap(Det, Beam)
    np.save(output_file, qmap)

def calc_qmap(Det, Beam):
    """
    Qmap shape is (3, num_panels, slow_dim, fast_dim)
    Its assumed all pixels are square, and all panels have same slow,fast dims
    :param Det: dxtbx Detector model
    :param return_qmap: if true, return the qmap after computing
    :param Beam: dxtbx Beam model
    """
    # TODO generalize for panel thickness
    panel_fdim, panel_sdim = Det[0].get_image_size()
    num_panels = len(Det)
    image_sh = num_panels, panel_sdim, panel_fdim

    Ki = np.array(Beam.get_unit_s0())
    wave = Beam.get_wavelength()

    qmap = np.zeros( (3,) + (image_sh))
    for pid in range(num_panels):
        F = np.array(Det[pid].get_fast_axis())
        S = np.array(Det[pid].get_slow_axis())
        O = np.array(Det[pid].get_origin())
        J,I = np.indices((panel_sdim, panel_fdim))

        pxsize = Det[pid].get_pixel_size()[0]
        Kx = O[0] + I*F[0]*pxsize + J*S[0]*pxsize
        Ky = O[1] + I*F[1]*pxsize + J*S[1]*pxsize
        Kz = O[2] + I*F[2]*pxsize + J*S[2]*pxsize
        Kmag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        Kx /= Kmag
        Ky /= Kmag
        Kz /= Kmag
        Qx = 1./wave * (Kx-Ki[0])
        Qy = 1./wave * (Ky-Ki[1])
        Qz = 1./wave * (Kz-Ki[2])
        qmap[0,pid,:,:] = Qx
        qmap[1,pid,:,:] = Qy
        qmap[2,pid,:,:] = Qz

    return qmap


def stills_process_params_from_file(phil_file):
    """
    :param phil_file: path to phil file for stills_process
    :return: phil params object
    """
    phil_file = open(phil_file, "r").read()
    user_phil = parse(phil_file)
    phil_sources = [user_phil]
    working_phil, unused = phil_scope.fetch(
        sources=phil_sources, track_unused_definitions=True)
    params = working_phil.extract()
    return params


def save_expt_refl_file(filename, expts, refls, check_exists=False):
    """
    Save an input file for bg_and_probOri (the EMC initializer script)
    expt and refl names will be given absolute paths
    :param filename: input expt_refl name to be written (passable to script bg_and_probOri.py)
    :param expts: list of experiments
    :param refls: list of reflection tables
    :param check_exists: ensure files actually exist
    :return:
    """
    with open(filename, "w") as o:
        for expt, refl in zip(expts, refls):
            expt = os.path.abspath(expt)
            refl = os.path.abspath(refl)
            if check_exists:
                assert os.path.exists(expt)
                assert os.path.exists(refl)
            o.write("%s %s\n" % (expt, refl))


def load_expt_refl_file(input_file):
    """

    :param input_file: file created by method save_expt_refl_file
    :return: two lists, one for expts, one for refls
    """
    expts,refls = [],[]
    lines = open(input_file, "r").readlines()
    for l in lines:
        l = l.strip().split()
        assert(len(l)==2)
        expt,refl = l
        assert os.path.exists(expt)
        assert os.path.exists(refl)
        expts.append(expt)
        refls.append(refl)
    return expts, refls


def load_geom(input_geo, strip_thick=True):
    """
    :param input_geo: experiment list file containing a detector and beam model
    :param strip_thick: if True, return a detector with no panel depth
    :return: (dxtbx detector, dxtbx beam)
    """
    El = ExperimentList.from_file(input_geo, False)
    DETECTOR = El.detectors()[0]
    if strip_thick:
        DETECTOR = db_utils.strip_thickness_from_detector(DETECTOR)

    BEAM = El.beams()[0]
    return DETECTOR, BEAM


class RotInds(dict):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)

    def add_record(self, rot_ind, i_data, rank, P_dr):
        if rot_ind not in self:
            self[rot_ind] = {}
            self[rot_ind]["i_data"] = []
            self[rot_ind]["rank"] = []
            self[rot_ind]["P_dr"] = []
        self[rot_ind]["i_data"].append(i_data)
        self[rot_ind]["rank"].append(rank)
        self[rot_ind]["P_dr"].append(P_dr)

    def iter_record(self, rot_ind):
        rec = self[rot_ind]
        for i_data, rank, P_dr in zip(rec["i_data"], rec["rank"], rec["P_dr"]):
            yield i_data, rank, P_dr

    def merge(self, other):
        for rot_ind in other:
            if rot_ind not in self:
                self[rot_ind] = {}
                self[rot_ind]['i_data'] = []
                self[rot_ind]['rank'] = []
                self[rot_ind]['P_dr'] = []
            self[rot_ind]['i_data'] += other[rot_ind]['i_data']
            self[rot_ind]['rank'] += other[rot_ind]['rank']
            self[rot_ind]['P_dr'] += other[rot_ind]['P_dr']

    @property
    def on_one_rank(self):
        inds = []
        for rot_ind in self:
            ranks = self[rot_ind]['rank']
            if len(set(ranks)) == 1:
                inds.append(rot_ind)
        return set(inds)

    @property
    def on_multiple_ranks(self):
        inds = []
        for rot_ind in self:
            ranks = self[rot_ind]['rank']
            if len(set(ranks)) > 1:
                inds.append(rot_ind)
        return set(inds)

    def tomogram_sends_and_recvs(self):
        """
        Instructions for ranks to send / recv tomograms
        such that W_rt can be computed when components of it exist on different ranks
        Note: this method returns None,None unless rank==0 ; the results should be broadcast to all ranks
            (this is because of the random.choice which is rank-dependent
        """
        multis = self.on_multiple_ranks
        send_to = {}
        recv_from = {}
        req_tag = 0
        for rot_ind in multis:
            ranks = set(self[rot_ind]['rank'])
            tomo_manager = np.random.choice(list(ranks))
            senders = ranks.difference({tomo_manager})

            if tomo_manager not in recv_from:
                recv_from[tomo_manager] = {'rot_inds': [], 'comms_info':[]}
            recv_comms_info = []

            for rank, i_data, P_dr in zip(
                    self[rot_ind]['rank'],
                    self[rot_ind]['i_data'],
                    self[rot_ind]['P_dr']):
                if rank == tomo_manager:
                    continue
                if np.isnan(P_dr):
                    continue
                assert rank in senders
                sender = rank
                if sender not in send_to:
                    send_to[sender] = []
                send_to[sender].append((tomo_manager, i_data, req_tag))
                recv_comms_info.append((sender, P_dr, req_tag))
                req_tag += 1

            if recv_comms_info:
                recv_from[tomo_manager]['rot_inds'].append(rot_ind)
                recv_from[tomo_manager]['comms_info'].append(recv_comms_info)

        return send_to, recv_from


def symmetrize(density, symbol="P43212", dens_sh=(256,256,256),
               reshape=True, how=0, friedel=True):
    """
    :param density: can be 1d or 3d (usually 1d)
    :param symbol: space group lookup symbol
    :param dens_sh: 3d shape of density
    """
    if how==0:
        if lerpy is None:
            raise ImportError("emc extension module failed to load")
    sgi = sgtbx.space_group_info(symbol)
    sg = sgtbx.space_group(sgi.type().hall_symbol())
    O = sg.all_ops()
    sym_rot_mats = []
    sym_xyz = []
    for o in O:
        r = o.r()
        R = np.reshape( r.as_double(), (3,3))
        sym_rot_mats.append(R)
        sym_xyz.append(r.as_xyz())
    sym_rot_mats = np.array(sym_rot_mats)

    qvecs = np.vstack(tuple(map(lambda x: x.ravel(), np.meshgrid(const.QCENT, const.QCENT, const.QCENT) ))).T
    if how==0:
        L = lerpy()
        qvecs = qvecs.astype(L.array_type)
        num_data_pix = maxNumQ = const.NBINS**3
        maxRotInds = len(sym_rot_mats)
        corners, deltas = corners_and_deltas(dens_sh, const.X_MIN, const.X_MAX)
        W = np.zeros(dens_sh, L.array_type)
        dev_id = 0
        L.allocate_lerpy(
            dev_id, sym_rot_mats.astype(L.array_type).ravel(),
            W.ravel(), int(maxNumQ),
            tuple(corners), tuple(deltas), qvecs.ravel(),
            maxRotInds, int(num_data_pix))

        L.toggle_insert()
        for i in range(maxRotInds):
            L.trilinear_insertion(i, density.ravel())

        d = L.densities()
        w = L.wts()
        d = errdiv(d,w)
        if reshape:
            d = d.reshape(dens_sh)
        L.free()
    elif how==1:
        A = np.zeros(dens_sh)
        B = np.zeros(dens_sh)
        for rot in sym_rot_mats:
            qcoords_rot = np.dot( rot.T, qvecs.T).T
            is_inbounds = qs_inbounds(qcoords_rot, dens_sh, const.X_MIN, const.X_MAX)
            trilinear_insertion(
                A,B,
                vectors=np.ascontiguousarray(qcoords_rot[is_inbounds]),
                insert_vals=density.ravel().astype(np.float64)[is_inbounds],
                x_min=const.X_MIN, x_max=const.X_MAX)
        d = errdiv(A,B)
    else:
        raise NotImplementedError("still working out the kinds of index-based symmetry")
        d = np.zeros_like(density)
        for ii, xyz in enumerate(sym_xyz):
            xyz = xyz.split(',')
            #a,b,c = xyz.split(',')

            #s_put = [slice(None) for _ in range(3)]
            swap=False
            if 'y' in xyz[0]:
                swap = True
            invert_map = {'x':2, 'y':1, 'z':0}
            invert = []
            for v in xyz:
                if v.startswith('-'):
                    invert.append(invert_map[v[1]])
            #invert = [invert_map[i_ax] for i_ax in range(3) if xyz[i_ax].startswith('-')]
            #for i,x in enumerate((a,b,c)):
            #    invert = False
            #    if x.startswith('-'):
            #        x = x[1]
            #        invert = True
            #    if invert:
            #        s_put[i] = slice(density.shape[0],None,-1)
            #    else:
            #        s_put[i] = slice(0, density.shape[0], 1)

            print(ii, xyz, invert, swap)
            print("")
            d_term = density.copy()
            if invert:
                d_term = np.flip(d_term, axis=tuple(invert))
            if swap:
                d_term = d_term.swapaxes(2,1)
            d+= d_term
            #if swap:
            #    d = d.swapaxes(2,1)
            #if invert:
            #    d = np.flip(d, axis=tuple(invert))
            #d += density
            #if invert:
            #    d = np.flip(d, axis=tuple(invert))
            #if swap:
            #    d = d.swapaxes(2,1)

        d /= len(sym_xyz)

    if friedel:
        d = 0.5*(d +np.flip(d))

    return d


def errdiv(v1, v2, posinf=0, neginf=0):
    """
    carefully divide v1 by v2. Note posinf,neginf set to 0 because:
        https://stackoverflow.com/q/71667082/2077270
    :param v1:
    :param v2:
    :param posinf:
    :param neginf:
    :return:
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        v3 = np.nan_to_num(v1 / v2, posinf=posinf, neginf=neginf)
    return v3


def qs_inbounds(qcoords, dens_sh, x_min, x_max):
    corner,deltas = corners_and_deltas(dens_sh, x_min, x_max)
    kji = np.floor((qcoords - corner) / deltas)
    bad = np.logical_or(kji < 0, kji > dens_sh[0]-2)
    inbounds = ~np.any(bad, axis=1)
    return inbounds


def dials_find_spots(data_img, params, trusted_flags=None):
    """

    :param data_img: numpy array image
    :param params: instance of stills_process params.spotfinder
    :param trusted_flags:
    :return:
    """
    if trusted_flags is None:
        trusted_flags = np.ones(data_img.shape, bool)
    thresh = SpotFinderFactory.configure_threshold(params)
    flex_data = flex.double(np.ascontiguousarray(data_img))
    flex_trusted_flags = flex.bool(np.ascontiguousarray(trusted_flags))
    spotmask = thresh.compute_threshold(flex_data, flex_trusted_flags)
    spotmask = spotmask.as_numpy_array()
    lab, nlab = nd.label(spotmask)
    npix_per_ref = nd.sum(spotmask, lab, index=list(range(1, nlab+1)))
    minpix = 1
    if isinstance(params.spotfinder.filter.min_spot_size, int):
        minpix = params.spotfinder.filter.min_spot_size
    maxpix = np.inf
    if isinstance(params.spotfinder.filter.max_spot_size, int):
        maxpix = params.spotfinder.filter.max_spot_size
    bad_ref_labels = np.where( np.logical_or(npix_per_ref < minpix, npix_per_ref > maxpix))[0]
    spotmask2 = spotmask.copy()
    for i_lab in bad_ref_labels:
        spotmask[lab==i_lab+1] = False

    return spotmask


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None,
                    max_spot_size=1000, phil_file=None, **kwargs):
    """
    This is for converting the centroids in the noiseless simtbx images
    to a multi panel reflection table
    :param panel_imgs: list or 3D array of detector panel simulations
    :param detector: dxtbx  detector model of a caspad
    :param beam:  dxtxb beam model
    :param thresh: threshol intensity for labeling centroids
    :param filter: optional filter to apply to images before
        labeling threshold, typically one of scipy.ndimage's filters
    :param pids: panel IDS , else assumes panel_imgs is same length as detector
    :param kwargs: kwargs to pass along to the optional filter
    :return: a reflection table of spot centroids
    """
    if panel_ids is None:
        panel_ids = np.arange(len(detector))
    pxlst_labs = []
    for i, pid in enumerate(panel_ids):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if phil_file is not None:
            params = stills_process_params_from_file(phil_file)
            mask = dials_find_spots(img, params)
        elif filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        img_sz = detector[int(pid)].get_image_size()  # for some reason the int cast is necessary in Py3
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append(plab)

    El = db_utils.explist_from_numpyarrays(panel_imgs, detector, beam)
    iset = El.imagesets()[0]
    refls = pixel_list_to_reflection_table(
        iset, pxlst_labs,
        min_spot_size=1,
        max_spot_size=max_spot_size,  # TODO: change this ?
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)[0]
    if phil_file is not None:
        x,y,z = refls['xyzobs.px.value'].parts()
        x -=0.5
        y -=0.5
        refls['xyzobs.px.value'] = flex.vec3_double(x,y,z)

    return refls


def compute_P_dr_from_log_R_dr(log_R_dr, beta=1, min_p=0):
    R_dr = []
    R_dr_sum = sympy.S(0)
    for val in log_R_dr:
        r =  sympy.exp(sympy.S(val)) ** beta
        R_dr.append(r)
        R_dr_sum += r

    P_dr = []
    for r in R_dr:
        p = r / R_dr_sum
        p = float(p)
        P_dr.append(p)

    P_dr = np.array(P_dr)
    y = P_dr.sum()
    P_dr[P_dr < min_p] = 0
    x = errdiv(y, P_dr.sum())
    P_dr *= x
    return P_dr


def deriv_P_dr_from_Q_and_dQ(Q, dQ_dphi):
    dQ = np.array(dQ_dphi)
    P = compute_P_dr_from_log_R_dr(Q)
    sum_P_dQ = np.sum(P*dQ)
    dP = P*(dQ_dphi - sum_P_dQ)
    return dP


def compute_log_R_dr(L, shots, prob_rots, shot_scales, mask=None,bg=None, deriv=0):
    """
    helper function called by ScaleUpdater (TODO: update methods in EMC class to use this method)
    :param L: lerpy instance
    :param shots: list of numpy arrays shots
    :param prob_rots: list of probable orientation index lists
    :param shot_scales: list of per-shot scales
    :param mask: mask, same shape as one of the shots (boolean array, True is masked)
    :param bg: same shape as one of the shots (float array of background pixels)
    :param deriv: 0,1 or 2 (flag to specify if computing R_dr or its derivatives
        0- assume R_dr correponds to the logLikelihood if the image
        1- compute derivative of log_R_dr w.r.t. the shot scale factors
    :return: log_R_dr per shot . If deriv is 1 or 2, then return the respective gradient as well
    """
    assert len(shots) > 0
    if isinstance(deriv, bool):
        print("WARNING!!! make deriv an int (allowed vals: 0,1 or 2)")
    deriv = int(deriv)

    shot_log_R_dr = []
    shot_deriv_logR = []
    nshots = len(shots)
    assert len(prob_rots) == nshots
    if mask is not None:
        assert mask.shape == shots[0].shape
    else:
        mask = np.ones(shots[0].shape, bool)

    if bg is None:
        dummie_bg = np.zeros_like(shots[0])

    for i_shot, (img, rot_inds, scale_factor) in enumerate(zip(shots, prob_rots, shot_scales)):
        if bg is not None:
            bg_img = bg[i_shot]
        else:
            bg_img = dummie_bg
        L.copy_image_data(img.ravel(), mask, bg_img)
        L.equation_two(rot_inds, False, scale_factor)
        log_R_dr_vals = np.array(L.get_out())
        shot_log_R_dr.append(log_R_dr_vals)

        if deriv ==1:
            L.equation_two(rot_inds, False, scale_factor, deriv=deriv)
            deriv_log_R_dr = np.array(L.get_out())
            shot_deriv_logR.append(deriv_log_R_dr)
    if deriv == 0:
        return shot_log_R_dr
    else:
        return shot_log_R_dr, shot_deriv_logR


def signal_level_of_image(R, img):
    """
    :param R: DIALS reflection table for image (strong spots, needs bbox and pid)
    :param img: numpy image array (3-dim), shape should be (numPanels, panelSlowDim,panelFastDim)
    :return: average signal in strong spot on image
    """
    signal_level = 0
    for i in range(len(R)):
        refl = R[i]
        x1,x2,y1,y2,_,_ = refl["bbox"]
        pid = refl['panel']
        signal_level += img[pid, y1:y2, x1:x2].mean()
    signal_level /= len(R)
    return signal_level


def get_prob_rots_per_shot(O, R, hcut, min_pred, detector=None, beam=None):
    if detector is None:
        detector = sim_const.DETECTOR
        exit()
    if beam is None:
        beam = sim_const.BEAM
        exit()
    qvecs = db_utils.refls_to_q(R, detector, beam)
    qvecs = qvecs.astype(O.array_type)
    prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, False)
    return prob_rot


def get_prob_rot(dev_id, list_of_refl_tables, rotation_samples, Bmat_reference=None,
                 max_num_strong_spots=1000, hcut=0.1, min_pred=3, verbose=True,
                detector=None,beam=None):
    if probable_orients is None:
        print("probable_orients failed to import")
        return
    O = probable_orients()
    O.allocate_orientations(dev_id, rotation_samples.ravel(), max_num_strong_spots)
    if Bmat_reference is None:
        O.Bmatrix = sim_const.CRYSTAL.get_B()
        exit()
    else:
        O.Bmatrix = Bmat_reference.elems
    prob_rots_per_shot =[]
    for i_img, R in enumerate(list_of_refl_tables):
        t = time.time()
        prob_rot = get_prob_rots_per_shot(O, R, hcut, min_pred, detector, beam)
        prob_rots_per_shot.append(prob_rot)
        if verbose:
            print("%d probable rots on shot %d / %d with %d strongs (%f sec)"
                   % ( len(prob_rot),i_img+1, len(list_of_refl_tables) , len(R), time.time()-t), flush=True )
    O.free_device()
    return prob_rots_per_shot


def label_strong_reflections(predictions, strong, pix=1, col="xyzobs.px.value"):
    strong_tree = cKDTree(strong[col])
    predicted_tree = cKDTree(predictions[col])
    xyz_obs = [(-1,-1,-1)]*len(predictions)
    xyz_cal = [(-1,-1,-1)]*len(predictions)

    # for each strong refl, find all predictions within q_cutoff of the strong rlp
    pred_idx_candidates = strong_tree.query_ball_tree(predicted_tree,pix)

    #predictions["xyzcal.px"] = predictions['xyzobs.px.value']

    is_strong = flex.bool(len(predictions), False)
    for i_idx, cands in enumerate(pred_idx_candidates):
        if not cands:
            continue
        if len(cands) == 1:
            # if 1 spot is within q_cutoff , then its the closest
            pred_idx = cands[0]
        else:
            # in this case there are multiple predictions near the strong refl, we choose the closest one
            dists = []
            for c in cands:
                d = distance.euclidean(strong_tree.data[i_idx], predicted_tree.data[c])
                dists.append(d)
            pred_idx = cands[np.argmin(dists)]
        is_strong[pred_idx] = True
        xyz_obs[pred_idx] = strong["xyzobs.px.value"][i_idx]
        cal = predictions["xyzobs.px.value"][pred_idx]
        xyz_cal[pred_idx] = cal
    predictions["is_strong"] = is_strong
    predictions["xyzobs.px"] = flex.vec3_double(xyz_obs)
    predictions["xyzcal.px"] = flex.vec3_double(xyz_cal)

