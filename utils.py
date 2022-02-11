import numpy as np
import os
from scipy.spatial.transform import Rotation
from simemc.compute_radials import RadPros
from dials.command_line.stills_process import phil_scope
from dxtbx.model import ExperimentList
from simtbx.diffBragg import utils as db_utils
from libtbx.phil import parse


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

def load_quat_file(quat_bin):
    """
    Load the data file written by Ti-Yen's quaternion grid sampler
    """
    num_quat = np.fromfile(quat_bin, np.int32, 1)[0]
    quat_data = np.fromfile(quat_bin, np.float64, offset=4)
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


def save_qmap(output_file, Det, Beam, return_qmap=False):
    """
    Qmap shape is (3, num_panels, slow_dim, fast_dim)
    Its assumed all pixels are square, and all panels have same slow,fast dims
    :param output_file: name of file to store the qmap can be loaded with method load_qmap
    :param Det: dxtbx Detector model
    :param return_qmap: if true, return the qmap after computing
    :param Beam: dxtbx Beam model
    """
    # TODO generalize for panel thickness
    panel_fdim, panel_sdim = Det[0].get_image_size()
    num_panels = len(Det)
    image_sh = num_panels, panel_sdim, panel_sdim

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

    np.save(output_file, qmap)


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


def save_expt_refl_file(filename, expts, refls):
    with open(filename, "w") as o:
        for expt, refl in zip(expts, refls):
            expt = os.path.abspath(expt)
            refl = os.path.abspath(refl)
            o.write("%s %s\n" % (expt, refl))


def load_expt_refl_file(input_file):
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
