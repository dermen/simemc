# TODO generalize for variable wavelength
# NOTES: dxtbx radial average (which wraps xfel's radial average , a C++ extension module)
# could also work, but I noticed the beam center is forced to be an integer in that implementation
# Probably doesnt matter much for this work..

from scipy import ndimage
import numpy as np
from dxtbx.model import ExperimentList
from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.array_family import flex


def dials_find_spots(data_img, trusted_flags, params):
    assert data_img.shape==trusted_flags.shape
    thresh = SpotFinderFactory.configure_threshold(params)
    flex_data = flex.double(np.ascontiguousarray(data_img))
    flex_trusted_flags = flex.bool(np.ascontiguousarray(trusted_flags))
    spotmask = thresh.compute_threshold(flex_data, flex_trusted_flags)
    return spotmask.as_numpy_array()


def strong_spot_mask(refl_tbl, detector):
    """
    Returns a mask the same shape as detector
    :param refl_tbl: dials.array_family.flex.reflection_table
    :param detector: dxtbx detector
    :return:  strong spot mask same shape as detector (multi panel)
    """
    nfast, nslow = detector[0].get_image_size()
    n_panels = len(detector)
    is_strong_pixel = np.zeros((n_panels, nslow, nfast), bool)
    panel_ids = refl_tbl["panel"]
    for i_refl, (x1, x2, y1, y2, _, _) in enumerate(refl_tbl['bbox']):
        i_panel = panel_ids[i_refl]
        bb_ss = slice(y1, y2, 1)
        bb_fs = slice(x1, x2, 1)
        is_strong_pixel[i_panel, bb_ss, bb_fs] = True
    return is_strong_pixel


class RadPros:

    def __init__(self, refGeom, maskFile=None, numBins=500):
        """
        :param refGeom: either a dict with {'D': dxtbx_detector, 'B': dxtbx_beam} or a string that points to an experiment containing a beam and detector
        :param numBins: number of radial bins for the radial profiles
        """

        self.numBins = numBins
        
        # reference geometry!
        if isinstance(refGeom, dict):
            assert set(refGeom.keys()) == set(["D", "B"])
            self.detector = refGeom["D"]
            self.beam = refGeom["B"]
        else:
            assert isinstance(refGeom, str)
            El = ExperimentList.from_file(refGeom)
            self.detector = El[0].detector
            self.beam = El[0].beam
        
        fastDim, slowDim = self.detector[0].get_image_size()
        self.panel_sh = slowDim, fastDim
        self.img_sh = len(self.detector), slowDim, fastDim

        if maskFile is not None:
            mask = np.load(maskFile)
            if len(mask.shape) == 2:
                mask = np.array([mask])
        else:
            mask = np.ones(self.img_sh).astype(bool)
        self.mask = mask

        # not generalized yet for thick detectors
        assert self.detector[0].get_mu() == 0

        self.unit_s0 = self.beam.get_unit_s0()
        self._setupQbins()

        # geom correction containers (panel_id -> 2D numpy.array)
        self.POLAR = np.ones(self.img_sh)
        self.OMEGA = np.ones(self.img_sh)
        self._index = np.arange(1, self.numBins+1)

    def _setupQbins(self):
        Qmags = {}
        self.DIFFRACTED = {}
        self.AIRPATH ={}
        for pid in range(len(self.detector)):
            FAST = np.array(self.detector[pid].get_fast_axis())
            SLOW = np.array(self.detector[pid].get_slow_axis())
            ORIG = np.array(self.detector[pid].get_origin())

            Ypos, Xpos = np.indices(self.panel_sh)
            px = self.detector[pid].get_pixel_size()[0]
            Ypos = Ypos* px
            Xpos = Xpos*px

            SX = ORIG[0] + FAST[0]*Xpos + SLOW[0]*Ypos
            SY = ORIG[1] + FAST[1]*Xpos + SLOW[1]*Ypos
            SZ = ORIG[2] + FAST[2]*Xpos + SLOW[2]*Ypos
            self.AIRPATH[pid] = np.sqrt(SX**2 + SY**2 + SZ**2)   # units of mm

            Snorm = np.sqrt(SX**2 + SY**2 + SZ**2)

            SX /= Snorm
            SY /= Snorm
            SZ /= Snorm

            self.DIFFRACTED[pid] = np.array([SX, SY, SZ])

            QX = (SX - self.unit_s0[0]) / self.beam.get_wavelength()
            QY = (SY - self.unit_s0[1]) / self.beam.get_wavelength()
            QZ = (SZ - self.unit_s0[2]) / self.beam.get_wavelength()
            Qmags[pid] = np.sqrt(QX**2 + QY**2 + QZ**2)

        minQ = min([q.min() for q in Qmags.values()])
        maxQ = max([q.max() for q in Qmags.values()])

        self.bins = np.linspace(minQ-1e-6, maxQ+1e-6, self.numBins+1)

        self.bin_cent = (self.bins[:-1] + self.bins[1:])*.5
        self.all_Qbins = np.zeros(self.img_sh, int)
        for pid in Qmags:
            self.all_Qbins[pid] = np.digitize(Qmags[pid], self.bins)

    def makeRadPro(self, data_pixels=None, data_expt=None, strong_refl=None, strong_params=None,
                apply_corrections=True, use_median=True):
        """
        Create a 1d radial profile of the background pixels in the image
        :param data_pixels: image pixels same shape as detector model 
        :param data_expt: filename of an experiment list containing an imageset
        :param strong_refl: filename of a strong spots reflection table
        :param strong_params:  phil params for dials.spotfinder 
        :param apply_corrections: if True, correct for polarization and solid angle
        :param use_median: compute radial median profile, as opposed to radial mean profile
        :return: radial profile as a numpy array
        """
        if data_expt is not None:
            data_El = ExperimentList.from_file(data_expt)
            iset = data_El[0].imageset
            # load the pixels
            data = iset.get_raw_data(0)
            if not isinstance(data, tuple):
                data = (data,)
            data = np.array([d.as_numpy_array() for d in data])
        else:
            assert data_pixels is not None
            data = data_pixels

        if strong_refl is None:
            assert strong_params is not None
            all_peak_masks = [~dials_find_spots(data[pid], self.mask[pid], strong_params)\
                                for pid in range(len(data))]
        else:
            all_peak_masks = ~strong_spot_mask(strong_refl, self.detector)

        if apply_corrections:
            data /= (self.POLAR*self.OMEGA)
        bin_labels = self.all_Qbins.copy()
        combined_mask = np.logical_and(all_peak_masks, self.mask)
        bin_labels[~combined_mask] = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            if use_median:
                radPro = ndimage.median(data, bin_labels,self._index)
            else:
                radPro = ndimage.mean(data, bin_labels,self._index)
        return radPro

    def expand_radPro(self, radPro):
        """
        After computing a radial profile, expand it into a full, azimuthally isotropic image
        :param radPro: numpy array, 1-D, radial profile returned by self.makeRadPro 
        :returns: a numpy array, same shape as the detector, containing background image
        """
        return self.expand_background_1d_to_2d(radPro, self.img_sh, self.all_Qbins)

    def solidAngle_correction(self):
        """vectorized solid angle correction for every pixel; follows nanoBragg implementation"""
        sq_pixel_size = self.detector[0].get_pixel_size()[0]**2
        for pid in range(len(self.detector)):
            close_distance = self.detector[pid].get_distance()
            airPathCubed = np.power(self.AIRPATH[pid], 3)
            omega_pixel = sq_pixel_size * close_distance / airPathCubed
            self.OMEGA[pid] = omega_pixel

    def polarization_correction(self):
        """
        vectorized polarization correction for each detector panel
        This is the same expression used in simtbx.nanoBragg based off of the Kahn paper
        Sets the .POLAR attribute, container for the polarization correction as a 2D numpy array, one per panel
        """
        incident = np.array(self.unit_s0)
        pol_axis = np.array(self.beam.get_polarization_normal())
        kahn_factor = self.beam.get_polarization_fraction()
        B_in = np.cross(pol_axis, incident)
        E_in = np.cross(incident, B_in)

        for pid in self.DIFFRACTED:
            d = self.DIFFRACTED[pid]
            sx, sy, sz = map( lambda x: x.ravel(), d)
            diffracted = np.vstack((sx, sy, sz))
            cos2theta = np.dot(incident, diffracted)
            cos2theta_sqr = cos2theta * cos2theta;
            sin2theta_sqr = 1 - cos2theta_sqr;

            psi = 0
            if kahn_factor != 0.0:
                kEi = np.dot(diffracted.T, E_in)
                kBi = np.dot(diffracted.T, B_in)
                psi = -np.arctan2(kBi, kEi);
            polar = 0.5 * (1.0 + cos2theta_sqr - kahn_factor * np.cos(2 * psi) * sin2theta_sqr)
            self.POLAR[pid] = np.reshape(polar, d[0].shape)

    @staticmethod
    def expand_background_1d_to_2d(radPro, img_sh, all_Qbins):
        """
        This is a special helper method to convert radial profile data
        into a 2D image. Its assumed `all_Qbins` and `img_sh` are copies of this classes members that
        were originally used to create the radPro
        :param radPro: radial profile created by makeRadPro
        :param img_sh: shape of the 2d image
        :param all_Qbins:
        :return: 2D image of background
        """
        expanded = radPro[all_Qbins-1]
        return expanded
