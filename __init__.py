
try:
    # this will fail without the proper environment loaded
    from . import emc
    #from .emc_injection import *
except ImportError:
    pass

import boost_adaptbx.boost.python as bp
import numpy as np


# TODO: make this conditional
@bp.inject_into(emc.lerpy)
class _():

    def allocate_lerpy(self, dev_id, rotMats, densities, maxNumQ, corners, deltas, qvecs, maxNumRotInds, numDataPix):
        """
        :param dev_id:
        :param _rotMats:
        :param densities:
        :param maxNumQ:
        :param corners:
        :param deltas:
        :param qvecs:
        :param maxNumRotInds:
        :param numDataPix:
        :return:
        """
        rotMats = self.check_arrays(rotMats)
        densities = self.check_arrays(densities)
        qvecs = self.check_arrays(qvecs)
        self._allocate_lerpy(dev_id, rotMats, densities, maxNumQ, corners, deltas, qvecs, maxNumRotInds, numDataPix)

    def trilinear_insertion(self, rot_idx, vals, verbose=False):
        vals = self.check_arrays(vals)
        self._trilinear_insertion(rot_idx, vals, verbose)

    def update_density(self, new_dens):
        new_dens = self.check_arrays(new_dens)
        self._update_density(new_dens)

    def check_arrays(self, vals, dt=None):
        """
        :param vals:
        :param dt: optional np.dtype
        :return:
        """
        if len(vals.shape) > 1:
            vals = vals.copy().ravel()
        if dt is None:
            if self.size_of_cudareal == 4:
                if vals.dtype != np.float32:
                    if self.auto_convert_arrays:
                        vals = vals.astype(np.float32)
                    else:
                        raise TypeError("Array elem should be same size as CUDAREAL (float32)")
            elif self.size_of_cudareal==8:
                if vals.dtype != np.float64:
                    if self.auto_convert_arrays:
                        vals = vals.astype(np.float64)
                    else:
                        raise TypeError("Array elem should be same size as CUDAREAL (float64)")
        else:
            if vals.dtype != dt:
                if self.auto_convert_arrays:
                    vals = vals.astype(dt)
                else:
                    raise TypeError("Arral elems have incorrect type, should be %s" % str(dt))

        if not vals.flags.c_contiguous:
            vals = np.ascontiguousarray(vals)
        return vals

    def copy_image_data(self, pixels):
        """
        :param pixels:
        :return:
        """
        pixels = self.check_arrays(pixels)
        self._copy_image_data(pixels)

    def equation_two(self, rot_inds, verbose=True):
        rot_inds = self.check_arrays(rot_inds, np.int32)
        self._equation_two(rot_inds, verbose)
