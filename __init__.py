
try:
    # this will fail without the proper environment loaded
    from . import emc
except ImportError:
    pass

import boost_adaptbx.boost.python as bp
import numpy as np


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

    def update_density(self, new_dens):
        new_dens = self.check_arrays(new_dens)
        self._update_density(new_dens)

    def check_arrays(self, vals):
        """
        :param vals:
        :return:
        """
        if len(vals.shape) > 1:
            vals = vals.copy().ravel()
        if self.size_of_cudareal == 4:
            if vals.dtype != np.float32:
                if self.auto_convert_arrays:
                    vals = vals.astype(np.float32)
                else:
                    raise TypeError("Vals should be same size as CUDAREAL (float32)")
        elif self.size_of_cudareal==8:
            if vals.dtype != np.float64:
                if self.auto_convert_arrays:
                    vals = vals.astype(np.float64)
                else:
                    raise TypeError("Vals should be same size as CUDAREAL (float64)")

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
