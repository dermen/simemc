
import boost_adaptbx.boost.python as bp
import numpy as np

from simemc import emc

from simemc import utils


def add_check_arrays(cls):
    """
    :param cls: either lerpy or probable_orients
    """
    def check_arrays(self, vals, dt=None):
        """
        :param vals:
        :param dt: optional np.dtype
        :return:
        """
        if len(vals.shape) > 1:
            print("copy  / ravel")
            vals = vals.copy().ravel()
        if dt is None:
            if self.size_of_cudareal == 4:
                if vals.dtype != np.float32:
                    if self.auto_convert_arrays:
                        print("convert type")
                        vals = vals.astype(np.float32)
                    else:
                        raise TypeError("Array elem should be same size as CUDAREAL (float32)")
            elif self.size_of_cudareal==8:
                if vals.dtype != np.float64:
                    if self.auto_convert_arrays:
                        print("convert type")
                        vals = vals.astype(np.float64)
                    else:
                        raise TypeError("Array elem should be same size as CUDAREAL (float64)")
        else:
            if vals.dtype != dt:
                if self.auto_convert_arrays:
                    print("convert type")
                    vals = vals.astype(dt)
                else:
                    raise TypeError("Arral elems have incorrect type, should be %s" % str(dt))

        if not vals.flags.c_contiguous:
            print("make contiguous")
            vals = np.ascontiguousarray(vals)
        return vals
    cls.check_arrays = check_arrays
    return cls


@bp.inject_into(emc.probable_orients)
@add_check_arrays
class _():
    def allocate_orientations(self, dev_id, rotMats, maxNumQ):
        """

        :param dev_id:
        :param rotMats:
        :param maxNumQ:
        :return:
        """
        rotMats = self.check_arrays(rotMats)
        self._allocate_orientations(dev_id, rotMats, maxNumQ)

    def orient_peaks(self, qvecs, hcut, min_within, verbose=False):
        """

        :param qvecs:
        :param hcut:
        :param min_within:
        :param verbose:
        :return:
        """
        qvecs = self.check_arrays(qvecs)
        is_prob = self._orient_peaks(qvecs, hcut, min_within, verbose)
        probable_rot_inds = np.where(is_prob)[0]
        return probable_rot_inds

@bp.inject_into(emc.lerpy)
@add_check_arrays
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
        #qvecs = self.check_arrays(qvecs)
        self.qvecs = self.check_arrays(qvecs)
        self._allocate_lerpy(dev_id, rotMats, densities, maxNumQ, tuple(corners), tuple(deltas), self.qvecs, maxNumRotInds, numDataPix)

    def trilinear_interpolation(self, rot_idx, verbose=False):
        return self._trilinear_interpolation(int(rot_idx), verbose)

    def trilinear_insertion(self, rot_idx, vals, verbose=False):
        """

        :param rot_idx:
        :param vals:
        :param verbose:
        :return:
        """
        vals = self.check_arrays(vals)
        self._trilinear_insertion(int(rot_idx), vals, verbose)

    def update_density(self, new_dens):
        """
        :param new_dens:
        :return:
        """
        new_dens = self.check_arrays(new_dens)
        self._update_density(new_dens)

    def normalize_density(self):
        new_dens = utils.errdiv(self.densities(), self.wts())
        new_dens = self.check_arrays(new_dens)
        self._update_density(new_dens)


    def copy_image_data(self, pixels):
        """
        :param pixels:
        :return:
        """
        pixels = self.check_arrays(pixels)
        self._copy_image_data(pixels)

    def equation_two(self, rot_inds, verbose=True):
        """

        :param rot_inds:
        :param verbose:
        :return:
        """
        rot_inds = self.check_arrays(rot_inds, np.int32)
        self._equation_two(rot_inds, verbose)
