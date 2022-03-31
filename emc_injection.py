
import boost_adaptbx.boost.python as bp
import numpy as np

from simemc import emc

from simemc import utils


def add_type_methods(cls):
    """
    :param cls: either lerpy or probable_orients
    """
    @property
    def array_type(self):
        if self.size_of_cudareal==4:
            return np.float32
        else:
            return np.float64
    cls.array_type = array_type

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
@add_type_methods
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
@add_type_methods
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

    def trilinear_insertion(self, rot_idx, vals, mask=None, verbose=False, tomo_wt=1):
        """

        :param tomo_wt:
        :param rot_idx:
        :param vals:
        :param verbose:
        :return:
        """
        if not isinstance(tomo_wt, float):
            tomo_wt = float(tomo_wt)
        vals = self.check_arrays(vals)
        if mask is None:
            mask = np.ones(vals.shape, bool)
        else:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype==bool
            assert mask.shape==vals.shape
        self._trilinear_insertion(int(rot_idx), vals, mask, verbose, tomo_wt)

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


    def copy_image_data(self, pixels, mask=None):
        """
        :param pixels:
        :param mask:
        :return:
        """
        pixels = self.check_arrays(pixels)
        if mask is None:
            mask = np.ones(pixels.shape, dtype=bool)
        else:
            assert mask.dtype==bool
            assert mask.shape == pixels.shape
        self._copy_image_data(pixels, mask)

    def equation_two(self, rot_inds, verbose=True, shot_scale_factor=1, deriv=False):
        """

        :param rot_inds:
        :param verbose:
        :param shot_scale_factor:
        :return:
        """
        if not isinstance(shot_scale_factor, float):
            shot_scale_factor = float(shot_scale_factor)
        rot_inds = self.check_arrays(rot_inds, np.int32)
        self._equation_two(rot_inds, verbose, shot_scale_factor, deriv)
