
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

from reborn.misc.interpolate import trilinear_interpolation, trilinear_insertion
from simemc import utils, sim_const, sim_utils, const
from simemc.emc import lerpy

# first simulate an image, and insert it into the 3D density

# simulate
np.random.seed(0)
C = sim_utils.random_crystal()
SIM = sim_utils.get_noise_sim(0)
Famp = sim_utils.get_famp()
img = sim_utils.synthesize_cbf(
    SIM, C, Famp,
    dev_id=0,
    xtal_size=0.002, outfile=None, background=0, just_return_img=True)

# nominal qmap
qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
qx, qy, qz = map(lambda x: x.ravel(), qmap)
qcoords = np.vstack((qx, qy, qz)).T

# rotated qmap
Umat = np.reshape(C.get_U(), (3, 3))
qcoords_rot = np.dot(Umat.T, qcoords.T).T

# insert the image into the 3d density
W = utils.insert_slice(img.ravel(), qcoords_rot, const.QBINS)

# lengthscale parameters of the density array
dens_shape = const.DENSITY_SHAPE
corner, deltas = utils.corners_and_deltas(dens_shape, const.X_MIN, const.X_MAX)
qs = corner[0] + np.arange(const.DENSITY_SHAPE[0]) * deltas[0]

# make a nearest neighbor and a linear interpolator using scipy methods
rgi_line = RegularGridInterpolator((qs, qs, qs), W, method='linear', fill_value=0,bounds_error=False)
#rgi_near = RegularGridInterpolator((qs, qs, qs), W, method='nearest', fill_value=0,bounds_error=False)

Wr_line = rgi_line(qcoords_rot).reshape(img.shape)
Wr_reborn = trilinear_interpolation(np.ascontiguousarray(W), np.ascontiguousarray(qcoords_rot),
                                  x_min=const.X_MIN,
                                  x_max=const.X_MAX).reshape(img.shape)

# now try with the CUDA interpolator
L = lerpy()
L.allocate_lerpy(0,
                 np.array([np.reshape(C.get_U(),(3,3))]),
                 W, len(qcoords),
                 tuple(corner), tuple(deltas), qcoords,
                 1, len(qcoords))
Wr_simemc = L.trilinear_interpolation(0).reshape(img.shape)

inbounds = utils.qs_inbounds(qcoords, const.DENSITY_SHAPE, const.X_MIN, const.X_MAX).reshape(img.shape)
assert pearsonr(Wr_reborn[inbounds], Wr_simemc[inbounds])[0] >.999
assert pearsonr(Wr_line[inbounds], Wr_simemc[inbounds])[0] >.999

print("OK!")
