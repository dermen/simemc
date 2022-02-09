from trilerp import lerpy
import numpy as np
from simemc import utils

from reborn.misc.interpolate import trilinear_interpolation
import time

fdim = 2463
sdim = 2527
maxNumQ = fdim*sdim 
qmin = 1/40.
qmax = 1/4.
Umat = np.array([[-0.46239704,  0.87328348,  0.15350879],
                 [-0.19088188, -0.26711064,  0.94457187],
                 [ 0.86588284,  0.40746519,  0.29020516]])

#from dxtbx.model import ExperimentList
#El = ExperimentList.from_file()

np.random.seed(789)
data = np.random.random((sdim, fdim)) * 100

# do interpolation
qx, qy, qz = utils.load_qmap("../qmap.npy")
qcoords = np.vstack((qx,qy,qz)).T
qmags = np.linalg.norm(qcoords, axis=1)
qcoords_rot = np.dot(qcoords, Umat)
inbounds = np.logical_and(qmags > qmin, qmags < qmax)

I = np.load("../resultTestBG.npz")['result']
qbins = np.load("../resultTestBG.npz")['qbins']
rotMats, wts = utils.load_quat_file("quatgrid/c-quaternion20.bin")
rotMats[0] = Umat

dens_sh = I.shape
qcent = (qbins[:-1] + qbins[1:])*.5
xmin = qcent[0], qcent[0], qcent[0]
xmax = qcent[-1], qcent[-1], qcent[-1]
c,d = utils.corners_and_deltas(dens_sh, xmin, xmax )
numDataPix = maxNumQ

maxRotInds = 100
talloc = time.time()
L = lerpy()
L.allocate_lerpy(0, rotMats.ravel(), I.ravel(), maxNumQ, tuple(c), tuple(d), qcoords[inbounds].ravel(), maxRotInds, numDataPix)
talloc = time.time()-talloc
print("Took %.4f sec to allocate device (this only ever happens once per EMC computation)" % talloc)

tcopy = time.time()
L.copy_image_data(data.ravel())
tcopy = time.time()-tcopy
print("Takes %.4f sec to copy data to GPU" % tcopy)

inds = np.arange(maxRotInds).astype(np.int32)
print(inds)
t2 = time.time()
L.equation_two(inds)
Rgpu = L.get_out()
t2 = time.time() - t2
print("First 3 R_dr values:")
print("GPU:",np.round(Rgpu[:3], 3), "(%.4f sec)" % t2)

Rcpu = []
t = time.time()
data1 = data.ravel()
for i_rot in range(maxRotInds):
    qcoords_rot = np.dot(qcoords, rotMats[i_rot])
    W = trilinear_interpolation(I, qcoords_rot, x_min=xmin, x_max=xmax)
    sel = np.logical_and(inbounds, W>0)
    Wsel = W[sel]
    r = np.sum( np.log(Wsel)*data1[sel] - Wsel )#data1[sel]*np.log(Wsel) - Wsel)
    Rcpu.append(r)
t = time.time()-t

print("CPU:",np.round(Rcpu[:3],3), "(%.3f sec)" % t)

print("The values might not be equal, but they should be close and highly correlated!")
from scipy.stats import pearsonr, linregress
l = linregress(Rgpu, Rcpu)
c,p = pearsonr(Rgpu, Rcpu)
assert c > 0.99
assert p < 1e-5
assert l.slope > 0.9

print("Took %.4f sec with CUDA" % t2)
print("Took %.4f sec with fortran + openMP" % t)

print("OK!")
