from trilerp import lerpy
import numpy as np
from simemc import utils

from reborn.misc.interpolate import trilinear_interpolation
import time

maxNumQ = 2463*2527
qmin = 1/40.
qmax = 1/4.
Umat = np.array([[-0.46239704,  0.87328348,  0.15350879],
                 [-0.19088188, -0.26711064,  0.94457187],
                 [ 0.86588284,  0.40746519,  0.29020516]])

# do interpolation
qx, qy, qz = utils.load_qmap("../qmap.npy")
qcoords = np.vstack((qx,qy,qz)).T
qmags = np.linalg.norm(qcoords, axis=1)
qcoords_rot = np.dot(qcoords, Umat)


L = lerpy()
I = np.load("../resultTestBG.npz")['result']
qbins = np.load("../resultTestBG.npz")['qbins']
rotMats, wts = utils.load_quat_file("quatgrid/c-quaternion20.bin")
rotMats[0] = Umat.T
dens_sh = I.shape


qcent = (qbins[:-1] + qbins[1:])*.5
xmin = qcent[0], qcent[0], qcent[0]
xmax = qcent[-1], qcent[-1], qcent[-1]
c,d = utils.corners_and_deltas(dens_sh, xmin, xmax )
numDataPix = maxNumQ

L.allocate_lerpy(0, rotMats.ravel(), I.ravel(), maxNumQ, tuple(c), tuple(d), qcoords.ravel(), 1000, numDataPix)


#print("RotMat in Python land:\n",rotMats[1])
#print("\n\nRotMat in C land:")
#L.print_rotMat(1)

inds = np.arange(1000).astype(np.int32)[::-1]
L.trilinear_interpolation(inds)

out = np.array(L.get_out())
img = out.reshape((2527, 2463))

t1 = time.time()
out2 = trilinear_interpolation(I, qcoords_rot, x_min=xmin, x_max=xmax)
img2 = out2.reshape( (2527, 2463))
t2 = time.time()
print("reborn: %f sec" % (t2-t1))

inbounds = np.logical_and(qmags > qmin, qmags < qmax)
assert np.allclose(out[inbounds], out2[inbounds], atol=1e-5)


print("OK!")
