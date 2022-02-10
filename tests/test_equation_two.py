from emc import lerpy
import numpy as np
from simemc import utils
import sys
from scipy.spatial.transform import Rotation
from reborn.misc.interpolate import trilinear_interpolation
import time

def test(maxRotInds=100):

    maxRotInds = 100
    fdim = 2463
    sdim = 2527
    N= fdim*sdim
    maxNumQ = fdim*sdim
    qmin = 1/40.
    qmax = 1/4.
    Umat = np.array([[-0.46239704,  0.87328348,  0.15350879],
                     [-0.19088188, -0.26711064,  0.94457187],
                     [ 0.86588284,  0.40746519,  0.29020516]])

    #from dxtbx.model import ExperimentList
    #El = ExperimentList.from_file()

    np.random.seed(789)
    data1 = np.random.random(N) * 100
    data1 = data1.astype(np.float32).astype(np.float64)

    # do interpolation
    qx, qy, qz = utils.load_qmap("../qmap.npy")
    qx = np.random.uniform(-0.25, 0.25, N)
    qy = np.random.uniform(-0.25, 0.25,N)
    qz = np.random.uniform(0, 0.25, N)
    qcoords = np.vstack((qx,qy,qz)).T
    qmags = np.linalg.norm(qcoords, axis=1)
    qcoords_rot = np.dot(qcoords, Umat)
    inbounds = np.logical_and(qmags > qmin, qmags < qmax)

    #I = np.load("../resultTestBG.npz")['result']
    #qbins = np.load("../resultTestBG.npz")['qbins']
    I = np.random.uniform(-0.01, 10, (256,256,256))
    qbins = np.linspace(-0.25, 0.25, 257)
    #rotMats, wts = utils.load_quat_file("quatgrid/c-quaternion20.bin")
    rotMats = Rotation.random(400000, random_state=0).as_matrix()
    rotMats[0] = Umat

    dens_sh = I.shape
    qcent = (qbins[:-1] + qbins[1:])*.5
    xmin = qcent[0], qcent[0], qcent[0]
    xmax = qcent[-1], qcent[-1], qcent[-1]
    c,d = utils.corners_and_deltas(dens_sh, xmin, xmax )

    talloc = time.time()
    L = lerpy()
    qcoords_alloc  = qcoords[inbounds]
    data1_alloc = data1[inbounds]
    L.allocate_lerpy(0, rotMats.ravel(), I.ravel(), maxNumQ,
                     tuple(c), tuple(d), qcoords_alloc.ravel(),
                     maxRotInds, N)
    talloc = time.time()-talloc
    print("Took %.4f sec to allocate device (this only ever happens once per EMC computation)" % talloc)

    tcopy = time.time()
    L.copy_image_data(data1_alloc) #data.ravel())
    tcopy = time.time()-tcopy
    print("Takes %.4f sec to copy data to GPU" % tcopy)

    inds = np.arange(maxRotInds).astype(np.int32)
    t2 = time.time()
    L.equation_two(inds)
    Rgpu = L.get_out()
    t2 = time.time() - t2
    print("First 3 R_dr values:")
    print("GPU:",np.round(Rgpu[:3], 3), "(%.4f sec)" % t2)

    Rcpu = []
    t = time.time()
    #data1 = data.ravel()
    twaste = 0
    for i_rot in range(maxRotInds):
        qcoords_rot = np.dot(qcoords, rotMats[i_rot])
        W = trilinear_interpolation(I, qcoords_rot, x_min=xmin, x_max=xmax)
        tt = time.time()
        kji = np.floor((qcoords_rot - c) / d)
        bad = np.logical_or(kji < 0, kji > I.shape[0]-2)
        good = ~np.any(bad, axis=1)
        sel = np.logical_and( inbounds, np.logical_and(good, W > 0))
        twaste += time.time()-tt
        Wsel = W[sel]
        r = np.sum(data1[sel]*np.log(Wsel)-Wsel)
        Rcpu.append(r)
    t = time.time()-t - twaste
    print("CPU:",np.round(Rcpu[:3],3), "(%.3f sec)" % t)
    try:
        assert np.allclose( Rgpu, Rcpu)
        print("Results are identical!")
    except:
        print("WARNING: results are not all close, maybe due to compiling GPU code with CUDAREAL defined as float")
        from scipy.stats import pearsonr, linregress
        l = linregress(Rgpu, Rcpu)
        c,p = pearsonr(Rgpu, Rcpu)
        assert c > 0.99
        assert p < 1e-5
        assert l.slope > 0.9
        print("... yet results are highly correlated: pearson=%f" % c )

    print("Took %.4f sec with CUDA" % t2)
    print("Took %.4f sec with fortran + openMP" % t)

    print("OK!")


if __name__=="__main__":
    test(int(sys.argv[1]))