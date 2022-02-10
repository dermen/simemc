import numpy as np
from simemc import emc
import time

def test_orient():
    np.random.seed(12345)
    hcut = 0.3
    minPred = 3
    numRot = 3000000
    METHOD = 1

    rotMats = np.ascontiguousarray(np.random.random((numRot,3,3)))
    qvecs = np.ascontiguousarray(np.random.random((20,3))*10)
    O = emc.probable_orients()
    print("Allocating!")
    t = time.time()
    O.allocate_orientations(0, rotMats.ravel(), 10)
    t = time.time()-t
    print("Allocation took %.5f sec" % t)

    t = time.time()
    rot_inds = []
    O.orient_peaks(qvecs.ravel(), hcut, minPred, True)
    rot_inds = O.get_probable_orients()
    t = time.time()-t
    print("GPU took %.5f sec" % t)

    t = time.time()
    Hf = np.dot(rotMats, qvecs.T)
    Hi = np.ceil(Hf-0.5)
    deltaH = Hf - Hi
    Hnorm = np.linalg.norm(deltaH, axis=1)
    out2 = np.sum(Hnorm < hcut, axis=1) >= minPred
    rot_inds2 = np.where(out2)[0]
    t = time.time()-t
    print("CPU took %.5f sec" % t)
    diff_inds = set(rot_inds).difference(rot_inds2)
    print(diff_inds)
    if diff_inds:
        print("WARNING THERE ARE DIFFERENCES BUT DID YOU COMPILE WITH CUDAREAL defined as float ? ")
        print("If so, then diff_inds should be few..")
        n = len(diff_inds) / len(rot_inds2)
        assert n < 1e-5
    else:
        print("No differences between CPU and GPU code!")

    O.free_device()
    print("OK")

if __name__=="__main__":
    test_orient()
