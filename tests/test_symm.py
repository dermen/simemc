import numpy as np
import time

from simemc import utils
import pytest


@pytest.mark.mpi_skip()
def test_lowRes():
    _test(256,0.25)

@pytest.mark.mpi_skip()
def test_highRes():
    _test(512,0.5)

def _test(dens_dim, max_q):
    #np.random.seed(0)
    dens_dim=256
    max_q=0.25
    W = np.random.random((dens_dim, dens_dim, dens_dim))
    #W = np.zeros((256,256,256))
    x,y,z = 128,128,128
    Z,Y,X = np.indices(W.shape)
    R = np.sqrt((Z-z)**2 + (Y-y)**2 +(X-x)**2)
    PHI = np.arctan2(Y-y,X-x)* 180/np.pi
    THETA = np.nan_to_num(np.arccos(utils.errdiv(Z-z,R))) * 180 / np.pi
    sel1 = np.logical_and(THETA <= 90, THETA >= 0)
    sel2 = np.logical_and(PHI <= 90, PHI >=0 )
    sel3 = R < 128
    W[~(sel3*sel2*sel1)] = 0

    t = time.time()
    W2 = utils.symmetrize(W, dens_dim, max_q,how=0)
    t2 = time.time()-t
    t = time.time()
    W3 = utils.symmetrize(W,dens_dim, max_q,how=1)
    t3 = time.time()-t
    t = time.time()
    #W4 = utils.symmetrize(W,dens_dim, max_q,how=2)
    t4 = time.time()-t
    print("cuda %.4f sec, cpu=%.4f sec, cpu slicing=%.4f sec" % (t2,t3, t4))
    #from IPython import embed;embed();exit()
    assert np.allclose(W2, W3)
    #assert np.allclose(W2, W4)
    print("OK")


if __name__=="__main__":
    import sys
    switch = int(sys.argv[1])
    if switch:
        test_highRes()
    else:
        test_lowRes()

