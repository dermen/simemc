import numpy as np
import time

from simemc import utils
from scipy.stats import pearsonr
import pytest


@pytest.mark.mpi_skip()
def test_lowRes():
    _test(256,0.25)

@pytest.mark.mpi_skip()
def test_highRes():
    _test(512,0.5)

def _test(dens_dim, max_q, friedel=False):
    uc,symbol = utils.ucell_and_symbol(None,None)
    print(symbol)
    np.random.seed(0)
    #W = np.random.random((dens_dim, dens_dim, dens_dim))
    W = utils.get_W_init(dens_dim, max_q, ndom=15)
    cent = dens_dim/2.
    x,y,z = cent,cent,cent
    Z,Y,X = np.indices(W.shape)
    X =X-x
    Y =Y-y
    Z =Z-z
    selX = X > 0
    selY = Y > 0
    selZ = Z > 0
    sel = selX * selY * selZ

    vox_res = utils.voxel_resolution(dens_dim, max_q)
    highRes=1/max_q
    vox_mask = vox_res > highRes
    W*= vox_mask

    W_asu = W.copy()
    W_asu[~sel] = 0

    t = time.time()
    W2 = utils.symmetrize(W_asu, dens_dim, max_q,how=0, uc=uc, symbol=symbol, friedel=friedel)
    t2 = time.time()-t
    t = time.time()
    #W3 = utils.symmetrize(W_asu,dens_dim, max_q,how=1, uc=uc, symbol=symbol, friedel=friedel)
    #assert np.allclose(W2, W3)
    t3 = time.time()-t
    print("cuda %.4f sec, cpu=%.4f sec" % (t2,t3))

    # choose a slice

    idx = 140 if highRes else 281
    img_asu = W_asu[idx]
    img2 = W2[idx]
    ysel = (Y<0)[idx]
    xsel = (X<0)[idx]

    img2_rot180 = np.rot90(img2*(ysel*xsel),k=2)
    cc = pearsonr(img_asu.ravel(), img2_rot180.ravel())[0]
    #from IPython import embed;embed()
    assert cc >0.9

    print("OK")


if __name__=="__main__":
    import sys
    switch = int(sys.argv[1])
    if switch:
        test_highRes()
    else:
        test_lowRes()

