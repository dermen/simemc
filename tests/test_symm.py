import numpy as np
import time

from simemc import utils
from scipy.stats import pearsonr
import pytest
from pylab import *


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
    Wsym = utils.symmetrize(W_asu, dens_dim, max_q,how=0, uc=uc, symbol=symbol, friedel=friedel)
    t2 = time.time()-t
    t = time.time()
    #W3 = utils.symmetrize(W_asu,dens_dim, max_q,how=1, uc=uc, symbol=symbol, friedel=friedel)
    #assert np.allclose(Wsym, W3)
    t3 = time.time()-t
    print("cuda %.4f sec, cpu=%.4f sec" % (t2,t3))

    # choose a slice

    qbins = np.linspace(-max_q, max_q,dens_dim+1)
    q=0.0244  # peak for lysozyme
    idx = np.searchsorted(qbins,q)-1
    print("Looking at index %d" % idx)


    img_asu = W_asu[idx]
    img_sym = Wsym[idx]
    ysel = (Y<0)[idx]
    xsel = (X<0)[idx]

    img_sym_rot180 = np.rot90(img_sym*(ysel*xsel),k=2)
    cc = pearsonr(img_asu.ravel(), img_sym_rot180.ravel())[0]
    Y= slice(195,215,1)
    X = slice(125,165,1)
    #subplot(211)
    #imshow(img_sym_rot180[Y,X], vmax=5e-27, vmin=0)
    #subplot(212)
    #imshow(img_asu[Y,X], vmax=5e-27, vmin=0)
    #show()
    assert cc >0.9

    print("OK")


if __name__=="__main__":
    import sys
    switch = int(sys.argv[1])
    if switch:
        test_highRes()
    else:
        test_lowRes()

