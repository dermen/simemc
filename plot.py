# coding: utf-8
from pylab import *
from simemc import utils

import sys
import h5py
infile = sys.argv[1]
sym=False
symbol="P43212"

try:
    data = np.load(infile)['result']
except ValueError:
    data = h5py.File(infile, "r")["Wprime"][()]

    
if len(data.shape)==1:
    ndim =round(np.power(data.shape[0], 1/3.))
    ndim = int(ndim)
    data = np.reshape(data, (ndim,ndim,ndim))

ave_ucell=79.1,79.1,38.4,90,90,90
ave_ucell = 68.48, 68.48, 104.38, 90,90,90
#ave_ucell=40.36, 180.74, 142.8, 90, 90, 90
dens_dim=256
max_q=0.25
#dens_dim=330
#max_q=0.33

if sym:
    data = utils.symmetrize(data.ravel(), dens_dim, max_q, symbol)
        
_, relp_mask = utils.whole_punch_W(data,dens_dim, max_q, 1, ucell_p=ave_ucell)

vox_res = utils.voxel_resolution(dens_dim, max_q)
highRes_limit = 1/max_q
mask = vox_res >= highRes_limit
mask = mask*relp_mask

data*= mask

#m = data[data > 0].mean()
from simtbx.diffBragg.utils import is_outlier
vals = data[mask].ravel()
#m = vals[~is_outlier(vals, 30)].mean()
#s = vals[~is_outlier(vals,30)].std()
m = vals.mean()
#print(m,s)
#s = data[data > 0].std()
for i in range(data.shape[0]):
    cla()
    gca().set_title("%s: slice %d / %d" % (infile, i, data.shape[0]))
    
    imshow(data[:,:,i], vmin=0,vmax=m*0.5)
    #imshow(data[:,:,i], vmin=m-s,vmax=m+3*s) #vmax=vals.max()*0.1)
    draw()
    pause(0.1)
    
