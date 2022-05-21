# coding: utf-8
from pylab import *
from simemc import utils

import sys
import h5py
infile = sys.argv[1]
if len(sys.argv) > 2:
    sym=True
else:
    sym=False

try:
    data = np.load(infile)['result']
except ValueError:
    data = h5py.File(infile, "r")["Wprime"][()]

    
if len(data.shape)==1:
    ndim =round(np.power(data.shape[0], 1/3.))
    ndim = int(ndim)
    data = np.reshape(data, (ndim,ndim,ndim))

if sym:
    data = utils.symmetrize(data.ravel())

ave_ucell = 68.48, 68.48, 104.38, 90,90,90
        
_, relp_mask = utils.whole_punch_W(data, 1, ucell_p=ave_ucell)

vox_res = utils.voxel_resolution()
highRes_limit = 4.
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
    
    imshow(data[:,:,i], vmin=0,vmax=m*0.5) #vmax=vals.max()*0.1)
    #imshow(data[:,:,i], vmin=m-s,vmax=m+3*s) #vmax=vals.max()*0.1)
    draw()
    pause(0.1)
    
