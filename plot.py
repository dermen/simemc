# coding: utf-8
from pylab import *

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
    from simemc import utils
    data = utils.symmetrize(data.ravel())


m = data[data > 0].mean()
s = data[data > 0].std()
for i in range(data.shape[0]):
    cla()
    gca().set_title("%s: slice %d / %d" % (infile, i, data.shape[0]))
    imshow(data[:,:,i], vmin=m-s, vmax=m+s)
    draw()
    pause(0.1)
    
