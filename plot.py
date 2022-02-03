# coding: utf-8
from pylab import *

import sys
infile = sys.argv[1]

data = np.load(infile)['result']
m = data[data > 0].mean()
s = data[data > 0].std()
for i in range(data.shape[0]):
    cla()
    gca().set_title("%s: slice %d / %d" % (infile, i, data.shape[0]))
    imshow(data[:,i,:], vmin=m-s, vmax=m+3*s)
    draw()
    pause(0.1)
    
