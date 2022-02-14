import numpy as np

# the refinement uses 256 bins whose edges are defined as
NBINS = 256
QBINS = np.linspace(-0.25, 0.25, NBINS+1)
QCENT = (QBINS[:-1] +QBINS[1:])*.5

# these are the for the reborn reciprocal space convention
X_MIN = QCENT[0],QCENT[0], QCENT[0]
X_MAX = QCENT[-1], QCENT[-1], QCENT[-1]

