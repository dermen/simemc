import numpy as np

# the refinement uses 256 bins whose edges are defined as
NBINS = 256
MAX_Q = 0.25
DENSITY_SHAPE=NBINS,NBINS,NBINS
QBINS = np.linspace(-MAX_Q, MAX_Q, NBINS+1)
QCENT = (QBINS[:-1] +QBINS[1:])*.5

# these are the for the reborn reciprocal space convention
X_MIN = QCENT[0],QCENT[0], QCENT[0]
X_MAX = QCENT[-1], QCENT[-1], QCENT[-1]

QBINS_DRAGON = np.linspace(-0.0289,   0.0289, NBINS+1)
QCENT_DRAGON = (QBINS_DRAGON[:-1] +QBINS_DRAGON[1:])*.5

# these are the for the reborn reciprocal space convention
X_MIN_DRAGON = tuple([QCENT_DRAGON[0]]*3 )
X_MAX_DRAGON = tuple([QCENT_DRAGON[-1]]*3 )

