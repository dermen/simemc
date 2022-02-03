# coding: utf-8
import numpy as np
from dxtbx.model import ExperimentList
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
import glob
# variables
qmin = 1/40.
qmax = 1/4.

# constants
img_sh = 2527, 2463
numQ  = 256
num_process = np.inf #10000

Qx = Qy = Qz = None
if COMM.rank==0:
    Qx, Qy, Qz = map( lambda x: x.ravel(), np.load("qmap.npy") ) 
Qx = COMM.bcast(Qx)
Qy = COMM.bcast(Qy)
Qz = COMM.bcast(Qz)
Qmag = np.sqrt( Qx**2 + Qy**2 + Qz**2)

qbins = np.linspace( -qmax, qmax, numQ + 1)
sel = np.logical_and(Qmag > qmin, Qmag < qmax)
qXYZ = Qx[sel], Qy[sel], Qz[sel]

def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)

print0("Loading files")
fnames = None
if COMM.rank==0:
    fnames = glob.glob("shots_2um/proc/*integrated.expt")
fnames = COMM.bcast(fnames)
vals = np.zeros( (numQ, numQ, numQ))
counts = np.zeros( (numQ, numQ, numQ))

for i_f, f in enumerate(fnames):
    if i_f > num_process:
        break
    if i_f % COMM.size != COMM.rank:
        continue
    
    print0("Loading shot %d" % i_f, flush=True)
    El = ExperimentList.from_file(f, True)
    Umat = np.reshape(El[0].crystal.get_U(), (3,3))
    data = El[0].imageset.get_raw_data(0)[0].as_numpy_array().ravel()
    qdata = np.vstack(qXYZ)
    qsampX, qsampY, qsampZ = np.dot(Umat.T, qdata)
    qsamples = qsampX, qsampY, qsampZ
    counts += np.histogramdd( qsamples, bins=[qbins, qbins, qbins])[0]
    vals += np.histogramdd( qsamples, bins=[qbins, qbins, qbins], weights=data[sel])[0]

counts = COMM.reduce(counts)
vals = COMM.reduce(vals)
if COMM.rank==0:
    print0("Writing result")
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.nan_to_num(vals / counts)
    np.savez("resultALL-T_2um", result=result, qbins=qbins)
