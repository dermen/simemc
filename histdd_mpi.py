# coding: utf-8
import numpy as np
from dxtbx.model import ExperimentList
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
import glob
from simemc.compute_radials import RadPros
from simemc.mpi_utils import print0
from simemc import utils


# variables
qmin = 1/40.
qmax = 1/4.

subBG = True
radProMaker = None
phil_file = "proc.phil"
num_process = np.inf

# constants
img_sh = 2527, 2463
numQ  = 256

params = utils.stills_process_params_from_file(phil_file)

Qx = Qy = Qz = None
if COMM.rank==0:
    Qx, Qy, Qz = utils.load_qmap("qmap.npy") 
Qx = COMM.bcast(Qx)
Qy = COMM.bcast(Qy)
Qz = COMM.bcast(Qz)
Qmag = np.sqrt( Qx**2 + Qy**2 + Qz**2)

qbins = np.linspace( -qmax, qmax, numQ + 1)
sel = np.logical_and(Qmag > qmin, Qmag < qmax)
qXYZ = Qx[sel], Qy[sel], Qz[sel]

print0("Loading files")
fnames = None
if COMM.rank==0:
    fnames = glob.glob("shots/proc/*integrated.expt")
    #fnames = glob.glob("shots/proc/*integrated.expt")
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
    data = El[0].imageset.get_raw_data(0)[0].as_numpy_array()
    
    if radProMaker is None and subBG:
        print0("Creating radial profile maker!")
        # TODO: add support for per-shot wavelength
        refGeom = {"D": El[0].detector, "B": El[0].beam}
        radProMaker = RadPros(refGeom)
        radProMaker.polarization_correction()
        radProMaker.solidAngle_correction()

    if subBG:
        data *= (radProMaker.POLAR[0] * radProMaker.OMEGA[0])
        data /= data.max()
        data *= 1e2
        radialProfile = radProMaker.makeRadPro(
                data_pixels=np.array([data]), 
                strong_params=params,
                apply_corrections=False)
        BGdata = radProMaker.expand_radPro(radialProfile)[0]
        data = data-BGdata
    data = data.ravel()

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
    np.savez("result.BGsub", result=result, qbins=qbins)
