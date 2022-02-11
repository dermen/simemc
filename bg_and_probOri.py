
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import time

from dials.array_family import flex
from dxtbx.model import ExperimentList


from simemc.sim_const import CRYSTAL
from simemc.compute_radials import RadPros
from simemc import utils
from simemc.mpi_utils import print0
from simemc.emc import probable_orients


qmap_file = "../qmap.npy"
input_file = "../input_1600sim.txt"
num_gpu_dev = 8
max_num_strong_spots = 1000
quat_file = "quatgrid/c-quaternion30.bin"
num_process = np.inf


# variables
qmin = 1/40.
qmax = 1/4.
hcut = 0.05
min_pred = 3

# constants
img_sh = 2527, 2463
numQ  = 256
gpu_device = COMM.rank % num_gpu_dev
RENORM = 1e12

rotMats, rotMatWeights = utils.load_quat_file(quat_file)
expts_names, refl_names = utils.load_expt_refl_file(input_file)

Qx = Qy = Qz = None
if COMM.rank==0:
    Qx, Qy, Qz = utils.load_qmap(qmap_file)
Qx = COMM.bcast(Qx)
Qy = COMM.bcast(Qy)
Qz = COMM.bcast(Qz)
Qmag = np.sqrt( Qx**2 + Qy**2 + Qz**2)

qbins = np.linspace( -qmax, qmax, numQ + 1)
sel = np.logical_and(Qmag > qmin, Qmag < qmax)
qXYZ = Qx[sel], Qy[sel], Qz[sel]

print0("Found %d experiment files" % len(expts_names))
# TODO: this script assumes a single panel image format, generalizing is trivial, but should be done

# make the probable orientation identifier
O = probable_orients()
O.allocate_orientations(gpu_device, rotMats.ravel(), max_num_strong_spots)

radProMaker = None

# NOTE: assume one knows the unit cell:
O.Bmatrix = CRYSTAL.get_B()

for i_f, (expt_f, refl_f) in enumerate(zip(expts_names, refl_names)):
    if i_f > num_process:
        break
    if i_f % COMM.size != COMM.rank:
        continue

    print0("Loading shot %d" % i_f, flush=True)
    El = ExperimentList.from_file(expt_f, True)
    data = El[0].imageset.get_raw_data(0)[0].as_numpy_array()

    R = flex.reflection_table.from_file(refl_f)
    R.centroid_px_to_mm(El)
    R.map_centroids_to_reciprocal_space(El)

    ##########################
    # Get the background image
    ##########################
    if radProMaker is None:
        print0("Creating radial profile maker!")
        # TODO: add support for per-shot wavelength
        refGeom = {"D": El[0].detector, "B": El[0].beam}
        radProMaker = RadPros(refGeom)
        radProMaker.polarization_correction()
        radProMaker.solidAngle_correction()

    t = time.time()
    data *= (radProMaker.POLAR[0] * radProMaker.OMEGA[0])
    data /= data.max()
    data *= RENORM
    radialProfile = radProMaker.makeRadPro(
            data_pixels=np.array([data]),
            strong_refl=R,
            apply_corrections=False)
    BGdata = radProMaker.expand_radPro(radialProfile)[0]
    tbg = time.time()-t

    ####################################
    # Get the probable orientations list
    ####################################
    t = time.time()
    qvecs = R['rlp'].as_numpy_array()
    verbose_flag = COMM.rank==0
    O.orient_peaks(qvecs.ravel(), hcut, min_pred, verbose_flag)
    prob_rot = O.get_probable_orients()
    tori = time.time()-t

    print0("Took %.4f sec for background estimation and %.4f sec for prob. ori. estimation" % (tbg, tori))

