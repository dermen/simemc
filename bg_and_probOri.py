from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("outdir", type=str, help="this folder will be the emc input folder")
parser.add_argument("quat", type=str, help="path to the quaternion grid file")
parser.add_argument("input", type=str, help="path to input file created with simemc.utils.save_expt_refl_file")
parser.add_argument("--ndev", type=int, help="number of gpu devices (default=1)", default=1)
args = parser.parse_args()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import h5py
import os
import time

from dials.array_family import flex
from dxtbx.model import ExperimentList
from simtbx.diffBragg.utils import strip_thickness_from_detector, image_data_from_expt

from simemc.sim_const import CRYSTAL
from simemc.compute_radials import RadPros
from simemc import utils
from simemc import mpi_utils
from simemc.mpi_utils import print0
from simemc.emc import probable_orients


hcut = 0.12
min_pred = 4
outdir = args.outdir
quat_file = args.quat
input_file=args.input
num_gpu_dev = args.ndev
max_num_strong_spots = 1000
num_process = None
qmin = 1/40.
qmax = 1/4.

# constants
img_sh = 2527, 2463
numQ = 256
num_radial_bins = 500
gpu_device = COMM.rank % num_gpu_dev

mpi_utils.make_dir(outdir)
outfile = os.path.join(outdir, "emc_input%d.h5" %COMM.rank)

rotMats, rotMatWeights = utils.load_quat_file(quat_file)
expt_names, refl_names = utils.load_expt_refl_file(input_file)
if num_process is not None:
    expt_names = expt_names[:num_process]
    refl_names = refl_names[:num_process]

shot_numbers = np.arange(len(expt_names))
shot_num_rank = np.array_split(np.arange(len(expt_names)), COMM.size)[COMM.rank]

Qx = Qy = Qz = None
detector = beam = None
if COMM.rank==0:
    dummie_expt = ExperimentList.from_file(expt_names[0], False)[0]
    detector = dummie_expt.detector
    detector = strip_thickness_from_detector(detector)
    beam = dummie_expt.beam
    qmap = utils.calc_qmap(detector, beam)
    Qx,Qy,Qz = map(lambda x: x.ravel(), qmap)
detector = COMM.bcast(detector)
beam = COMM.bcast(beam)

Qx = COMM.bcast(Qx)
Qy = COMM.bcast(Qy)
Qz = COMM.bcast(Qz)
Qmag = np.sqrt( Qx**2 + Qy**2 + Qz**2)

qbins = np.linspace( -qmax, qmax, numQ + 1)
sel = np.logical_and(Qmag > qmin, Qmag < qmax)
qXYZ = Qx[sel], Qy[sel], Qz[sel]

print0("Found %d experiment files total, dividing across ranks" % len(expt_names), flush=True)
# TODO: this script assumes a single panel image format, generalizing is trivial, but should be done

# make the probable orientation identifier
O = probable_orients()
O.allocate_orientations(gpu_device, rotMats.ravel(), max_num_strong_spots)

radProMaker = None
correction = None

# NOTE: assume one knows the unit cell:
O.Bmatrix = CRYSTAL.get_B()


with h5py.File(outfile, "w") as OUT:
    num_shots = len(shot_num_rank)  # number of shots to load on this rank
    prob_rot_dset = OUT.create_dataset(
        name="probable_rot_inds", shape=(num_shots,),
        dtype=h5py.vlen_dtype(int))
    bg_dset = OUT.create_dataset(
        name="background", shape=(num_shots,num_radial_bins))
    for i_f, i_shot in enumerate(shot_num_rank):
        expt_f = expt_names[i_shot]
        refl_f = refl_names[i_shot]

        El = ExperimentList.from_file(expt_f, True)
        data = image_data_from_expt(El[0])

        R = flex.reflection_table.from_file(refl_f)
        R.centroid_px_to_mm(El)
        R.map_centroids_to_reciprocal_space(El)

        ##########################
        # Get the background image
        ##########################
        if radProMaker is None:
            print0("Creating radial profile maker!", flush=True)
            # TODO: add support for per-shot wavelength
            refGeom = {"D": detector, "B": beam}
            radProMaker = RadPros(refGeom, numBins=num_radial_bins)
            radProMaker.polarization_correction()
            radProMaker.solidAngle_correction()
            correction = radProMaker.POLAR * radProMaker.OMEGA
            correction /= np.mean(correction)

        t = time.time()
        data *= correction
        radialProfile = radProMaker.makeRadPro(
                data_pixels=data,
                strong_refl=R,
                apply_corrections=False, use_median=True)
        tbg = time.time()-t

        ####################################
        # Get the probable orientations list
        ####################################
        t = time.time()
        qvecs = R['rlp'].as_numpy_array()
        verbose_flag = False #COMM.rank==0
        prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, verbose_flag)
        tori = time.time()-t

        ### Save stuff
        prob_rot_dset[i_f] = prob_rot
        bg_dset[i_f] = radialProfile
        print0("(%d/%d) bkgrnd est. took %.4f sec, prob. ori. est. %.4f sec . %d prob ori from %d strong spots."
               % (i_f+1, num_shots, tbg, tori, len(prob_rot), len(qvecs)), flush=True)

    OUT.create_dataset("background_img_sh", data=radProMaker.img_sh)
    OUT.create_dataset("all_Qbins", data=radProMaker.all_Qbins)
    OUT.create_dataset("polar", data=radProMaker.POLAR)
    OUT.create_dataset("omega", data=radProMaker.OMEGA)
    OUT.create_dataset("correction", data=correction)
    Es = [expt_names[i] for i in shot_num_rank]
    Rs = [refl_names[i] for i in shot_num_rank]
    for dset_name, lst in [("expts", Es), ("refls", Rs)]:
        dset = OUT.create_dataset(dset_name, shape=(num_shots,), dtype=h5py.string_dtype(encoding="utf-8"))
        dset[:] = lst
