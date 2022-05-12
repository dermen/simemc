
import sys
import os
from mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
from dials.array_family import flex

from simemc import mpi_utils, utils
from simemc import const
from simemc.compute_radials import RadPros
from simtbx.diffBragg import utils as db_utils
from dxtbx.model import ExperimentList
from simemc.emc import lerpy


highRes = 4
ndev = 1
input_file = "/data/dermen/dataset_1/small_caxis_exp_ref.txt"
ref_geom_file = "/data/dermen/dataset_1/optimized_geom.expt"
mask_file = "/data/dermen/dataset_1/test_mask.pkl"
quat_file ="/data/dermen/simemc/quatgrid/c-quaternion70.bin"
outdir=sys.argv[1]
ave_ucell = 68.48, 68.48, 104.38, 90,90,90
hcut=0.06
min_pred=3
niter=100
maxN = 300

mpi_utils.make_dir(outdir)

ucell_man = db_utils.manager_from_params(ave_ucell)
Brecip = ucell_man.B_recipspace

GeoExpt = ExperimentList.from_file(ref_geom_file,False)
BEAM = GeoExpt[0].beam
DET = db_utils.strip_thickness_from_detector(GeoExpt[0].detector)
num_radial_bins = 500
assert BEAM is not None
assert DET is not None
MASK = db_utils.load_mask(mask_file)
DEV_ID = COMM.rank % ndev
this_ranks_imgs, this_ranks_refls = mpi_utils.mpi_load_exp_ref(input_file, maxN=maxN)

print0 = mpi_utils.print0f
print0("Creating radial profile maker!")
refGeom = {"D": DET, "B": BEAM}
radProMaker = RadPros(refGeom, numBins=num_radial_bins)
radProMaker.polarization_correction()
radProMaker.solidAngle_correction()


correction = radProMaker.POLAR * radProMaker.OMEGA
correction /= correction.mean()
for i_img in range(len(this_ranks_imgs)):
    this_ranks_imgs[i_img] *= correction

rots, wts = utils.load_quat_file(quat_file)

for i,R in enumerate(this_ranks_refls):
    Q = db_utils.refls_to_q(R, DET, BEAM)
    dspace = 1 / np.linalg.norm(Q, axis=1)
    sel = flex.bool(dspace >= highRes)
    Rsel = R.select(sel)
    assert len(Rsel) >= 3
    this_ranks_refls[i] = Rsel

this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
                                         Bmat_reference=Brecip, hcut=hcut, min_pred=min_pred)

# fit background to each image; estimate signal level per image
this_ranks_bgs = []
this_ranks_signal_levels = []
for i_img, (img, R) in enumerate(zip(this_ranks_imgs, this_ranks_refls)):
    radial_pro = radProMaker.makeRadPro(
        data_pixels=img,
        strong_refl=R,
        apply_corrections=False, use_median=False)

    img_background = radProMaker.expand_radPro(radial_pro)

    this_ranks_bgs.append(img_background)

    signal_level = utils.signal_level_of_image(R, img)
    if signal_level ==0:
        print("WARNING, shot has 0 signal level")
    this_ranks_signal_levels.append(signal_level)
    print0("Done with background image %d / %d" % (i_img+1, len(this_ranks_imgs)))


all_ranks_signal_levels = COMM.reduce(this_ranks_signal_levels)
ave_signal_level = None
if COMM.rank==0:
    ave_signal_level = np.mean(all_ranks_signal_levels)
ave_signal_level = COMM.bcast(ave_signal_level)
# let the initial density estimate be constant gaussians (add noise?)
Wstart = utils.get_W_init(ucell_p=ave_ucell)
Wstart /= Wstart.max()
Wstart *= ave_signal_level

# get the emc trilerp instance
qmap = utils.calc_qmap(DET, BEAM)
qx,qy,qz = map(lambda x: x.ravel(), qmap)
qcoords = np.vstack([qx,qy,qz]).T

Winit = np.zeros(const.DENSITY_SHAPE, np.float32)
corner, deltas = utils.corners_and_deltas(const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
maxRotInds = 10000000  # TODO make this a property of lerpy, and catch if trying to pass more rot inds

num_pix = MASK.size
L = lerpy()
rots = rots.astype(L.array_type)
Winit = Winit.astype(L.array_type)
qcoords = qcoords.astype(L.array_type)

L.allocate_lerpy(DEV_ID, rots.ravel(), Winit.ravel(), num_pix,
                 corner, deltas, qcoords.ravel(),
                 maxRotInds, num_pix)


# convert the type of images to match the lerpy instance array_type (prevents annoying warnings)
for i, img in enumerate(this_ranks_imgs):
    this_ranks_imgs[i] = img.astype(L.array_type).ravel()
    this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

# set the starting density
L.toggle_insert()
L.update_density(Wstart)
if COMM.rank==0:
    np.save(os.path.join(outdir, "Starting_density_relp"), Wstart)

init_shot_scales = np.ones(len(this_ranks_imgs))

# mask pixels that are outside the shell
# TODO: mask pixels not in bounds
inbounds = utils.qs_inbounds(qcoords, const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
print0("INIT SHOT SCALES:", init_shot_scales)
SHOT_MASK = inbounds*MASK.ravel()
from IPython import embed;embed();exit()

# make the mpi emc object
emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                    shot_mask=SHOT_MASK,
                    shot_background=this_ranks_bgs,
                    min_p=1e-5,
                    outdir=outdir,
                    beta=1,
                    shot_scales=init_shot_scales,
                    refine_scale_factors=True,
                    ave_signal_level=ave_signal_level,
                    density_update_method="lbfgs",
                    ucell_p=ave_ucell)

# run emc for specified number of iterations
emc.do_emc(niter)
