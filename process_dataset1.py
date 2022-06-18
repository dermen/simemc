from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("outdir", help="name of output folder", type=str)
parser.add_argument("--perlmutter", action="store_true", help="setup for perlmutter")
parser.add_argument("--allshots", action="store_true", help="run on all shots")
parser.add_argument("--allrefls", action="store_true", help="use all reflection to determine probable orientations. DEfault is to just use those refls out to max_q")
parser.add_argument("--highres", action="store_true", help="use the high resolution setting (max_q=0.5, dens_dim=512). Otherwise use low-res (max_q=0.25, dens_dim=256)")
parser.add_argument("--hcut", default=0.02, type=float, help="distance from prediction to observed spot, used in determining probably orientaitons")
parser.add_argument('--min_pred', default=4, type=int, help="minimum number of spots that must be within hcut of a prediction in order for an orientation to be deemed probable")
parser.add_argument("--restartfile", help="density file output by a previous run, e.g. Witer10.h5", default=None, type=str)
args = parser.parse_args()


import sys
import os
from mpi4py import MPI
from copy import deepcopy
COMM = MPI.COMM_WORLD

import numpy as np
import h5py
from scipy.ndimage import median_filter as mf
from dials.array_family import flex
import time

from simtbx.modeling.forward_models import diffBragg_forward
from dxtbx.model import Crystal
from simemc import mpi_utils, utils
from simemc.compute_radials import RadPros
from simtbx.diffBragg import utils as db_utils
from dxtbx.model import ExperimentList
from simemc.emc import lerpy


outdir=sys.argv[1]
perlmutt = args.perlmutter
highRes=args.highres
ndevice = 4 if perlmutt else 8
TEST_UCELLS = False

if highRes:
    dens_dim = 512
    max_q = 0.5
else:
    dens_dim=256
    max_q=0.25


mf_filter_sh=8,8
if perlmutt:
    datadir = os.environ["SCRATCH"] + "/dataset_1"
    quat_file = "/global/cfs/cdirs/lcls/dermen/lyso/alcc-recipes/cctbx/modules/simemc/quatgrid/c-quaternion120.bin"
else:
    datadir = os.environ["CSCRATCH"] + "/dataset_1"
    quat_file = "/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules/simemc/quatgrid/c-quaternion120.bin"

USE_RSEL= not args.allrefls
if args.allshots:
    input_file = "all_short_Caxis_from_probRot_test.txt"
else:
    input_file = "small_Caxis_1315.txt" 


start_file = args.restartfile
ref_geom_file = datadir + "/split_0000.expt"
mask_file =     datadir + "/test_mask.pkl"
ave_ucell = 68.48, 68.48, 104.38, 90,90,90
symbol="P43212"
hcut=args.hcut
min_pred=args.min_pred
niter=100
num_radial_bins = 1000
maxN = None
highRes = 1./max_q

mpi_utils.make_dir(outdir)

ucell_man = db_utils.manager_from_params(ave_ucell)
Brecip = ucell_man.B_recipspace

GeoExpt = ExperimentList.from_file(ref_geom_file,False)
BEAM = GeoExpt[0].beam
DET = db_utils.strip_thickness_from_detector(GeoExpt[0].detector)
assert BEAM is not None
assert DET is not None
MASK = db_utils.load_mask(mask_file)
DEV_ID = COMM.rank % ndevice
this_ranks_imgs, this_ranks_refls, this_ranks_names = mpi_utils.mpi_load_exp_ref(input_file, maxN=maxN)

print0 = mpi_utils.print0f
print0("Creating radial profile maker!")
refGeom = {"D": DET, "B": BEAM}
radProMaker = RadPros(refGeom, numBins=num_radial_bins, maskFile=MASK)
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
    if USE_RSEL:
        Rsel = R.select(sel)
        assert len(Rsel) >= 3
        this_ranks_refls[i] = Rsel
    else:
        this_ranks_refls[i] = R


if TEST_UCELLS:
    ave_ucell1 = 68.48, 68.48, 104.38, 90,90,90
    ave_ucell2 = 68.17, 68.17, 108.19, 90,90,90
    all_prob_rot = []
    for uc in ave_ucell1, ave_ucell2: 
        ucell_man = db_utils.manager_from_params(uc)
        Brecip = ucell_man.B_recipspace
        this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
                                                 Bmat_reference=Brecip, hcut=hcut, min_pred=min_pred,
                                                verbose=COMM.rank==0, detector=DET,beam=BEAM)
        all_prob_rot.append( this_ranks_prob_rot)


    preferred_ucell = np.argmax(list(zip([len(p) for p in all_prob_rot[0]], [len(p) for p in all_prob_rot[1]])),axis=1)
    ucell_info = COMM.gather(list(zip(this_ranks_names, preferred_ucell)))
    if COMM.rank==0:
        np.save("ucell_info", ucell_info)
    COMM.barrier()
    exit()

else:        
    this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
            Bmat_reference=Brecip, hcut=hcut, min_pred=min_pred,
            verbose=COMM.rank==0, detector=DET,beam=BEAM)


n_prob_rot = [len(prob_rots) for prob_rots in this_ranks_prob_rot]

# fit background to each image; estimate signal level per image
this_ranks_bgs = []
this_ranks_signal_levels = []
for i_img, (img, R) in enumerate(zip(this_ranks_imgs, this_ranks_refls)):
    t = time.time()
    radial_pro, bg_mask = radProMaker.makeRadPro(
        data_pixels=img,
        strong_refl=R,
        apply_corrections=False, use_median=True, return_mask=True)

    img_background = radProMaker.expand_radPro(radial_pro)
    img_filled = img.copy()
    img_filled[~bg_mask] = img_background[~bg_mask]
    bg = mf(img_filled[0], mf_filter_sh)

    this_ranks_bgs.append(np.array([bg]))
    t = time.time()-t

    signal_level = utils.signal_level_of_image(R, img)
    if signal_level ==0:
        print("WARNING, shot has 0 signal level")
    bg_signal_level = utils.signal_level_of_image(R, this_ranks_bgs[-1])
    assert bg_signal_level <= signal_level
    this_ranks_signal_levels.append(signal_level-bg_signal_level)

    # TODO subtract bg signal level ? 
    print0("Done with background image %d / %d (Took %f sec to fit bg) (signal=%f, bg_sig=%f)" % (i_img+1, len(this_ranks_imgs), t, signal_level, bg_signal_level))


all_ranks_signal_levels = COMM.reduce(this_ranks_signal_levels)
ave_signal_level = None
if COMM.rank==0:
    ave_signal_level = np.mean(all_ranks_signal_levels)
ave_signal_level = COMM.bcast(ave_signal_level)
# let the initial density estimate be constant gaussians (add noise?)
Wstart = utils.get_W_init(dens_dim, max_q, ucell_p=ave_ucell, symbol=symbol)
Wstart /= Wstart.max()
Wstart *= ave_signal_level

if start_file is not None:
    Wstart = h5py.File(start_file, 'r')["Wprime"][()]

# get the emc trilerp instance
qmap = utils.calc_qmap(DET, BEAM)
qx,qy,qz = map(lambda x: x.ravel(), qmap)
qcoords = np.vstack([qx,qy,qz]).T

L = lerpy()
L.dens_dim = dens_dim
L.max_q = max_q

Winit = np.zeros(L.dens_sh, np.float32)
corner, deltas = utils.corners_and_deltas(L.dens_sh, L.xmin, L.xmax)
maxRotInds = 100000  # TODO make this a property of lerpy, and catch if trying to pass more rot inds
max_n= max(n_prob_rot)
print("Max rots on rank %d: %d" % (COMM.rank, max_n))
if maxRotInds < max_n:
    maxRotInds = max_n

num_pix = MASK.size
rots = rots.astype(L.array_type)
Winit = Winit.astype(L.array_type)
qcoords = qcoords.astype(L.array_type)

L.allocate_lerpy(DEV_ID, rots.ravel(), Winit.ravel(), num_pix,
                 corner, deltas, qcoords.ravel(),
                 maxRotInds, num_pix)

# convert the image dtypes to match the lerpy instance array_type (prevents annoying warnings)
for i, img in enumerate(this_ranks_imgs):
    this_ranks_imgs[i] = img.astype(L.array_type).ravel()
    this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

# set the starting density
L.toggle_insert()
L.update_density(Wstart)
L.dens_dim=dens_dim
if COMM.rank==0:
    np.save(os.path.join(outdir, "Starting_density_relp"), Wstart)

init_shot_scales = np.ones(len(this_ranks_imgs))

# mask pixels that are outside the shell
# TODO: mask pixels not in bounds
inbounds = utils.qs_inbounds(qcoords, L.dens_sh , L.xmin, L.xmax)
inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
print0("INIT SHOT SCALES:", init_shot_scales)
SHOT_MASK = inbounds*MASK.ravel()

#class Model:
#    def __init__(self):
#        self.fluxes = [1e12]
#        self.energies = [db_utils.ENERGY_CONV/BEAM.get_wavelength()]
#        self.Famp = db_utils.make_miller_array("P43212", ave_ucell)
#        hall = self.Famp.space_group_info().type().hall_symbol()
#        ucell_man = db_utils.manager_from_params(ave_ucell)
#        Bmat = np.reshape(ucell_man.B_realspace, (3, 3))
#        a1, a2, a3 = map(tuple, [Bmat[:, 0], Bmat[:, 1], Bmat[:, 2]])
#        cryst_descr = {'__id__': 'crystal',
#                       'real_space_a': a1,
#                       'real_space_b': a2,
#                       'real_space_c': a3,
#                       'space_group_hall_symbol': hall}
#
#        self.CRYST= Crystal.from_dict(cryst_descr)
#
#    def sanity_check_model(self, umat):
#        C= deepcopy(self.CRYST)
#        umat = tuple(umat.ravel())
#        C.set_U(umat)
#        model = diffBragg_forward(
#            C, DET, BEAM, self.Famp, self.energies, self.fluxes,
#            oversample=1, Ncells_abc=(20,20,20),
#            mos_dom=1, mos_spread=0, beamsize_mm=0.001,
#            device_Id=0,
#            show_params=False, crystal_size_mm=0.005, printout_pix=None,
#            verbose=0, default_F=0, interpolate=0, profile="gauss",
#            mosaicity_random_seeds=None,
#            show_timings=False,
#            nopolar=False, diffuse_params=None)
#        return model

#M = Model()
#
#for i,img in enumerate(this_ranks_imgs):
#    R = this_ranks_refls[i]
#    x, y, z = R['xyzobs.px.value'].parts()
#
#    x-=0.5
#    y-=0.5
#    models =[]
#    new_group = True
#    prev_model = None
#    for rot_id in this_ranks_prob_rot[i]:
#
#        umat = rots[rot_id]
#        model = M.sanity_check_model(umat)
#        if prev_model is None:
#            prev_model = model
#            new_group = True
#        else:
#            if np.allclose(prev_model, model):
#                new_group = False
#            else:
#                new_group = True
#                prev_model = model
#        if new_group:
#
#            Rpred = utils.refls_from_sims(model, DET, BEAM, thresh=1e-4)
#            utils.label_strong_reflections(Rpred, R, 1)
#            nclose = np.sum(Rpred['is_strong'])
#            Rclose = Rpred.select(Rpred["is_strong"])
#            xobs, yobs, _ = Rclose["xyzobs.px"] .parts()
#            xcal, ycal,_ = Rclose["xyzcal.px"].parts()
#
#            pred_dist = np.mean(np.sqrt((xobs-xcal)**2 + (yobs-ycal)**2))
#            print("model %d, nclose=%d, predoffset=%.4f pix" %(rot_id, nclose, pred_dist ))
#
#    from IPython import embed;embed()
#
#

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
                    scale_update_method="bfgs",
                    shot_names=this_ranks_names,
                    density_update_method="lbfgs",
                    symbol=symbol,
                    ucell_p=ave_ucell)

# run emc for specified number of iterations
emc.do_emc(niter)
