
from mpi4py import MPI
COMM = MPI.COMM_WORLD

from argparse import ArgumentParser
if COMM.rank == 0:
    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="exp ref file (see utils.make_exp_ref_spec_file)")
    parser.add_argument("mask", type=str, help="path to a dials mask (pickle) file")
    parser.add_argument("geom", type=str, help="path to a dxtbx Experiment list containing accurate detector model")
    parser.add_argument("quat", type=str, help="path to the quaternion file (see simemc/quatgrid/README)")
    parser.add_argument("outdir", help="name of output folder", type=str)
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--wholePunch", type=int, default=1)
    parser.add_argument("--even", action="store_true", help="read in data from even line numbers in args.input file")
    parser.add_argument("--odd", action="store_true", help="read in data from odd line numbers in args.input file")
    parser.add_argument("--allrefls", action="store_true", help="use all reflection to determine probable orientations. DEfault is to just use those refls out to max_q")
    # TODO change to dens_dim and max_q arguments!
    parser.add_argument("--highres", action="store_true", help="use the high resolution setting (max_q=0.5, dens_dim=512). Otherwise use low-res (max_q=0.25, dens_dim=256)")
    parser.add_argument("--hcut", default=0.02, type=float, help="distance from prediction to observed spot, used in determining probably orientaitons")
    parser.add_argument("--medFilt", type=int, default=8)
    parser.add_argument('--minPred', default=4, type=int, help="minimum number of spots that must be within hcut of a prediction in order for an orientation to be deemed probable")
    parser.add_argument("--restartfile", help="density file output by a previous run, e.g. Witer10.h5", default=None, type=str)
    parser.add_argument("--maxProc", default=None, type=int, help="only read this many shots from args.input file")
    parser.add_argument("--useCrystals", action="store_true", help="if experiments in args.expt contain crystal models, use those as probable orientations")
    parser.add_argument("--saveBg", action="store_true", help="write background images to disk( will be used upon restart)")
    parser.add_argument("--useSavedBg", action="store_true")
    parser.add_argument("--noSym", action="store_true")
    parser.add_argument("--maxIter", type=int, default=60)
    args = parser.parse_args()
else:
    args = None

args = COMM.bcast(args)

import sys
import os
import numpy as np
import h5py
from scipy.ndimage import median_filter as mf
import time

from dials.array_family import flex
from simtbx.diffBragg import utils as db_utils
from dxtbx.model import ExperimentList


from simemc import mpi_utils
from simemc import utils
from simemc.compute_radials import RadPros
from simemc.emc import lerpy


ndevice = args.ndev
TEST_UCELLS = False

if args.highres:
    dens_dim = 751
    max_q = 0.5
else:
    dens_dim=351
    max_q=0.25


mf_filter_sh= args.medFilt, args.medFilt

USE_RSEL= not args.allrefls


ref_geom_file = args.geom
mask_file = args.mask
ave_ucell = 68.5, 68.5, 104.4, 90,90,90
symbol="P43212"
niter=100
num_radial_bins = 1000
highRes = 1./max_q

mpi_utils.make_dir(args.outdir)
if COMM.rank==0:
    with open(os.path.join(args.outdir, "command_line_input.txt"), "w") as o:
        o.write("Command line input:\n %s\n" % " ".join(sys.argv))

mpi_utils.setup_rank_log_files(args.outdir + "/ranklogs", utils.LOGNAME)

ucell_man = db_utils.manager_from_params(ave_ucell)
Brecip = ucell_man.B_recipspace

GeoExpt = ExperimentList.from_file(ref_geom_file,False)
BEAM = GeoExpt[0].beam
DET = db_utils.strip_thickness_from_detector(GeoExpt[0].detector)
assert BEAM is not None
assert DET is not None
MASK = db_utils.load_mask(mask_file)
DEV_ID = COMM.rank % ndevice
this_ranks_imgs, this_ranks_refls, this_ranks_names, this_ranks_crystals = mpi_utils.mpi_load_exp_ref(args.input, maxN=args.maxProc, even=args.even, odd=args.odd)

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

DEV_COMM = mpi_utils.get_host_dev_comm(DEV_ID)
print("Got device communicator")
if DEV_COMM.rank==0:
    rots, wts = utils.load_quat_file(args.quat)
else:
    rots = np.empty([])
    # wts = np.empty([])

mpi_utils.printRf("loaded rot mats")

if args.useCrystals:#  and np.any([C is not None for C in this_ranks_crystals]):
    extra_rots = [np.reshape(C.get_U(), (3,3)) if C is not None else None for C in this_ranks_crystals]
    extra_rots = COMM.gather(extra_rots)
    all_req = []
    if COMM.rank==0:
        extra_rots_not_none = []
        extra_rot_ind = rots.shape[0]
        for i_rank, more_rots in enumerate(extra_rots):
            inds = []  # inds is either None, or the index of the crystal rotation matrix in the grid
            for i_rot, Umat in enumerate(more_rots):
                if Umat is None:
                    inds.append(None)
                else:
                    extra_rots_not_none.append(Umat)
                    inds.append(extra_rot_ind)
                    extra_rot_ind += 1

            req = COMM.isend(inds, dest=i_rank, tag=i_rank)
            all_req.append(req)
    else:
        extra_rots_not_none = None

    print0("broadcasting extra rots")
    extra_rots_not_none = COMM.bcast(extra_rots_not_none)

    print0("receiving requests")
    this_ranks_gt_inds = COMM.recv(source=0, tag=COMM.rank)
    for req in all_req:
        req.wait()

    print0("Sent all rot mat indices")
    if DEV_COMM.rank==0:
        rots = np.append(rots, extra_rots_not_none, axis=0)
        wts = np.append(wts, np.ones(len(extra_rots_not_none)) * np.mean(wts))
    print0("appended extra rot mats")

num_with_less_than_3 = 0
for i,R in enumerate(this_ranks_refls):
    Q = db_utils.refls_to_q(R, DET, BEAM)
    dspace = 1 / np.linalg.norm(Q, axis=1)
    sel = flex.bool(dspace >= highRes)
    Rsel = R.select(sel)
    nsel = len(Rsel)
    if nsel <3:
        num_with_less_than_3 += 1
    if USE_RSEL and nsel >=3:
        this_ranks_refls[i] = Rsel
    else:
        this_ranks_refls[i] = R

num_with_less_than_3 = COMM.reduce(num_with_less_than_3)
if COMM.rank==0:
    print("Number of shots with fewer than 3 refls within the  max_q cutoff=%d" %num_with_less_than_3)

#if TEST_UCELLS:
#    ave_ucell1 = 68.48, 68.48, 104.38, 90,90,90
#    ave_ucell2 = 68.17, 68.17, 108.19, 90,90,90
#    all_prob_rot = []
#    for uc in ave_ucell1, ave_ucell2:
#        ucell_man = db_utils.manager_from_params(uc)
#        Brecip = ucell_man.B_recipspace
#        this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
#                                                 Bmat_reference=Brecip, hcut=args.hcut, min_pred=args.minPred,
#                                                verbose=COMM.rank==0, detector=DET,beam=BEAM)
#        all_prob_rot.append( this_ranks_prob_rot)
#
#    preferred_ucell = np.argmax(list(zip([len(p) for p in all_prob_rot[0]], [len(p) for p in all_prob_rot[1]])),axis=1)
#    ucell_info = COMM.gather(list(zip(this_ranks_names, preferred_ucell)))
#    if COMM.rank==0:
#        np.save("ucell_info", ucell_info)
#    COMM.barrier()
#    exit()
#
#else:
this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
        Bmat_reference=Brecip, hcut=args.hcut, min_pred=args.minPred,
        verbose=COMM.rank==0, detector=DET,beam=BEAM, hcut_incr=0.0025,
        device_comm=DEV_COMM)

# for all of the loaded experiments with crystals, lets add the ground truth rotation matrix
# to the list of probable rotation indices....
nmissing_gt = 0
nwith_gt = 0
for ii,(gt, inds) in enumerate(zip(this_ranks_gt_inds, this_ranks_prob_rot)):
    if gt is not None:
        nwith_gt += 1
        if gt not in set(inds):
            this_ranks_prob_rot[ii] = np.append(this_ranks_prob_rot[ii], gt)
            nmissing_gt += 1
nwith_gt = COMM.reduce(nwith_gt)
nmissing_gt = COMM.reduce(nmissing_gt)
if COMM.rank==0:
    print0("Out of %d experiments with provided crystals, %d did not determine the gt rot ind as probable" %(nwith_gt, nmissing_gt))

# sanity test on gt rot inds
if DEV_COMM.rank==0:
    for gt, C in zip(this_ranks_gt_inds, this_ranks_crystals):
        if C is None:
            continue
        Umat = rots[gt]
        Umat2 = np.reshape(C.get_U(), (3,3))
        assert np.allclose(Umat, Umat2)


has_no_rots = [len(prob_rots)==0 for prob_rots in this_ranks_prob_rot]
has_no_rots = COMM.reduce(has_no_rots)
if COMM.rank==0:
    n_with_no_rots = sum(has_no_rots)
    print0("Shots with 0 prob rots=%d" % n_with_no_rots)

n_prob_rot = [len(prob_rots) for prob_rots in this_ranks_prob_rot]
all_ranks_max_n_prob = COMM.gather(np.max(n_prob_rot))
if COMM.rank==0:
    max_nprob = np.max(all_ranks_max_n_prob)
    print("Maximum number of probable orientations=%d"% max_nprob)


def background_fit(img, R, radProMaker):
    radial_pro, bg_mask = radProMaker.makeRadPro(
        data_pixels=img,
        strong_refl=R,
        apply_corrections=False, use_median=True, return_mask=True)

    img_background = radProMaker.expand_radPro(radial_pro)
    img_filled = img.copy()
    img_filled[~bg_mask] = img_background[~bg_mask]
    bg = mf(img_filled[0], mf_filter_sh)
    return bg



# fit background to each image; estimate signal level per image
this_ranks_bgs = []
this_ranks_signal_levels = []
for i_img, (img,name, R) in enumerate(zip(this_ranks_imgs, this_ranks_names, this_ranks_refls)):
    t = time.time()
    h5name = os.path.splitext(name)[0]+"_background.h5"
    if os.path.exists(h5name) and args.useSavedBg:
        with h5py.File(h5name,"r") as h5:
            bg = h5["bg"][()]
    else:
        bg = np.array([background_fit(img, R, radProMaker)])
    this_ranks_bgs.append(bg)
    t = time.time()-t

    signal_level = utils.signal_level_of_image(R, img)
    if signal_level ==0:
        print("WARNING, shot has 0 signal level")
    bg_signal_level = utils.signal_level_of_image(R, this_ranks_bgs[-1])
    assert bg_signal_level <= signal_level
    this_ranks_signal_levels.append(signal_level-bg_signal_level)

    # TODO subtract bg signal level ? 
    print0("Done with background image %d / %d (Took %f sec to fit bg) (signal=%f, bg_sig=%f)" % (i_img+1, len(this_ranks_imgs), t, signal_level, bg_signal_level))


for img, bg, name in zip(this_ranks_imgs, this_ranks_bgs, this_ranks_names):
    h5name = os.path.splitext(name)[0] +"_background.h5"
    if args.saveBg:
        with h5py.File(h5name, "w") as h5:
            h5.create_dataset("bg", data=bg)
            h5.create_dataset("img", data=img)

all_ranks_signal_levels = COMM.reduce(this_ranks_signal_levels)
ave_signal_level = None
if COMM.rank==0:
    ave_signal_level = np.mean(all_ranks_signal_levels)
ave_signal_level = COMM.bcast(ave_signal_level)
# let the initial density estimate be constant gaussians (add noise?)

# get the emc trilerp instance
qmap = utils.calc_qmap(DET, BEAM)
qx,qy,qz = map(lambda x: x.ravel(), qmap)
qcoords = np.vstack([qx,qy,qz]).T

L = lerpy()
L.rank = COMM.rank
L.dens_dim = dens_dim
L.max_q = max_q

corner, deltas = utils.corners_and_deltas(L.dens_sh, L.xmin, L.xmax)
maxRotInds = 100000  # TODO make this a property of lerpy, and catch if trying to pass more rot inds
max_n= max(n_prob_rot)
#print("Max rots on rank %d: %d" % (COMM.rank, max_n))
if maxRotInds < max_n:
    maxRotInds = max_n

num_pix = MASK.size
rots = rots.astype(L.array_type)
qcoords = qcoords.astype(L.array_type)

L.allocate_lerpy(DEV_ID, rots.ravel(), num_pix,
                 corner, deltas, qcoords.ravel(),
                 maxRotInds, num_pix)

# convert the image dtypes to match the lerpy instance array_type (prevents annoying warnings)
for i, img in enumerate(this_ranks_imgs):
    this_ranks_imgs[i] = img.astype(L.array_type).ravel()
    this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

L.set_sym_ops(ave_ucell, symbol)

# set the starting density
L.toggle_insert()

print0("getting W init")

Wstart = None

if args.restartfile is not None:
    if COMM.rank==0:
        Wstart = h5py.File(args.restartfile, 'r')["Wprime"][()]

elif args.useCrystals:
    # TODO: is it better to start the density as 3D gaussians ?
    for ii, (gt_rot_idx, img, bg) in enumerate(zip(this_ranks_gt_inds, this_ranks_imgs, this_ranks_bgs)):
        if gt_rot_idx is None:
            continue
        print0("inserting %d / %d" % (ii+1,len(this_ranks_imgs)))
        L.trilinear_insertion(gt_rot_idx, img-bg)  # TODO: should we insert the difference ?

    L.reduce_densities(COMM)
    L.reduce_weights(COMM)

    print0("dividing")
    if COMM.rank==0:
        W = L.densities()
        wt = L.wts()
        Wstart = utils.errdiv(W, wt)
        print0("Symmetrizing")
        L.update_density(Wstart.ravel())
        print0("applying crystal symm")
        L.symmetrize()
        print0("applying friedel")
        L.apply_friedel_symmetry()
        Wstart = L.densities().reshape(L.dens_sh)
        Wstart[Wstart<0] = 0

else:
    if COMM.rank==0:
        Wstart = utils.get_W_init(dens_dim, max_q, ucell_p=ave_ucell, symbol=symbol)


if COMM.rank==0:
    Wstart /= Wstart.max()
    Wstart *= ave_signal_level

#if COMM.rank==0:
#    with h5py.File("temp%d_hr_init.h5"% dens_dim, "w") as h:
#        h.create_dataset("Wprime", data=Wstart)
#        h.create_dataset("ucell", data=ave_ucell)

# this method sets Wstart on rank0 and broadcasts to other ranks
L.mpi_set_starting_densities(Wstart, COMM)

init_shot_scales = np.ones(len(this_ranks_imgs))

# mask pixels that are outside the shell
# TODO: mask pixels not in bounds
inbounds = utils.qs_inbounds(qcoords, L.dens_sh , L.xmin, L.xmax)
inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
print0("INIT SHOT SCALES:", init_shot_scales)
SHOT_MASK = inbounds*MASK.ravel()

# make the mpi emc object
print0("instantiating emc class instance")
emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                    shot_mask=SHOT_MASK,
                    shot_background=this_ranks_bgs,
                    min_p=1e-5,
                    outdir=args.outdir,
                    beta=1,
                    whole_punch=args.wholePunch,
                    shot_scales=init_shot_scales,
                    refine_scale_factors=True,
                    ave_signal_level=ave_signal_level,
                    scale_update_method="bfgs",
                    shot_names=this_ranks_names,
                    density_update_method="lbfgs",
                    symbol=symbol,
                    symmetrize=not args.noSym,
                    ucell_p=ave_ucell)

emc.max_iter = args.maxIter
# run emc for specified number of iterations
print0("Begin EMC")
error_logger = mpi_utils.setup_rank_log_files(args.outdir+"/ranklogs", name="errors", ext="err")
try:
    emc.do_emc(niter)
except Exception as err:
    from traceback import format_tb
    # prepend RANK to each line of the traceback
    _,_,tb = sys.exc_info()
    tb_s = "".join(format_tb(tb))
    tb_s = tb_s.replace("\n", "\nRANK%04d"%COMM.rank)
    err_s = str(err)+"\n"+ tb_s
    error_logger.critical(err_s)# , exc_info=True)
    #error_logger.critical(err, exc_info=True)
    COMM.Abort()

print0("Finish EMC")
