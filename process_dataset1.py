
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
    parser.add_argument("--hcut", default=0.02, type=float, help="distance from prediction to observed spot, used in determining probably orientaitons")
    parser.add_argument("--medFilt", type=int, default=8)
    parser.add_argument('--minPred', default=4, type=int, help="minimum number of spots that must be within hcut of a prediction in order for an orientation to be deemed probable")
    parser.add_argument("--restartfile", help="density file output by a previous run, e.g. Witer10.h5", default=None, type=str)
    parser.add_argument("--maxProc", default=None, type=int, help="only read this many shots from args.input file")
    parser.add_argument("--useCrystals", action="store_true", help="if experiments in args.expt contain crystal models, use those as probable orientations")
    parser.add_argument("--minProbRot", type=int, help="minumum number of probable orientations per shot", default=0)
    parser.add_argument("--saveBg", action="store_true", help="write background images to disk( will be used upon restart)")
    parser.add_argument("--useSavedBg", action="store_true")
    parser.add_argument("--noSym", action="store_true")
    parser.add_argument("--maxIter", type=int, default=60)
    parser.add_argument("--subsampleRots", default=None, type=float, nargs=2, help="2 numbers, first specifying size, second specifying angular resultion, for subsampling. E.g. 0.2 0.05 will subsample the probable rotations according to degree offsets from [-0.2 -0.15 -0.1 -0.5 0.5 1 1.5 2]",
                        metavar=("angularSpread","deltaAng"))
    parser.add_argument("--initDensity", type=str, default=None, choices=["fromMTZ", "fromRestart", "fromIndexed"])
    parser.add_argument("--mtz", type=str, nargs=2, default=None, metavar=("mtzFileName", "mtzCol"))
    parser.add_argument("--scaleFile", type=str, default=None, help="A Scales_iter%d.npz file from a previous emc job")
    parser.add_argument("--lowResLimit", type=float, default=None)
    parser.add_argument("--densDim", type=int, default=256)
    parser.add_argument("--highResLimit", type=float, default=4)
    parser.add_argument("--maxQ", type=float, default=4, help="specifies the size of the voxel grid")
    args = parser.parse_args()
    if args.initDensity=="fromMTZ":
        assert args.mtz is not None
else:
    args = None

args = COMM.bcast(args)

import sys
import os
import numpy as np
import h5py
import time
from itertools import product
from scipy.ndimage import median_filter as mf
from scipy.spatial.transform import Rotation
import socket

from dials.array_family import flex
#from copy import deepcopy
#from dxtbx.model.crystal import CrystalFactory
from simtbx.diffBragg import utils as db_utils
from dxtbx.model import ExperimentList


from simemc import mpi_utils
from simemc import utils
from simemc.compute_radials import RadPros
from simemc.emc import lerpy


ndevice = args.ndev
TEST_UCELLS = False

dens_dim = args.densDim
max_q = args.maxQ

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
if DEV_COMM.rank==0:
    rots, wts = utils.load_quat_file(args.quat)
else:
    rots = np.empty([])
    # wts = np.empty([])

print0("loaded rot mats")

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
        # TODO use these ? Do these even matter ?
        wts = np.append(wts, np.ones(len(extra_rots_not_none)) * np.mean(wts))
    print0("appended extra rot mats")

num_with_less_than_3 = 0
for i,R in enumerate(this_ranks_refls):
    Q = db_utils.refls_to_q(R, DET, BEAM)
    dspace = 1 / np.linalg.norm(Q, axis=1)
    sel = flex.bool(dspace >= highRes)
    Rsel = R.select(sel)
    nsel = len(Rsel)
    if nsel < 3:
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
        device_comm=DEV_COMM, minimum_prob_rot=args.minProbRot)

if args.subsampleRots is not None:
    # generate all rotational perturbations within the grid defined by subsampleRots
    ang_size, delta_ang = args.subsampleRots
    angs = np.arange(-ang_size, ang_size+1e-6, delta_ang)
    angs_xyz = product(angs, repeat=3)
    all_reff = []
    rot_x, rot_y, rot_z = np.eye(3)
    for ang_x, ang_y, ang_z in angs_xyz:
        if np.allclose([ang_x, ang_y, ang_z],0):
            continue
        rx=rot_x*ang_x
        ry=rot_y*ang_y
        rz=rot_z*ang_z
        rxyz=Rotation.from_rotvec([rx,ry,rz], degrees=True)
        reff = rxyz[0]*rxyz[1]*rxyz[2]
        all_reff.append(reff.as_matrix())
    print0("Adding %d rotation perturbation per probable rot" % len(all_reff))
    print0("Perturbing along rotational grid defined by the following list (degrees): ", angs)

    # Because rots only exists on DEV_COMM roots, we must scatter the probable rotation matrices to
    # each of the non-root ranks
    all_ranks_prob_rot = DEV_COMM.gather(this_ranks_prob_rot)
    all_ranks_prob_Umats = []
    if DEV_COMM.rank == 0:
        for ranks_prob_rot in all_ranks_prob_rot:
            ranks_prob_Umats = []
            for i_shot, rot_inds in enumerate(ranks_prob_rot):
                shot_Umats = [rots[i] for i in rot_inds]
                ranks_prob_Umats.append(shot_Umats)
            all_ranks_prob_Umats.append(ranks_prob_Umats)
    this_ranks_prob_Umats = DEV_COMM.scatter(all_ranks_prob_Umats)
    # now every rank has a list of Umats for each shot that corresonds to the list of probable rot indices

    # for each shot (on this rank), we will generate a larger list of probable rotation matrices, according to the rotation perturbations above
    this_ranks_U_perturbs = []
    for i_shot, prob_Umats in enumerate(this_ranks_prob_Umats):
        U_perturbs = []
        for ii, U in enumerate(prob_Umats):
            U_perturbs.append(np.dot(all_reff, U))
        this_ranks_U_perturbs.append(np.concatenate(U_perturbs))
        # these perturbation computations could/should be done on GPU, but that would require a large code change to the kernels... benchmark here for now to see how powerful perturbations are

    # Now for each dev_comm root, we need to update the rots ndarray to include all perturbation matrices (for ranks sharing that device/rots array (cuda IPC protocol))
    all_ranks_U_perturbs = DEV_COMM.gather(this_ranks_U_perturbs)

    # for dev comm roots, update the rots matrices to include "ALL" ranks perturbations (all means all on device)
    # then create the new rot_inds to append to this_rank_prob_rot
    all_ranks_extra_prob_rot = []
    if DEV_COMM.rank==0:
        start = rots.shape[0]
        for i_rank, rank_U_perturbs in enumerate(all_ranks_U_perturbs):
            print0("rank %d / %d appending rot mat perturbations" % (i_rank+1, DEV_COMM.size))
            ranks_extra_prob_rot = []
            for i_shot, U_perturbs in enumerate(rank_U_perturbs):
                # TODO only do this np.append once to avoid slow memory management
                rots = np.append(rots, U_perturbs, axis=0)
                new_rot_inds = np.arange(start, start+len(U_perturbs))
                ranks_extra_prob_rot.append(new_rot_inds)
                start += len(U_perturbs)
            all_ranks_extra_prob_rot.append(ranks_extra_prob_rot)

    # report to other ranks on the device there new extra prob rot indices
    this_ranks_extra_prob_rot = DEV_COMM.scatter(all_ranks_extra_prob_rot)
    # this_ranks_extra_prob_rot specifies the indices in the global rots array that correspond to the perturbations to this_ranks_prob_rot

    for i_shot, rot_inds in enumerate(this_ranks_extra_prob_rot):
        assert len(rot_inds) == len(this_ranks_prob_rot[i_shot]) * len(all_reff)
        this_ranks_prob_rot[i_shot] = np.hstack((this_ranks_prob_rot[i_shot], rot_inds))

    # make sure all prob_rot_inds are less than the total number of rots
    total_umats = None
    if DEV_COMM.rank==0:
        total_umats = len(rots)
        print("Total Umats on device %d (%s) = %d" % (DEV_ID, socket.gethostname(), total_umats))
    total_umats = DEV_COMM.bcast(total_umats)
    for i_shot, rot_inds in enumerate(this_ranks_prob_rot):
        assert np.max(rot_inds) < total_umats


if args.useCrystals:
    # for all of the loaded experiments with crystals, lets add the ground truth rotation matrix
    # to the list of probable rotation indices....
    nmissing_gt = 0
    nwith_gt = 0
    comps = {}
    for ii,(gt, inds) in enumerate(zip(this_ranks_gt_inds, this_ranks_prob_rot)):
        if gt is not None:
            nwith_gt += 1
            if gt not in set(inds):
                #rot_gt = rots[gt]
                #_CRYSTAL_DICT = {
                #    '__id__': 'crystal',
                #    'real_space_a': (ave_ucell[0], 0.0, 0.0),
                #    'real_space_b': (0.0, ave_ucell[1], 0.0),
                #    'real_space_c': (0.0, 0.0, ave_ucell[2]),
                #    'space_group_hall_symbol': '-P 4 2'}
                #C = CrystalFactory.from_dict(_CRYSTAL_DICT)
                #Cg = deepcopy(C)
                #Cg.set_U(tuple(rots[gt].T.ravel()))
                #Cinds = []
                #for Ui in inds:
                #    Ci = deepcopy(C)
                #    Ci.set_U(tuple(rots[Ui].T.ravel()))
                #    Cinds.append(Ci)
                #comps[gt] = (Cg, Cinds)
                ##comp_oir = db_utils.compare_with_ground_truth(*Cg.get_real_space_vectors(), Cinds, "P43212")
                this_ranks_prob_rot[ii] = np.append(this_ranks_prob_rot[ii], gt)
                nmissing_gt += 1

    #outs = []
    #for i in comps:
    #    outs_i = []
    #    Cg, Cinds = comps[i]
    #    for Ci in Cinds:
    #        try:
    #          out = db_utils.compare_with_ground_truth(*Cg.get_real_space_vectors(), [Ci], "P43212")
    #          outs_i.append(out[0])
    #        except:
    #          pass
    #    if outs_i:
    #        outs.append( min(outs_i))

    nwith_gt = COMM.reduce(nwith_gt)
    nmissing_gt = COMM.reduce(nmissing_gt)
    if COMM.rank==0:
        print0("Out of %d experiments with provided crystals, %d did not determine the gt rot ind as probable" %(nwith_gt, nmissing_gt))

    # sanity test on gt rot inds (just use dev comm root, as thats the only ranks with finite rots ndarrays
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
    print("Maximum number of probable orientations=%d"% max_nprob)  #TODO this can be done at the DEV_COMM level, but probably doesnt matter ...


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
print0("Getting relp mask")
peak_mask=None
if COMM.rank==0:
    peak_mask = utils.whole_punch_W(dens_dim, max_q, width=args.wholePunch, ucell_p=ave_ucell, symbol=symbol)
    vox_res = utils.voxel_resolution(dens_dim, max_q)
    highResLimit = 1. / L.max_q
    if args.highResLimit is not None:
        highResLimit = args.highResLimit
    vox_inbounds = vox_res >= highResLimit
    if args.lowResLimit is not None:
        assert args.lowResLimit > highResLimit
        vox_inbounds = np.logical_and(vox_inbounds, vox_res <= args.lowResLimit)
    print0("applying resolution cutoff to relp mask")
    peak_mask = np.logical_and(peak_mask, vox_inbounds)
peak_mask = mpi_utils.bcast_large(peak_mask, verbose=True, comm=COMM)

L.allocate_lerpy(DEV_ID, rots.ravel(), num_pix,
                 corner, deltas, qcoords.ravel(),
                 maxRotInds, num_pix,
                 peak_mask=peak_mask)

# convert the image dtypes to match the lerpy instance array_type (prevents annoying warnings)
for i, img in enumerate(this_ranks_imgs):
    this_ranks_imgs[i] = img.astype(L.array_type).ravel()
    this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

L.set_sym_ops(ave_ucell, symbol)

# set the starting density
L.toggle_insert()

print0("getting W init")

Wstart = None

if args.initDensity=="fromRestart":
    if COMM.rank==0:
        Wstart = h5py.File(args.restartfile, 'r')["Wprime"][()]

elif args.initDensity=="fromMTZ":
    if COMM.rank==0:
        print("using mtz file to generate a starting density")
        Wstart = utils.init_from_mtz(args.mtz[0], dens_dim, max_q, ave_ucell, symbol, mtzlabel=args.mtz[1])

elif args.initDensity=="fromIndexed":
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
        L.apply_friedel_symmetry(peak_mask)
        Wstart = np.zeros(L.dens_dim**3)
        Wstart[peak_mask.ravel()] =  L.densities()
        Wstart[Wstart<0] = 0
        Wstart = Wstart.reshape(L.dens_sh)

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

if COMM.rank==0:
    Wstart = Wstart[peak_mask]
# this method sets Wstart on rank0 and broadcasts to other ranks
L.mpi_set_starting_densities(Wstart, COMM)

init_shot_scales = np.ones(len(this_ranks_imgs))

if args.scaleFile is not None:
    scale_data = np.load(args.scaleFile)
    scales = scale_data["scales"]
    names = scale_data["names"]
    scale_map = {name:s for name, s in zip(names, scales)}
    missing_names = []
    for name in this_ranks_names:
        if name not in scale_map:
            missing_names.append(name)
    if missing_names:
        raise ValueError("In shot %s, scale factors are missing for the following experiments:\n%s"
                        % (args.scaleFile, ", ".join(missing_names)))
    init_shot_scales = np.array([scale_map[name] for name in this_ranks_names])

# mask pixels that are outside the shell
# TODO: mask pixels not in bounds
inbounds = utils.qs_inbounds(qcoords, L.dens_sh , L.xmin, L.xmax)
inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
print0("INIT SHOT SCALES:", init_shot_scales)
SHOT_MASK = inbounds*MASK.ravel()

# make the mpi emc object
print0("instantiating emc class instance")
emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                    peak_mask=peak_mask,
                    shot_mask=SHOT_MASK,
                    shot_background=this_ranks_bgs,
                    min_p=0,
                    outdir=args.outdir,
                    beta=1,
                    shot_scales=init_shot_scales,
                    refine_scale_factors=True,
                    ave_signal_level=ave_signal_level,
                    scale_update_method="bfgs",
                    shot_names=this_ranks_names,
                    density_update_method="lbfgs",
                    symbol=symbol,
                    symmetrize=not args.noSym,
                    ucell_p=ave_ucell,
                    max_iter=args.maxIter)

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
    error_logger.critical(err_s)
    COMM.Abort()

print0("Finish EMC")
