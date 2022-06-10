
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("ndev", type=int, help="number of GPU devices per node")
parser.add_argument("nshot", type=int, help="number of shots per rank")
parser.add_argument("outdir", type=str, help="output folder" )
parser.add_argument("--optscale", action="store_true", help="refine scale factors for each shot")
parser.add_argument("--niter", type=int, default=100, help="number of emc iterations")
parser.add_argument("--phil", type=str, default=None, help="path to a stills process phil file (for spot finding). Required if --water flag is used")
parser.add_argument("--minpred", type=int, default=3, help="minimum number of strong spots that need to be predicted well by an orientation for it to be flagged as probable for the shots crystal")
parser.add_argument("--hcut", type=float, default=0.1, help="maximum distance (in hkl units) to Bragg peaks from prediction for prediction Bragg peak to be considered part of the lattice ")
parser.add_argument("--xtalsize", type=float, default=0.002, help="size of the crystallite in mm")
parser.add_argument("--nowater", action="store_true", help="dont add water background")
parser.add_argument("--onlyGenImages", action="store_true", help="exit after simulating images")
parser.add_argument("--precomputedDir", default=None, type=str, help="load simulated images and refl tables from the outdir")
parser.add_argument("--startingDen", default=None, type=str, help="Path to a Witer%d.h5 file (output from previous run)")
parser.add_argument("--densityUpdater", type=str, default="lbfgs", choices=["lbfgs", "line_search", "analytical"],
                    help="the method for updating the density between iterations")
parser.add_argument("--dens_dim", type=int, default=256,help="number of voxels along cubic density edge")
parser.add_argument("--max_q", type=float, default=0.25,help="max q at voxel corner")
parser.add_argument("--poly", type=float, default=None, help="fwhm percentage for poly spectra")
parser.add_argument("--quat", type=int, default=70, help="number used as input to quat program")
parser.add_argument("--noSymmetrize", action="store_true", help="do not expand symmetry during refinement")
ARGS = parser.parse_args()

import pylab as plt
import numpy as np
import os
from line_profiler import LineProfiler
import inspect
from simtbx.diffBragg import utils as db_utils
import time
import glob
import h5py
from simemc import emc_updaters
from simemc import utils
from simemc import sim_const, sim_utils
from simemc import mpi_utils
from dials.array_family import flex
from simemc.compute_radials import RadPros
from simemc.emc import lerpy

from mpi4py import MPI
COMM = MPI.COMM_WORLD

print0 = mpi_utils.print0f
printR = mpi_utils.printRf
MIN_REF_PER_SHOT=3


def load_images(filedir, n=-1):
    """
    :param filedir: dir to load images from
    :param n: optional, stop loading once number of loaded images reaches this amount
    :return:
    """
    this_ranks_imgs = []
    this_ranks_refls = []
    this_ranks_names = []
    filenames = glob.glob(filedir + "/*.h5")

    # first make sure all refls exist:
    for i_f, f in enumerate(filenames):
        if i_f % COMM.size != COMM.rank:
            continue
        h5name = f
        refname = f.replace(".h5", ".refl")
        img = h5py.File(h5name, "r")["image"][()]
        this_ranks_imgs.append(img)
        R = flex.reflection_table.from_file(refname)
        this_ranks_refls.append(R)
        this_ranks_names.append(f)
        printR("Loaded shot %s (%d/%d)" % (h5name, i_f, len(filenames)))
        if len(this_ranks_imgs) == n:
            break
    return this_ranks_imgs, this_ranks_refls, this_ranks_names


def generate_n_images(nshot, seed, dev_id, xtal_size, phil_file, file_prefix):
    Famp = sim_utils.get_famp()
    SIM = sim_utils.get_noise_sim(0)
    SIM.seed=seed

    this_ranks_imgs = []
    this_ranks_refls = []
    this_ranks_names = []

    print0("Simulating water scattering...")
    water = sim_utils.get_water_scattering()
    if ARGS.nowater:
        water *= 0

    for i_shot in range(nshot):
        num_ref = 0
        while num_ref < MIN_REF_PER_SHOT:
            C = sim_utils.random_crystal()
            img = sim_utils.synthesize_cbf(
                SIM, C, Famp,
                dev_id=dev_id,
                xtal_size=xtal_size,
                outfile=file_prefix + "_%d.cbf" % i_shot,
                background=water, poly_perc=ARGS.poly)
            img = np.array([img])

            R = utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM, phil_file=phil_file)
            num_ref = len(R)
            print0("Shot %d / %d on device %d simulated with %d refls (%d required to proceed)"
                   % (i_shot+1, nshot, dev_id, num_ref, MIN_REF_PER_SHOT), flush=True)

        R.as_file(file_prefix+ "_%d.refl" % i_shot)
        # store an h5 for optional quick reloading with method load_images (h5s load faster than cbfs)
        with h5py.File(file_prefix + "_%d.h5"% i_shot, "w") as h:
            h.create_dataset("image", data=img)
        this_ranks_imgs.append(img)
        this_ranks_refls.append(R)
        this_ranks_names.append(file_prefix)
    sim_utils.delete_noise_sim(SIM)
    return this_ranks_imgs, this_ranks_refls, this_ranks_names





def get_rad_pro_maker(num_radial_bins=500):
    print0("Creating radial profile maker!", flush=True)
    refGeom = {"D": sim_const.DETECTOR, "B": sim_const.BEAM}
    radProMaker = RadPros(refGeom, numBins=num_radial_bins)
    radProMaker.polarization_correction()
    radProMaker.solidAngle_correction()
    return radProMaker


def load_rotation_samples():
    quat_file = os.path.join(os.path.dirname(__file__), "../quatgrid/c-quaternion%d.bin" % ARGS.quat)
    if not os.path.exists(quat_file):
        raise OSError("Please generate the quaternion file %s  with `./quat -bin 70`" % quat_file)
    rots, wts = utils.load_quat_file(quat_file)
    rots = rots.astype(np.float32)
    return rots, wts





def get_lerpy(dev_id, rotation_samples, qcoords, dens_dim=256, max_q=0.25,
    maxRotInds=20000):
    sh = dens_dim, dens_dim, dens_dim
    Winit = np.zeros(sh, np.float32)

    L = lerpy()
    L.dens_dim=dens_dim
    L.max_q=max_q
    corner, deltas = utils.corners_and_deltas(sh, L.xmin,L.xmax) 
    rots = rotation_samples.astype(L.array_type)
    Winit = Winit.astype(L.array_type)
    qcoords = qcoords.astype(L.array_type)

    L.allocate_lerpy(dev_id, rots.ravel(), Winit.ravel(), 2463*2527,
                     corner, deltas, qcoords.ravel(),
                     maxRotInds, 2463*2527)
    return L


#@pytest.mark.skip(reason="in development")
#@pytest.mark.mpi(min_size=1)
def emc_iteration():
    """
    """
    cbfdir = os.path.join(ARGS.outdir, "cbfs")
    for dir in [ARGS.outdir, cbfdir]:
        if not os.path.exists(dir):
            mpi_utils.make_dir(dir)

    np.random.seed(COMM.rank)
    DEV_ID = COMM.rank % ARGS.ndev

    file_prefix=os.path.join(cbfdir, "rank%d_shot"% COMM.rank)
    if ARGS.precomputedDir is None:
        this_ranks_imgs, this_ranks_refls, this_ranks_names = generate_n_images(ARGS.nshot, COMM.rank, DEV_ID, ARGS.xtalsize, ARGS.phil,
                                                              file_prefix=file_prefix)
    else:
        this_ranks_imgs, this_ranks_refls, this_ranks_names = load_images(ARGS.precomputedDir, ARGS.nshot)
    COMM.barrier()
    print0("Finished with image loading")

    if ARGS.onlyGenImages:
        exit()

    radProMaker = get_rad_pro_maker()

    correction = radProMaker.POLAR * radProMaker.OMEGA
    correction /= correction.mean()
    for i_img in range(len(this_ranks_imgs)):
        this_ranks_imgs[i_img] *= correction

    # probable orientations list per image
    rots, wts = load_rotation_samples()
    this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
        hcut=ARGS.hcut, min_pred=ARGS.minpred)

    print0("Getting maximum number of rot inds across all shots")
    max_rot_this_rank= max([len(rots) for rots in this_ranks_prob_rot])
    max_rot = COMM.gather(max_rot_this_rank)
    if COMM.rank==0:
        max_rot = max(max_rot)
    else:
        max_rot=None
    max_rot = COMM.bcast(max_rot)
    print0("Max rot inds=%d" % max_rot)
     

    # fit background to each image; estimate signal level per image
    this_ranks_bgs = []
    this_ranks_signal_levels = []
    for i_img, (img, R) in enumerate(zip(this_ranks_imgs, this_ranks_refls)):
        if ARGS.nowater:
            img_background = np.zeros_like(img)
        else:
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
    if ARGS.startingDen is not None:
        Wstart = h5py.File(ARGS.startingDen, "r")["Wprime"][()]
    else:
        # let the initial density estimate be constant gaussians (add noise?)
        Wstart = utils.get_W_init(ARGS.dens_dim, ARGS.max_q)
        Wstart /= Wstart.max()
        Wstart *= ave_signal_level

    # get the emc trilerp instance
    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    qcoords = np.vstack([qx,qy,qz]).T
    L = get_lerpy(DEV_ID, rots, qcoords, dens_dim=ARGS.dens_dim, max_q=ARGS.max_q, maxRotInds=max_rot)

    # convert the type of images to match the lerpy instance array_type (prevents annoying warnings)
    for i, img in enumerate(this_ranks_imgs):
        this_ranks_imgs[i] = img.astype(L.array_type).ravel()
        this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

    # set the starting density
    L.toggle_insert()
    L.update_density(Wstart)
    if COMM.rank==0:
        np.save(os.path.join(ARGS.outdir, "Starting_density_relp"), Wstart)

    init_shot_scales = np.ones(len(this_ranks_imgs))
    #if perturb_scale:
    #    init_shot_scales = np.random.uniform(1e-1, 1e3, len(this_ranks_imgs))

    # mask pixels that are outside the shell
    inbounds = utils.qs_inbounds(qcoords, L.dens_sh, L.xmin, L.xmax)
    inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
    print0("INIT SHOT SCALES:", init_shot_scales)

    # make the mpi emc object
    ucell_p = sim_const.CRYSTAL.get_unit_cell().parameters() 
    sym = sim_const.CRYSTAL.get_space_group().info().type().lookup_symbol()
    emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                        shot_mask=inbounds,
                        shot_background=this_ranks_bgs,
                        min_p=0,
                        outdir=ARGS.outdir,
                        beta=1,
                        shot_scales=init_shot_scales,
                        refine_scale_factors=ARGS.optscale,
                        ave_signal_level=ave_signal_level,
                        density_update_method=ARGS.densityUpdater,
                        ucell_p=ucell_p,
                        shot_names=this_ranks_names,
                        symmetrize=not ARGS.noSymmetrize,
                        symbol=sym)

    # run EMC wrapped in the line profiler
    lprof = LineProfiler()
    for cls in [mpi_utils.EMC, emc_updaters.DensityUpdater]:
        func_names, funcs = zip(*inspect.getmembers(cls, predicate=inspect.isfunction))
        for f in funcs:
            lprof.add_function(f)
    # wrap the method to profile it
    do_emc = lprof(emc.do_emc)

    # run emc for specified number of iterations
    do_emc(ARGS.niter)

    # print the line profiler stats
    stats = lprof.get_stats()
    mpi_utils.print_profile(stats)

    # make some assertions...
    print0("OK")


def plot_models(emc, init=False):
    models = emc.success_rate(init=init, return_models=True)
    fig, axs = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches((9.81,4.8))
    for i in range(4):
        ax = axs[0,i]
        imgs = models[i].reshape((2527, 2463)), emc.shots[i][0]
        img_ax = axs[0,i], axs[1,i]
        for ax,img in zip(img_ax, imgs):
            ax.imshow(img, vmin=0, vmax=.01)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            ax.grid(ls='--', lw=0.75)
            ax.set_aspect('auto')
    plt.draw()
    plt.pause(0.1)


if __name__=="__main__":
    emc_iteration()
