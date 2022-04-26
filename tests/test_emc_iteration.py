
import pytest
import pylab as plt
import numpy as np
import os
from line_profiler import LineProfiler
import inspect
from simtbx.diffBragg import utils as db_utils
import time
import h5py
from simemc import utils
from simemc import sim_const, sim_utils
from simemc import mpi_utils
from dials.array_family import flex
from simemc.compute_radials import RadPros
from simemc.emc import lerpy, probable_orients

from simemc import const

print0 = mpi_utils.print0f
printR = mpi_utils.printRf


@pytest.mark.skip(reason="in development")
@pytest.mark.mpi(min_size=1)
def test_emc_iteration(ndev, nshots_per_rank=60, rots_from_grid=True, start_with_relp=False, outdir=None,
                       add_water=False, niter=100, phil_file=None, min_pred=7, hcut=0.03, cbfdir=None, xtal_size=0.002,
                       use_precomputed=False, refine_scale=False, perturb_scale=False):
    """

    :param ndev: number of gpu
    :param nshots_per_rank: number of shots to simulate per-rank
    :param rots_from_grid: force crystal orientations to lie on the quaternion grid
    :param start_with_relp: starting W model is a reciprocal lattice model (gaussians on relps)
    :param outdir: write Witer files here (density at each iteration, read by the plot.py and make_corr_plot.py scripts)
    :param add_water: add water to the simulated images
    :param niter: number of emc iteration
    :param phil_file: PHIL file for doing spot finding with DIALS
    :param min_pred: minimum number of strong spots predicted by lattice orientation in order for that orientation to be probable
    :param hcut: fractional miller index distance from strong spot to lattice such that strong spot might be called a prediction
    :param cbfdir: directory where to write cbf files (image simulations)
    :param xtal_size: size of the crystal (controls number of spots per image)
    :param use_precomputed: if re-running this command with similar parameters, just reload the data from the cbf dir
    :param refine_scale: refine scale factors for each shot along with the density
    :param perturb_scale: perturb the scale factors at the beginning of EMC
    :return:
    """
    MIN_REF_PER_SHOT=4

    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    if not os.path.exists(outdir):
        mpi_utils.make_dir(outdir)
    np.random.seed(COMM.rank)
    quat_file = os.path.join(os.path.dirname(__file__), "../quatgrid/c-quaternion70.bin")
    if not os.path.exists(quat_file):
        raise OSError("Please generate the quaternion file %s  with `./quat -bin 70`" % quat_file)
    rots, wts = utils.load_quat_file(quat_file)

    Famp = sim_utils.get_famp()

    SIM = sim_utils.get_noise_sim(0)
    SIM.seed=COMM.rank

    dev_id = COMM.rank % ndev
    this_ranks_imgs = []
    this_ranks_rot_indices = []
    this_ranks_ishots = []
    nshot = COMM.size*nshots_per_rank
    print0("Simulating %d shots on %d ranks" % (nshot, COMM.size))
    rot_indices = np.random.permutation(rots.shape[0])[:nshot]

    water = 0
    radProMaker = None
    correction = 1
    if add_water:
        if COMM.rank==0:
            print("Simulating water scattering...")
            water = sim_utils.get_water_scattering()
        water = COMM.bcast(water)

        print0("Creating radial profile maker!", flush=True)
        refGeom = {"D": sim_const.DETECTOR, "B": sim_const.BEAM}
        radProMaker = RadPros(refGeom, numBins=500)
        radProMaker.polarization_correction()
        radProMaker.solidAngle_correction()
        correction = radProMaker.POLAR * radProMaker.OMEGA
        correction /= correction.mean()

    gt_rots = []
    if cbfdir is not None:
        mpi_utils.make_dir(cbfdir)
    for i_shot in range(nshot):
        if i_shot % COMM.size != COMM.rank:
            continue
        print0("Shot %d / %d on device %d" % (i_shot+1, nshot, dev_id))
        rot_idx = rot_indices[i_shot]
        outfile = None
        if cbfdir is not None:
            outfile = os.path.join(cbfdir, "shot%d.cbf" % i_shot)
            h5_file = outfile.replace(".cbf", ".h5")
        if use_precomputed and cbfdir is not None and os.path.exists(h5_file):
            with h5py.File(h5_file, "r") as h5:
                img = h5["img"][()]
        else:
            C = sim_utils.random_crystal()
            if rots_from_grid:
                Umat = rots[rot_idx]
                C.set_U(Umat.ravel())
            gt_rots.append(np.reshape(C.get_U(), (3,3)))
            img = sim_utils.synthesize_cbf(
                SIM, C, Famp,
                dev_id=dev_id,
                xtal_size=xtal_size, outfile=outfile,
                background=water, just_return_img=outfile is None )
            if cbfdir is not None:
                with h5py.File(h5_file, "w") as h5:
                    h5.create_dataset("img", data=img)

        if rots_from_grid:
            this_ranks_rot_indices.append(rot_idx)
        this_ranks_imgs.append(np.array([img], np.float32))
        this_ranks_ishots.append(i_shot)

    gt_rots = np.array(gt_rots)

    rots = rots.astype(np.float32)
    O = probable_orients()
    max_num_strong_spots = 1000
    O.allocate_orientations(dev_id, rots.ravel(), max_num_strong_spots)
    O.Bmatrix = sim_const.CRYSTAL.get_B()

    this_ranks_prob_rot =[]
    this_ranks_signal_levels = []
    this_ranks_num_refs = []
    R = None
    nbad_shot = 0
    nref_on_bad_shot = []
    all_nref =[]
    for i_img, img in enumerate(this_ranks_imgs):
        t1 = time.time()
        water_file = None
        refl_file = None
        if cbfdir is not None:
            water_file = os.path.join(cbfdir, "water%d.npz" % this_ranks_ishots[i_img])
            refl_file =os.path.join(cbfdir, "strong%d.refl" % this_ranks_ishots[i_img])
        if add_water:
            has_water_file = water_file is not None and os.path.exists(water_file)
            has_refl_file = refl_file is not None and os.path.exists(refl_file)

            if use_precomputed and has_water_file and has_refl_file:
                R = flex.reflection_table.from_file(refl_file)
                radProData = np.load(water_file)
                radPro = radProData["radPro"][()]
                all_Qbins = radProData["all_Qbins"][()]
                img_sh = radProData["img_sh"][()]
                img = img*correction
                bgImage = radProMaker.expand_background_1d_to_2d(radPro, img_sh, all_Qbins)
                img -= bgImage
                this_ranks_imgs[i_img] = img
            else:
                t = time.time()
                R = utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM, phil_file=phil_file)
                img = img*correction
                radPro = radProMaker.makeRadPro(
                    data_pixels=img,
                    strong_refl=R,
                    apply_corrections=False, use_median=True)

                bgImage = radProMaker.expand_radPro(radPro)
                img -= bgImage
                this_ranks_imgs[i_img]= img
                t = time.time()-t
                print0("Water sub took %f sec" % (t))
                if cbfdir is not None:
                    np.savez(water_file, radPro=radPro, img_sh=radProMaker.img_sh, all_Qbins=radProMaker.all_Qbins)
                    R.as_file(refl_file)

        else:
            R = db_utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM)
        #if R is not None and len(R) > 0:
        #    signal_level = utils.signal_level_of_image(R, img)
        #else:
        signal_level = np.percentile(img, 99.9)
        this_ranks_signal_levels.append(signal_level)

        prob_rot_file = None
        if cbfdir is not None:
            prob_rot_file = os.path.join(cbfdir, "prob_rot%d.npy" % this_ranks_ishots[i_img])

        if use_precomputed and cbfdir is not None and os.path.exists(prob_rot_file):
            prob_rot = np.load(prob_rot_file)
        else:
            if len(R) > 0:
                qvecs = db_utils.refls_to_q(R, sim_const.DETECTOR, sim_const.BEAM)
                qvecs = qvecs.astype(O.array_type)
                prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, False)
            else:
                prob_rot = np.array([], np.int32)
            if cbfdir is not None:
                np.save(prob_rot_file, prob_rot)
        t1 = time.time()-t1
        nR = -1
        if R is not None:
            nR = len(R)
            all_nref.append(nR)
            this_ranks_num_refs.append(nR)
        print0("%d probable rots on shot %d / %d with %d strongs (%f sec)" % ( len(prob_rot),i_img+1, len(this_ranks_imgs) , nR, t1) )

        if rots_from_grid:
            rot_idx = this_ranks_rot_indices[i_img]
            if rot_idx not in prob_rot:
                nbad_shot += 1
                nref_on_bad_shot.append(len(R))
            #assert rot_idx in prob_rot
        #TODO: add an else statement and assert prob_rot is "close" to the crystal Umat for that shot
        this_ranks_prob_rot.append(prob_rot)
    O.free_device()
    nbad_shot = COMM.reduce(nbad_shot)
    all_nref = COMM.reduce(all_nref)
    nref_on_bad_shot = COMM.reduce(nref_on_bad_shot)
    if COMM.rank==0:
        if nbad_shot > 0:
            print("mean, min, max number of ref on bad shots=%.1f,%.1f,%.1f"
                  % (np.mean(nref_on_bad_shot), min(nref_on_bad_shot), max(nref_on_bad_shot)))
            print("Number of shots with rot_idx not in prob_rot list=%d" % nbad_shot, flush=True)
        print("mean number of ref on all shots=%.1f" % np.mean(all_nref))

    Winit = np.zeros(const.DENSITY_SHAPE, np.float32)

    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    corner, deltas = utils.corners_and_deltas(const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
    qcoords = np.vstack([qx,qy,qz]).T
    maxRotInds = 20000

    L = lerpy()
    rots = rots.astype(L.array_type)
    Winit = Winit.astype(L.array_type)
    qcoords = qcoords.astype(L.array_type)
    for i,img in enumerate(this_ranks_imgs):
        this_ranks_imgs[i] = img.astype(L.array_type).ravel()

    L.allocate_lerpy(dev_id, rots.ravel(), Winit.ravel(), 2463*2527,
                     corner, deltas, qcoords.ravel(),
                     maxRotInds, 2463*2527)

    L.toggle_insert()
    if not start_with_relp:
        for i_img, (img, rot_inds) in enumerate(zip(this_ranks_imgs, this_ranks_prob_rot)):
            printR("Inserting %d rots for img %d / %d" % (len(rot_inds), i_img+1, len(this_ranks_imgs)))
            for r in rot_inds:
                L.trilinear_insertion(r, img, verbose=False)
        den = COMM.bcast(COMM.reduce(L.densities()))
        wts = COMM.bcast(COMM.reduce(L.wts()))
        den = utils.errdiv(den, wts)
        L.update_density(den)
        np.save(os.path.join(outdir, "Starting_density_insert"), den)
    else:
        Wstart = utils.get_W_init()
        scale_factor = max([img[img > 0].mean() for img in this_ranks_imgs])
        scale_factor = COMM.bcast(COMM.reduce(scale_factor, MPI.MAX))
        print0("Maximum pixel value=%f" % scale_factor)
        Wstart /= Wstart.max()
        Wstart *= scale_factor
        L.update_density(Wstart)
        np.save(os.path.join(outdir, "Starting_density_relp"), Wstart)

    # filter shots with 0 probable rot inds
    temp_shots = []
    temp_rot_indices = []
    temp_prob_rot = []
    temp_ishots = []
    temp_signal_levels = [] 
    for i_shot, shot in enumerate(this_ranks_imgs):
        prob_rot = this_ranks_prob_rot[i_shot]
        if prob_rot.size or this_ranks_num_refs[i_shot] >= MIN_REF_PER_SHOT:
            temp_shots.append(shot)
            temp_prob_rot.append(prob_rot)
            temp_ishots.append(i_shot)
            temp_signal_levels.append(this_ranks_signal_levels[i_shot])
            if this_ranks_rot_indices:
                temp_rot_indices.append(this_ranks_rot_indices[i_shot])
    this_ranks_imgs = temp_shots
    this_ranks_prob_rot = temp_prob_rot
    this_ranks_rot_indices = temp_rot_indices
    this_ranks_signal_levels = temp_signal_levels
    this_ranks_ishots = temp_ishots
    if not this_ranks_imgs:
        raise RuntimeError("at least one rank has 0 shots fit for processing (perhaps too few refls). Increase number of shots per rank (shots arg) until this error goes away! Alternatively slowly increase xtalsize and maybe alter hcut and minpred to allow for more probable orientations")

    all_ranks_signal_level = COMM.reduce(this_ranks_signal_levels)
    ave_signal_level = None
    if COMM.rank==0:
        ave_signal_level = np.mean(all_ranks_signal_level)
    ave_signal_level = COMM.bcast(ave_signal_level)

    beta_init = 1
    init_shot_scales = np.ones(len(this_ranks_imgs))
    if perturb_scale:
        init_shot_scales = np.random.uniform(1e-1, 1e3, len(this_ranks_imgs))

    inbounds = utils.qs_inbounds(qcoords, const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
    inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
    print0("INIT SHOT SCALES:", init_shot_scales)
    emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                        shot_mask=inbounds,
                        min_p=0,
                        outdir=outdir,
                        beta=beta_init,
                        shot_scales=init_shot_scales,
                        refine_scale_factors=refine_scale,
                        ave_signal_level=ave_signal_level)
    lp = LineProfiler()
    func_names, funcs = zip(*inspect.getmembers(mpi_utils.EMC, predicate=inspect.isfunction))
    for f in funcs:
        lp.add_function(f)
    RUN = lp(emc.do_emc)
    RUN(niter)
    stats = lp.get_stats()
    mpi_utils.print_profile(stats)

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
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("ndev", type=int, help="number of GPU devices per node")
    parser.add_argument("nshot", type=int, help="number of shots per rank")
    parser.add_argument("outdir", type=str, help="output folder" )
    parser.add_argument("--relp", action="store_true", help="init density starts from gaussians on relp points (see utils.get_W_init)")
    parser.add_argument("--perturbScale", action="store_true", help="perturb the scale factors before merging")
    parser.add_argument("--nogrid", action="store_true", help="ground truth rotations do not lie on rotation grid")
    parser.add_argument("--optscale", action="store_true", help="refine scale factors for each shot")
    parser.add_argument("--useprecomputed", action="store_true", help="load cbf images and probable rots from the cbfdir")
    parser.add_argument("--water", action="store_true", help="add water to sim")
    parser.add_argument("--niter", type=int, default=100, help="number of emc iterations")
    parser.add_argument("--phil", type=str, default=None, help="path to a stills process phil file (for spot finding). Required if --water flag is used")
    parser.add_argument("--minpred", type=int, default=7, help="minimum number of strong spots that need to be predicted well by an orientation for it to be flagged as probable for the shots crystal")
    parser.add_argument("--hcut", type=float, default=0.03, help="maximum distance (in hkl units) to Bragg peaks from prediction for prediction Bragg peak to be considered part of the lattice ")
    parser.add_argument("--xtalsize", type=float, default=0.002, help="size of the crystallite in mm")
    parser.add_argument("--cbfdir", type=str, default=None, help="if not None , store CBFs here")
    args = parser.parse_args()
    ndev = sys.argv[1]
    if args.water:
        assert args.phil is not None, "To detect peaks in presence of background water, phil file is required"
    test_emc_iteration(int(ndev), nshots_per_rank=args.nshot,
                       start_with_relp=args.relp, rots_from_grid=not args.nogrid,
                       outdir=args.outdir, add_water=args.water, niter=args.niter,
                       phil_file=args.phil, min_pred=args.minpred, hcut=args.hcut,
                       cbfdir=args.cbfdir, xtal_size=args.xtalsize, use_precomputed=args.useprecomputed,
                       refine_scale=args.optscale, perturb_scale=args.perturbScale)
