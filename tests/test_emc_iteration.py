
import pytest
import pylab as plt
import numpy as np
import os
from simtbx.diffBragg import utils as db_utils
from simemc import utils
from simemc import sim_const, sim_utils
from simemc import mpi_utils
from simemc.compute_radials import RadPros
from simemc.emc import lerpy, probable_orients

import const

print0 = mpi_utils.print0f
printR = mpi_utils.printRf


@pytest.mark.skip(reason="in development")
@pytest.mark.mpi(min_size=1)
def test_emc_iteration(ndev, nshots_per_rank=60, rots_from_grid=True, start_with_relp=False, outdir=None,
                       add_water=False, niter=100, phil_file=None, min_pred=7, hcut=0.03, cbfdir=None, xtal_size=0.002):
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
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
    nshot = COMM.size*nshots_per_rank
    print0("Simulating %d shots on %d ranks" % (nshot, COMM.size))
    rot_indices = np.random.permutation(rots.shape[0])[:nshot]

    water = 0
    radProMaker = None
    correction = 1
    #scale = 1
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

        C = sim_utils.random_crystal()
        if rots_from_grid:
            Umat = rots[rot_idx]
            C.set_U(Umat.ravel())
        gt_rots.append(np.reshape(C.get_U(), (3,3)))
        outfile = None
        if cbfdir is not None:
            outfile = os.path.join(cbfdir, "shot%d.cbf" % i_shot)
        img = sim_utils.synthesize_cbf(
            SIM, C, Famp,
            dev_id=dev_id,
            xtal_size=xtal_size, outfile=outfile,
            background=water, just_return_img=outfile is None )

        if rots_from_grid:
            this_ranks_rot_indices.append(rot_idx)
        this_ranks_imgs.append(np.array([img], np.float32))
    #COMM.barrier()
    #exit()

    gt_rots = np.array(gt_rots)


    rots = rots.astype(np.float32)
    O = probable_orients()
    max_num_strong_spots = 1000
    O.allocate_orientations(dev_id, rots.ravel(), max_num_strong_spots)
    O.Bmatrix = sim_const.CRYSTAL.get_B()

    this_ranks_prob_rot =[]
    for i_img, img in enumerate(this_ranks_imgs):
        if add_water:
            R = utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM, phil_file=phil_file)
            img = img*correction

            radPro = radProMaker.makeRadPro(
                data_pixels=img,
                strong_refl=R,
                apply_corrections=False, use_median=True)

            bgImage = radProMaker.expand_radPro(radPro)
            img -= bgImage
            this_ranks_imgs[i_img]= img

        else:
            R = db_utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM)

        qvecs = db_utils.refls_to_q(R, sim_const.DETECTOR, sim_const.BEAM)
        qvecs = qvecs.astype(O.array_type)
        prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, False)
        print0("%d probable rots on shot " % len(prob_rot))
        if rots_from_grid:
            rot_idx = this_ranks_rot_indices[i_img]
            assert rot_idx in prob_rot
        #TODO: add and else statement and assert prob_rot is "close" to the crystal Umat for that shot
        this_ranks_prob_rot.append(prob_rot)
    O.free_device()

    Winit = np.zeros(const.DENSITY_SHAPE, np.float32)

    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    corner, deltas = utils.corners_and_deltas(const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
    qcoords = np.vstack([qx,qy,qz]).T
    maxRotInds = 20000

    #L = lerpy()
    #L.allocate_lerpy(dev_id, gt_rots.ravel(), Wstart.ravel(), 2463*2527,
    #                 corner, deltas, qcoords.ravel(),
    #                 len(gt_rots), 2463*2527)

    #L.toggle_insert()
    #assert len(this_ranks_imgs) == len(gt_rots)
    #for i_rot, img in enumerate(this_ranks_imgs):
    #    L.trilinear_insertion(i_rot, img)
    #den= L.densities()
    #wt = L.wts()
    #den = COMM.bcast(COMM.reduce(den))
    #wt = COMM.bcast(COMM.reduce(wt))
    #gt_merge = utils.errdiv(den, wt)
    #if COMM.rank==0:
    #    np.save("gt_merge", gt_merge)

    #L.free()
    #del L
    #L = None

    L = lerpy()
    rots = rots.astype(L.array_type)
    Winit = Winit.astype(L.array_type)
    qcoords = qcoords.astype(L.array_type)
    for i,img in enumerate(this_ranks_imgs):
        this_ranks_imgs[i] = img.astype(L.array_type).ravel()

    L.allocate_lerpy(dev_id, rots.ravel(), Winit.ravel(), 2463*2527,
                     corner, deltas, qcoords.ravel(),
                     maxRotInds, 2463*2527)

    #if rots_from_grid:
    #    L.toggle_insert()
    #    for i_img, (img, rot_ind) in enumerate(zip(this_ranks_imgs, this_ranks_rot_indices)):
    #        print0("Inserting gt img %d / %d" % (i_img+1, len(this_ranks_imgs)))
    #        L.trilinear_insertion(rot_ind, img, False)
    #    den = COMM.bcast(COMM.reduce(L.densities()))
    #    wts = COMM.bcast(COMM.reduce(L.wts()))
    #    gt_den = utils.errdiv(den, wts)
    ##TODO add get_gt_den using ground truth crystal Umats (merge_gt method as a function)

    L.toggle_insert()
    if not start_with_relp:
        for i_img, (img, rot_inds) in enumerate(zip(this_ranks_imgs, this_ranks_prob_rot)):
            print0("Inserting img %d / %d" % (i_img+1, len(this_ranks_imgs)))
            #rot_inds = [this_ranks_rot_indices[i_img]]
            for r in rot_inds:
                L.trilinear_insertion(r, img, False)
        den = COMM.bcast(COMM.reduce(L.densities()))
        wts = COMM.bcast(COMM.reduce(L.wts()))
        den = utils.errdiv(den, wts)
        L.update_density(den)
    else:
        Wstart = utils.get_W_init()
        scale_factor = max([img[img > 0].mean() for img in this_ranks_imgs])
        scale_factor = COMM.bcast(COMM.reduce(scale_factor, MPI.MAX))
        print0("Maximum pixel value=%f" % scale_factor)
        Wstart /= Wstart.max()
        Wstart *= scale_factor
        L.update_density(Wstart)

    beta_init = 1
    emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                        min_p=1e-6,
                        outdir=outdir,
                        beta=beta_init)
    init_models = emc.success_rate(init=True, return_models=True)
    from line_profiler import LineProfiler
    import inspect
    lp = LineProfiler()
    func_names, funcs = zip(*inspect.getmembers(mpi_utils.EMC, predicate=inspect.isfunction))
    for f in funcs:
        lp.add_function(f)
    RUN = lp(emc.do_emc)
    RUN(100)
    stats = lp.get_stats()
    mpi_utils.print_profile(stats)

    #emc.do_emc(niter)
    #try:
    #    from line_profiler import LineProfiler
    #except ImportError:
    #    LineProfiler = None
    #if LineProfiler is not None:
    #    lp = LineProfiler()
    #    RUN = lp(main)
    #else:
    #    lp = None
    #    RUN = main

    #mpi_utils.setup_profile_log_files("logTest")

    #RUN()

    #if lp is not None:
    #    stats = lp.get_stats()
    #    mpi_utils.print_profile(stats,["main"], "simemc.profile")


    models = emc.success_rate(init=False, return_models=True)
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
    parser.add_argument("--nogrid", action="store_true", help="ground truth rotations do not lie on rotation grid")
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
                       cbfdir=args.cbfdir, xtal_size=args.xtalsize)
