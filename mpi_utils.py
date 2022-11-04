
import os
import logging
import glob
import socket
import time
from copy import deepcopy
from itertools import groupby

from mpi4py import MPI
import sympy
from scipy.stats import pearsonr
import h5py
import numpy as np
from dials.array_family import flex
from xfel.merging.application.utils.memory_usage import get_memory_usage
from dxtbx.model import ExperimentList
from simtbx.diffBragg import mpi_logger
from simtbx.diffBragg.utils import image_data_from_expt


from simemc import utils
from simemc import emc_updaters
from simemc.sim_const import CRYSTAL
from simemc.compute_radials import RadPros

COMM = MPI.COMM_WORLD


def get_host_comm():
    """
    get an MPI communicator for all ranks sharing a specific host
    """
    HOST = socket.gethostname()
    unique_hosts = COMM.gather(HOST)
    HOST_MAP = None
    if COMM.rank==0:
        HOST_MAP = {HOST:i for i,HOST in enumerate(set(unique_hosts))}
    HOST_MAP = COMM.bcast(HOST_MAP)
    HOST_COMM = COMM.Split(color=HOST_MAP[HOST])
    return HOST_COMM


def get_host_dev_comm(dev_id=None):
    """
    get an MPI communicator for all ranks sharing a specific device on a specific host
    """
    HOST = socket.gethostname(), dev_id
    unique_hosts = COMM.gather(HOST)
    HOST_MAP = None
    if COMM.rank==0:
        HOST_MAP = {HOST: i for i,HOST in enumerate(set(unique_hosts))}
    HOST_MAP = COMM.bcast(HOST_MAP)
    HOST_COMM = COMM.Split(color=HOST_MAP[HOST])
    return HOST_COMM


def print0(*args, **kwargs):
    if COMM.rank==0:
        print("MPI-ROOT:", *args, **kwargs)


def print0f(*args, **kwargs):
    kwargs['flush'] = True
    if COMM.rank==0:
        print("MPI-ROOT: ", *args, **kwargs)


def printR(*args, **kwargs):
    print("rank%d"%COMM.rank,*args, **kwargs)


def printRf(*args, **kwargs):
    kwargs['flush'] = True
    print("rank%d"%COMM.rank,*args, **kwargs)


def make_dir(dirname):
    if COMM.rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    COMM.barrier()


def mpi_load_exp_ref(input_file, maxN=None, odd=False, even=False):
    exp_names, ref_names = utils.load_expt_refl_file(input_file)
    if maxN is not None:
        exp_names = exp_names[:maxN]
        ref_names = ref_names[:maxN]
    if odd or even:
        assert not (odd and even)
        start = 1 if odd else 0
        exp_names = exp_names[start::2]
        ref_names = ref_names[start::2]

    imgs, refls, names, crystals = [],[],[],[]
    for i,(exp_name, ref_name) in enumerate(zip(exp_names, ref_names)):

        if i % COMM.size != COMM.rank:
            continue
        explst = ExperimentList.from_file(exp_name)
        assert len(explst)==1
        img = image_data_from_expt(explst[0])
        crystal = explst[0].crystal  # these are optional, if no crystal is in explist, these will be None
        refl = flex.reflection_table.from_file(ref_name)
        imgs.append(img)
        refls.append(refl)
        names.append(exp_name)
        crystals.append(crystal)
        print0f("Done loading image %d / %d" % (i+1, len(exp_names)))
    return imgs, refls, names, crystals


def load_emc_input(input_dirs, dt=None, max_num=None, min_prob_ori=0, max_prob_ori=None):
    """

    :param input_dirs: list of emc input directories created by script bg_and_probOri.py
    :param dt: datatype to use for image pixels
    :param max_num: max number of images to load
    :param min_prob_ori: minimum number of probable orientations per image for that image to be loaded
    :param max_prob_ori: maximum number of probable orientations per image for that image to be loaded
    :return: an iterable, yielding (data, background, rot_inds, fname, i_shot, n_to_load )
        Here data is the image shot, background is the background for that shot, rot_inds is the list of
        probable orientation indices for that shot, fname is the input emc h5 filename containing the shot,
        i_shot is that h5 filenames internal index for the shot, n_to_load is the number of remaining images to load
    """
    inputNames_Shots = None

    if max_prob_ori is None:
        max_prob_ori = np.inf

    shots_to_load = {}
    all_inputs = []
    for dirname in input_dirs:
        input_fnames = glob.glob("%s/emc_input*.h5" % dirname)
        print0("Input dir %s has %d files" % (dirname, len(input_fnames)))
        all_inputs += input_fnames
    print0f("%d emc-input*.h5 files in total" % len(all_inputs))

    ntotal_shots = 0
    nshots_to_load = 0
    for i_f, f in enumerate(all_inputs):
        if i_f % COMM.size != COMM.rank:
            continue
        print0f("Examining emc-input %s (%d/%d)" % (f, i_f+1, len(all_inputs)))
        with h5py.File(f, 'r') as h:
            prob_inds = h['probable_rot_inds']
            nprob = np.array([len(p) for p in prob_inds])
            n_in_range = np.logical_and(nprob >= min_prob_ori, nprob <= max_prob_ori)
            ntotal_shots += len(prob_inds)
            nshots_to_load += n_in_range.sum()
            shots_to_load[f] = list(np.where(n_in_range)[0])
    nshots_to_load = COMM.reduce(nshots_to_load)
    ntotal_shots = COMM.reduce(ntotal_shots)
    if COMM.rank==0:
        perc_shots = nshots_to_load / ntotal_shots * 100.
        print0f("Total shots=%d; Total shots with %d <= n_prob_rot  <= %d = %d (%1.2f %%)"
                % (ntotal_shots, min_prob_ori, max_prob_ori, nshots_to_load, perc_shots))
    all_shots_to_load = COMM.gather(shots_to_load)
    if COMM.rank==0:
        temp = {}
        for shots_to_load in all_shots_to_load:
            for f in shots_to_load:
                temp[f] = shots_to_load[f]
        all_shots_to_load = temp
        # tally up total num images
        inputNames_Shots = []
        for f, shot_inds in all_shots_to_load.items():
            N = len(shot_inds)
            inputNames_Shots += list(zip([f]*N, list(shot_inds)))

        if max_num is not None:
            assert max_num <= len(inputNames_Shots)
            inputNames_Shots = inputNames_Shots[:max_num]
        ntot = len(inputNames_Shots)
        printR("There are %d total expst to load" % ntot, flush=True)
    inputNames_Shots = COMM.bcast(inputNames_Shots)
    inputNames_Shots_rank = np.array_split(inputNames_Shots, COMM.size)[COMM.rank]
    # note: the previous call to array_split converts shot index from int to str! so undo:
    inputNames_Shots_rank = [(input_h5, int(shot))
                             for input_h5, shot in inputNames_Shots_rank]

    my_open_h5,_ = map(set, zip(*inputNames_Shots_rank))
    printR("Need to open %d input emc files" % len(my_open_h5), flush=True)
    name_to_h5 = {}
    for fname in my_open_h5:
        name_to_h5[fname] = {}
        h = h5py.File(fname,"r")
        # these should all be the same in all input files..
        Qbins = h["all_Qbins"][()]
        img_sh = h['background_img_sh'][()]
        name_to_h5[fname]['handle'] = h
        name_to_h5[fname]["Qbins"] = Qbins
        name_to_h5[fname]["img_sh"] = img_sh
        name_to_h5[fname]["correction"] = h['correction'][()]

    n_to_load = len(inputNames_Shots_rank)
    for fname, i_shot in inputNames_Shots_rank:
        h5 = name_to_h5[fname]['handle']

        # get the data
        expt_name = h5["expts"][i_shot]
        El = ExperimentList.from_file(expt_name)
        data = image_data_from_expt(El[0])
        minSig, maxSig = El[0].detector[0].to_dict()['trusted_range']
        trusted = np.logical_and(data > minSig, data < maxSig)
        # get the background
        radial = h5["background"][i_shot]
        background = RadPros.expand_background_1d_to_2d(
            radial,
            img_sh=name_to_h5[fname]["img_sh"],
            all_Qbins=name_to_h5[fname]["Qbins"]
        )
        background = background.ravel()

        # get the correction
        data *= name_to_h5[fname]["correction"]
        rot_inds = h5["probable_rot_inds"][i_shot].astype(np.int32)

        if dt is not None:
            data = data.astype(dt)
            background = background.astype(dt)
        data = np.ascontiguousarray(data).ravel()
        n_to_load = n_to_load -1
        yield data, background, trusted, rot_inds, fname, i_shot, n_to_load


def setup_profile_log_files(logdir, name='simemc.profile', overwrite=True):
    make_dir(logdir)

    mpi_logger._make_logger(name,
                 os.path.join(logdir, mpi_logger.HOST+"-profile"),
                 level=logging.INFO,
                 overwrite=overwrite,
                 formatter=logging.Formatter(mpi_logger.SIMPLE_FORMAT))


def setup_rank_log_files(logdir, name, overwrite=True, ext="log"):
    make_dir(logdir)
    logger = mpi_logger._make_logger(name,
                        os.path.join(logdir, mpi_logger.HOST+"_simemc.%s" % ext),
                        level=logging.DEBUG,
                        overwrite=overwrite,
                        formatter=logging.Formatter(mpi_logger.SIMPLE_FORMAT))
    return logger


def print_profile(stats, timed_methods=None):

    #PROFILE_LOGGER = logging.getLogger(loggerName)
    for method in stats.timings.keys():
        filename, header_ln, name = method
        if timed_methods is not None and name not in timed_methods:
            continue
        info = stats.timings[method]
        printR("\n")
        printR("FILE: %s --- FUNC: %s" % (filename, name))
        if not info:
            continue
        unit = stats.unit

        line_nums, ncalls, timespent = zip(*info)
        fp = open(filename, 'r').readlines()
        total_time = sum(timespent)
        header_line = fp[header_ln-1][:-1]
        printR(header_line)
        printR("TOTAL FUNCTION TIME: %f ms" % (total_time*unit*1e3))
        printR("<><><><><><><><><><><><><><><><><><><><><><><>")
        printR("%5s%14s%9s%10s" % ("Line#", "Time", "%Time", "Line" ))
        printR("%5s%14s%9s%10s" % ("", "(ms)", "", ""))
        printR("<><><><><><><><><><><><><><><><><><><><><><><>")
        for i_l, l in enumerate(line_nums):
            frac_t = timespent[i_l] / total_time * 100.
            line = fp[l-1][:-1]
            printR("%5d%14.2f%9.2f%s" % (l, timespent[i_l]*unit*1e3, frac_t, line))


def bcast_large(v, sz=None, verbose=False, comm=None):
    """
    reduce /broadcast large numpy arrays
    :param v: numpy array with lots of elements e.g. np.random.random((700,700,700))
    :param sz: reduce v this many elements at a time
    :param verbose: print statements, to see what slow step is
    :param comm: mpi communicator
    :return: broadcast array
    """
    if comm is None:
        comm = COMM  # default is global comm
    v_sh = nchunk = nv = None
    if sz is None:
        sz = 32**3
    if comm.rank==0:
        v_sh = v.shape
        v = v.ravel()
        nchunk = int(np.ceil(len(v) / sz))
        nv = len(v)

    nv = comm.bcast(nv)
    v_sh = comm.bcast(v_sh)
    nchunk = comm.bcast(nchunk)
    v_slices = []

    for i_chunk, i_rng in enumerate(np.array_split(np.arange(nv), nchunk)):
        if verbose:
            print0f("bcast slice %d / %d" %(i_chunk+1, nchunk), end="\r")

        start = i_rng[0]
        stop = i_rng[-1]+1
        v_slc = None
        if comm.rank==0:
            v_slc = np.array(v[start:stop])
        comm.barrier()
        v_slc = comm.bcast(v_slc)
        comm.barrier()
        v_slices.append(v_slc)
    if verbose:
        print0f("")
    v_bcast = np.hstack(v_slices)
    v_bcast = v_bcast.reshape(v_sh)
    return v_bcast


def reduce_large_3d(v, verbose=False, buffers=False):
    """
    reduce /broadcast large numpy arrays
    :param v: numpy array with lots of elements e.g. np.random.random((700,700,700))
    :param verbose: print statements, to see what slow step is
    :param buffers: depending on system and size of array, buffers will speed things up
        NOTE: buffers has given inconsistent results in some settings.. see MPI.Reduce method emphasis on capital R
    :return: reduced array. return value is None for ranks>0
    """
    v_sh = v.shape
    if v.dtype==np.float64:
        dt = MPI.DOUBLE
    elif v.dtype==np.float32:
        dt = MPI.FLOAT
    else:
        assert v.dtype==int
        dt = MPI.INT
    v_slices = []
    for i in range(v.shape[0]):
        v_slc_rank = v[i]
        if verbose:
            print0f("reduce slice %d / %d" %(i+1, v.shape[0]), end="\r")
        COMM.barrier()
        if buffers:
            v_slc = np.empty_like(v_slc_rank, v.dtype)
            COMM.Reduce([v_slc_rank,dt], [v_slc, dt])
        #if broadcast:
        #    sh = None if COMM.rank>0 else v_slc.shape
        #    sh = COMM.bcast(sh)
        #    v_slc = COMM.bcast(v_slc.ravel()).reshape(sh)
        #    #COMM.Bcast([v_slc, dt])
        v_slices.append(v_slc)
    if verbose:
        print0f("")
    v_red = None
    if not np.any([slc is None for slc in v_slices]):
        v_red = np.array(v_slices)
    return v_red


def reduce_large(v, sz=None, broadcast=True, verbose=False, buffers=False):
    """
    reduce /broadcast large numpy arrays
    :param v: numpy array with lots of elements e.g. np.random.random((700,700,700))
    :param sz: reduce v this many elements at a time
    :param broadcast: broadcast result to all ranks
    :param verbose: print statements, to see what slow step is
    :param buffers: depending on system and size of array, buffers will speed things up
        NOTE: buffers has given inconsistent results in some settings.. see MPI.Reduce method emphasis on capital R
    :return: reduced array. If broadcast is False, return value is None for ranks>0
    """
    v_sh = v.shape
    if v.dtype==np.float64:
        dt = MPI.DOUBLE
    elif v.dtype==np.float32:
        dt = MPI.FLOAT
    else:
        assert v.dtype==int
        dt = MPI.INT
    if verbose:
        print0f("ravel")
    v = v.ravel()
    if sz is None:
        sz = 32**3
    v_slices = []
    nchunk = int(np.ceil(len(v) / sz))

    for i_chunk, i_rng in enumerate(np.array_split(np.arange(len(v)), nchunk)):
        if verbose:
            print0f("reduce slice %d / %d" % (i_chunk+1, nchunk), end="\r")

        start = i_rng[0]
        stop = i_rng[-1]+1
        v_slc_rank = np.array(v[start:stop])
        COMM.barrier()

        if buffers:
            v_slc = np.empty_like(v_slc_rank, v.dtype)
            COMM.Reduce([v_slc_rank,dt], [v_slc, dt])
        else:
            v_slc = COMM.reduce(v_slc_rank)
        if broadcast:
            v_slc = COMM.bcast(v_slc)
        v_slices.append(v_slc)
    if verbose:
        print0f("")
    v_red = None
    if not np.any([slc is None for slc in v_slices]):
        v_red = np.hstack(v_slices)
        v_red = v_red.reshape(v_sh)
    return v_red


class EMC:

    def __init__(self, L, shots, prob_rots, peak_mask, shot_background=None, shot_mask=None, min_p=0, outdir=None, beta=1,
                 symmetrize=True, img_sh=None, shot_scales=None,
                 refine_scale_factors=False, ave_signal_level=1, scale_update_method="analytical",
                 density_update_method="analytical", ucell_p=None, shot_names=None, symbol=None, max_iter=60):
        """
        run emc, the resulting density is stored in the L (emc.lerpy) object
        see call to this method in ests/test_emc_iteration
        :param L: instance of emc.lerpy in insert mode
        :param shots: array of images (Nshot, Npanel , slowDim , fastDim) or (Nshot, Npanel x slowDim x fastDim)
        :param prob_rots: list of array containing indices of probable orientations
        :param peak_mask:
        :param shot_mask: boolean np.ndarray (same shape as each shot in shots), corresponding to the masked pixels (True is a trusted pixel, False is masked)
        :param beta: float , controls the probability distribution for orientations. lower beta makes more orientations probable
        :param symmetrize: bool, update W after each iteration according to the space group symmetry operators
        :param shot_scales: list of floats, one per shot, same length as shots and prob_rots. Scale factor applied to
            tomograms for each shot
        :param refine_scale_factors: bool, update scale factors with 3d density
        :param scale_update_method: str, can be either 'analytical' or  'bfgs'. The latter uses L-BFGS, the former
            seeks the minimum directly
        :param density_update_method: str, can be either 'analytical' or  'line_search'. The latter uses numerical optimization
            (see emc_updaters.DensityUpdater class)
        :ucell_p: unit cell parameters e.g. 78,78,38,90,90,90  (a,b,c,alpha,beta,gamma in angstrom / degrees)
        :shot_names: names corresponding to the images in shots, for record keeping, these will be written to some outputfiles
        :symbol: space group symbol e.g. P43212
        :param max_iter: for density updater method lbfgs: maximum number of iterations per minimization cycle

        """

        self.LOGGER = logging.getLogger(utils.LOGNAME)
        self.peak_mask = peak_mask

        if ucell_p is None:
            self.ucell_p = CRYSTAL.get_unit_cell().parameters()
        else:
            assert len(ucell_p) == 6
            self.ucell_p = ucell_p

        if symbol is None:
            symbol = CRYSTAL.get_space_group().info().type().lookup_symbol()
        self.symbol=symbol

        self.max_iter = max_iter

        self.shot_names = shot_names  # optional names identifying each shot (for book-keeping)
        assert len(shots) > 0
        assert isinstance(shots[0], np.ndarray)
        assert scale_update_method in ["analytical", "bfgs"]
        assert density_update_method in ["analytical", "line_search", "lbfgs"]
        assert len(prob_rots) == len(shots)
        if shot_mask is not None:
            assert isinstance(shot_mask, np.ndarray)
            assert shot_mask.shape == shots[0].shape
            assert shot_mask.dtype == bool
            self.shot_mask = shot_mask
        else:
            self.shot_mask = np.ones(shots[0].shape, dtype=bool)
        if shot_background is not None:
            assert len(shot_background) == len(shots)
            assert isinstance(shot_background[0], np.ndarray)
            self.shot_background = shot_background
        else:
            self.shot_background = [np.zeros_like(shots[0]) for _ in range(len(shots))]

        self.scale_update_method = scale_update_method
        self.density_update_method = density_update_method

        self.ave_signal_level=ave_signal_level
        self.L = L  # lerpy instance
        self.L.rank = COMM.rank
        self.L.set_sym_ops(self.ucell_p, self.symbol)
        self.shots = shots
        self.max_scale = 1e6
        self.shot_sums = np.zeros(len(self.shots))
        for i,s in enumerate(self.shots):
            if s.dtype != L.array_type:
                s = s.astype(L.array_type)
            self.shots[i] = s.ravel()

        self.prob_rots = prob_rots
        for i,rots in enumerate(self.prob_rots):
            if rots.dtype != np.int32:
                rots = rots.astype(np.int32)
            self.prob_rots[i] = rots

        self.scale_changed = None
        if shot_scales is None:
            self.shot_scales = np.ones(len(shots))
        else:
            assert len(shots) == len(shot_scales)
            self.shot_scales = np.array(shot_scales)
        if refine_scale_factors:
            self.shot_scales *= self.ave_signal_level / self.shot_scales.mean()

        self.nshots = len(shots)
        self.min_p = min_p
        self.P_dr = None
        self.finite_rot_inds = None
        self.i_shot = None
        self.send_to = None
        self.recv_from = None
        self.rot_inds_on_one_rank_only = None
        if img_sh is None:
            self.img_sh = 1, 2527, 2463
        else:
            self.img_sh = img_sh
        self.num_data_pix = int(np.product(self.img_sh))
        self.symmetrize = symmetrize
        self.shot_P_dr = []
        self.outdir = outdir
        self.beta = beta
        self.all_Wresid = []
        self.nshot_tot = COMM.bcast(COMM.reduce(self.nshots))
        self.success_corr = 0.2
        self.i_emc = 0
        self.save_model_freq = 10
        self.dens_sh = self.L.dens_sh
        self.ave_time_per_iter = 0
        self.refine_scale_factors = refine_scale_factors
        self.update_scale_factors = False  # if the specific iteration is updating the per-shot scale factors
        self.update_density = True  # if the specific iteration is updating the density

        self.qs_inbounds = utils.qs_inbounds(self.L.qvecs.reshape((-1,3)), self.dens_sh, self.L.xmin, self.L.xmax)
        if COMM.rank==0:
            self.apply_density_rules()
        self.L.bcast_densities(COMM)

        for i,s in enumerate(self.shots):
            shot_mask_in_bounds = self.shot_mask[self.qs_inbounds]
            shot_in_bounds = s[self.qs_inbounds]
            bounded_unmasked_pixels = shot_in_bounds[shot_mask_in_bounds]
            self.shot_sums[i] = bounded_unmasked_pixels.sum()

        if outdir is not None:
            make_dir(outdir)
            if COMM.rank==0:
                init_density_file = os.path.join(self.outdir, "Winit.h5")
                self.save_h5(init_density_file)

    def add_records(self):
        for i_rot, rot_ind in enumerate(self.prob_rots[self.i_shot]):
            P_dr = self.P_dr[i_rot]
            if P_dr > 0:
                self.finite_rot_inds.add_record(rot_ind, self.i_shot, COMM.rank, P_dr)

    def distribute_rot_ind_records(self):
        if COMM.rank==0:
            rot_inds_global = deepcopy(self.finite_rot_inds)
            for rank in range(1,COMM.size):
                rot_inds = COMM.recv(source=rank)
                rot_inds_global.merge(rot_inds)

            self.send_to, self.recv_from = rot_inds_global.tomogram_sends_and_recvs()
            self.rot_inds_on_one_rank_only = rot_inds_global.on_one_rank

        else:
            COMM.send(self.finite_rot_inds, dest=0)

        self.send_to = COMM.bcast(self.send_to)
        self.recv_from = COMM.bcast(self.recv_from)
        self.rot_inds_on_one_rank_only = COMM.bcast(self.rot_inds_on_one_rank_only)

    def insert_one_rankers(self):
        """
        for rotations that are only ever sampled by one rank, then the trilinear insertion can be done
        without MPI comms
        """
        for i_rot_ind, rot_ind in enumerate(self.finite_rot_inds):
            if rot_ind not in self.rot_inds_on_one_rank_only:
                continue
            self.print("Finite rot ind %d / %d" %(i_rot_ind+1,len(self.finite_rot_inds) ))
            W_rt = np.zeros(self.num_data_pix, self.L.array_type)
            # NOTE: for now assuming all shots have same mask, so no need to compute normalization term mask_rt

            P_dr_sum_scaled = 0
            for i_shot, _, P_dr in self.finite_rot_inds.iter_record(rot_ind):
                W_rt += self.shots[i_shot]*P_dr
                P_dr_sum_scaled += P_dr*self.shot_scales[i_shot]
            W_rt = utils.errdiv(W_rt, P_dr_sum_scaled)
            self.L.trilinear_insertion(rot_ind, W_rt, self.shot_mask)

    def insert_multi_rankers(self):
        """
        if a rotation is sampled by more than one rank, then we need to insert the tomograms and the weights
        across all the ranks that sample that rotation.
        One rank is chosen to do trilinear insertion, and the relevant shot data from other ranks are sent to chosen rank
        """
        send_req = []
        if COMM.rank in self.send_to:
            for dest, i_data, tag in self.send_to[COMM.rank]:
                req = COMM.isend((self.shots[i_data], self.shot_scales[i_data]), dest=dest, tag=tag)
                send_req.append(req)

        if COMM.rank in self.recv_from:
            rot_inds_to_recv = self.recv_from[COMM.rank]['rot_inds']
            rot_inds_recv_info = self.recv_from[COMM.rank]['comms_info']
            for i_recv, (rot_ind, recv_info) in enumerate(zip(rot_inds_to_recv, rot_inds_recv_info)):
                self.print("recv %d (%d / %d)" % (rot_ind, i_recv+1, len(rot_inds_to_recv)))
                W_rt = np.zeros(self.num_data_pix, self.L.array_type)
                # NOTE: for now assuming all shots have same mask, so no need to compute normalization term mask_rt
                P_dr_sum_scaled = 0
                for source, P_dr, tag in recv_info:
                    assert source != COMM.rank, "though supported we dont permit self sending"
                    self.print("get from rank %d, tag=%d" % (source, tag))
                    shot_data = COMM.recv(source=source, tag=tag)
                    shot_img, shot_scale = shot_data
                    self.print("GOT from rank %d, tag=%d" % (source, tag))
                    W_rt += shot_img.ravel()*P_dr
                    P_dr_sum_scaled += P_dr*shot_scale

                self.print("Done recv %d /%d" %(i_recv+1, len(rot_inds_to_recv)))
                assert rot_ind in self.finite_rot_inds # this is True by definition if you read the RotInds class method above
                for i_data, rank, P_dr in self.finite_rot_inds.iter_record(rot_ind):
                    W_rt += self.shots[i_data] * P_dr
                    P_dr_sum_scaled += P_dr*self.shot_scales[i_data]
                W_rt = utils.errdiv(W_rt, P_dr_sum_scaled)
                self.L.trilinear_insertion(rot_ind, W_rt, self.shot_mask)
        for req in send_req:
            req.wait()
        self.print("Done with tomogram send/recv")

    def reduce_density(self):
        self.print("Waiting for other ranks to catch up before reducing")
        COMM.barrier()
        self.print("Reducing density")
        self.L.reduce_densities()
        self.print("wts reduce")
        self.L.reduce_wts()

        self.LOGGER.debug("Update density (i_emc=%d)" %self.i_emc)
        den = None
        if COMM.rank==0:
            den = utils.errdiv(self.L.densities(), self.L.wts())
            self.L.update_densities(den)
            self.apply_density_rules()
        self.L.bcast_densities(COMM)


    def set_new_density(self, den):
        pass

    def apply_density_rules(self):
        if self.symmetrize:
            self.print("Applying crystal symmetry")
            self.LOGGER.debug("Applying crystal symmetry (i_emc=%d)" % self.i_emc)
            self.L.symmetrize()
            self.print("Applying Friedel symmetry")
            self.LOGGER.debug("Applying friedel symmetry (i_emc=%d)" % self.i_emc)
            self.L.symmetrize()
            self.L.apply_friedel_symmetry(self.peak_mask)

    def prep_for_insertion(self):
        self.L.toggle_insert()

    def count_scale_factors_at_limit(self):
        num_at_min = np.allclose(self.shot_scales, 0)
        num_at_max = np.allclose(self.shot_scales, self.max_scale)
        num_at_min = COMM.reduce(num_at_min)
        num_at_max = COMM.reduce(num_at_max)
        if COMM.rank==0:
            self.print("%d scales at 0, %d scales at max (1e6) (%d total)" % (num_at_min, num_at_max, self.nshot_tot))

    def count_finite_inds_per_shot(self):
        seen_shots = {}
        for i in range(self.nshots):
            seen_shots[i] = 0
        for rot_ind in self.finite_rot_inds:
            for i_shot,_,_ in self.finite_rot_inds.iter_record(rot_ind):
                seen_shots[i_shot] += 1
        num_per_shot = list(seen_shots.values())
        num_per_shot = COMM.reduce(num_per_shot)
        if COMM.rank==0:
            num_zero = sum(np.array(num_per_shot)==0)
            ave_pos = np.mean(num_per_shot).mean()
            self.print("Number of shots with 0 finite rot inds= %.2f. Mean num rot inds=%.2f for shots with finite rot inds (total shots=%d)" \
                       % (num_zero, ave_pos, self.nshot_tot))

    def do_emc(self, num_iter):
        self.i_emc = 0
        iter_times = []
        while self.i_emc < num_iter:
            t = time.time()
            print_cpu_mem_usage(self.LOGGER)
            print_cpu_mem_usage()
            print_gpu_usage_across_ranks(self.L)

            self.update_scale_factors = self.i_emc > 4 and (self.i_emc % 2 == 1) and self.refine_scale_factors
            self.update_density = not self.update_scale_factors

            self.finite_rot_inds = utils.RotInds()  # empty dictionary container for storing finite rot in info
            self.shot_P_dr = []
            log_R_per_shot = self.log_R_dr()
            for self.i_shot in range(self.nshots):
                log_R_dr = log_R_per_shot[self.i_shot]
                self.P_dr = self.get_P_dr(log_R_dr)
                self.shot_P_dr.append(self.P_dr)
                self.add_records()
                self.LOGGER.debug("Done with shot %d / %d Pdr" % (self.i_shot+1, self.nshots))
            self.LOGGER.debug("Done with P_dr step")

            if self.update_scale_factors:
                self.LOGGER.debug("Beginning scale update step")
                self.print("**Updating scale factors**")
                updater = emc_updaters.ScaleUpdater(self)
                if self.scale_update_method == 'analytical':
                    self.scale_changed = updater.update(analytical=True)
                else:
                    updater.update(bfgs=True, reparam=True, max_scale=self.max_scale)
                norm_factor = self.ave_signal_level / np.mean(self.shot_scales)
                self.print("Normalizing scale factors... Multiplying all scale factors by %f" % norm_factor)
                self.shot_scales *= norm_factor

                self.i_emc += 1
                t = time.time()-t
                iter_times.append(t)
                self.ave_time_per_iter = np.mean(iter_times)
                self.LOGGER.debug("Done with scale update step")
                continue

            if self.density_update_method == "analytical":
                self.distribute_rot_ind_records()
                self.L.toggle_insert()
                self.insert_one_rankers()
                self.insert_multi_rankers()
                self.reduce_density()
            elif self.density_update_method in ["line_search", "lbfgs"]:
                self.LOGGER.debug("Beginning numerical density update step")
                density_updater = emc_updaters.DensityUpdater(self)
                self.LOGGER.debug("Instantiated density updater...")
                den = density_updater.update(how=self.density_update_method, lbfgs_maxiter=self.max_iter)
                self.set_new_density(den)
                self.LOGGER.debug("Update density (i_emc=%d)" % self.i_emc)
                if COMM.rank==0:
                    assert den is not None
                    self.L.update_density(den.ravel())
                    self.apply_density_rules()
                self.L.bcast_densities(COMM)
            else:
                raise NotImplementedError("Unknown method %s" % self.density_update_method)

            self.LOGGER.debug("Done with density update step")
            self.count_scale_factors_at_limit()
            self.count_finite_inds_per_shot()
            self.write_to_outdir()

            self.i_emc += 1
            t = time.time()-t
            iter_times.append(t)
            self.ave_time_per_iter = np.mean(iter_times)
            exit()

    def log_R_dr(self, deriv=False):
        shot_log_R_dr = []
        shot_deriv_logR = []
        for i_shot, (img, rot_inds, scale_factor) in enumerate(zip(self.shots, self.prob_rots, self.shot_scales)):
            self.print("Getting Rdr %d / %d (scale=%f)" % ( i_shot+1, self.nshots, scale_factor))
            bg_img = self.shot_background[i_shot]
            self.L.copy_image_data(img.ravel(), self.shot_mask, bg_img)
            self.L.equation_two(rot_inds, False, scale_factor)
            log_R_dr_vals = np.array(self.L.get_out())
            shot_log_R_dr.append(log_R_dr_vals)

            if deriv:
                self.L.equation_two(rot_inds, False, scale_factor, deriv=True)
                deriv_log_R_dr = np.array(self.L.get_out())
                shot_deriv_logR.append(deriv_log_R_dr)
        if not deriv:
            return shot_log_R_dr
        else:
            return shot_log_R_dr, shot_deriv_logR

    def get_P_dr(self, log_R_dr):
        """
        Use arbitrary precision library to get normalized probs
        :param log_R_dr:
        :return:
        """
        R_dr = []
        R_dr_sum = sympy.S(0)
        for val in log_R_dr:
            r = sympy.exp(sympy.S(val)) ** self.beta
            R_dr.append(r)
            R_dr_sum += r

        P_dr = []
        for r in R_dr:
            p = r / R_dr_sum
            p = float(p)
            P_dr.append(p)

        P_dr = np.array(P_dr)
        y = P_dr.sum()
        P_dr[P_dr < self.min_p] = 0
        x = utils.errdiv(y, P_dr.sum())
        P_dr *= x
        return P_dr

    def print(self, *args, **kwargs):
        update_s = "fixed W" if self.update_scale_factors else "fixed PHI"
        print0f("[EmcStp %d; sec/it=%f; %s]" % (self.i_emc, self.ave_time_per_iter, update_s), *args, **kwargs)

    def write_to_outdir(self):
        self.print("Write to outputdir")
        if self.outdir is None:
            return
        make_dir(self.outdir)
        all_scales = COMM.gather(self.shot_scales)
        all_scale_changed = None
        if self.scale_changed is not None:
            all_scale_changed = COMM.gather(self.scale_changed)
        if COMM.rank==0:
            density_file = os.path.join(self.outdir, "Witer%d.h5" % (self.i_emc+1))
            all_scales = np.hstack(all_scales)
            np.save(os.path.join(self.outdir,"Scales%d" % (self.i_emc+1)), all_scales)
            if all_scale_changed is not None:
                all_scale_changed = np.hstack(all_scale_changed)
                np.save(os.path.join(self.outdir, "ScalesChanged%d" % (self.i_emc+1)), all_scale_changed)

            self.save_h5(density_file)

        self.save_prob_rots_npz()

    def save_h5(self, density_file):
        den = np.zeros(self.L.dens_dim**3)
        den[self.peak_mask.ravel()] = self.L.densities()
        with h5py.File(density_file, "w") as out_h5:
            NBINS = self.L.dens_dim
            out_h5.create_dataset("Wprime",data=den.reshape((NBINS, NBINS, NBINS)), compression="lzf")
            out_h5.create_dataset("ucell", data=self.ucell_p)
            out_h5.create_dataset("max_q", data=self.L.max_q)

    def save_prob_rots_npz(self):
        dirname = os.path.join( self.outdir , "prob_rots", "rank%d" % COMM.rank)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, "P_dr_iter%d.npz" % (self.i_emc+1))
        # TODO use dtype obj for jagged arrays
        np.savez(fname, rots=self.prob_rots, Pdr=self.shot_P_dr, names=self.shot_names)

    def success_rate(self, init=False, return_models=False):
        """
        currently unused, but in theory could be used to compare data with models, to estimate convergence
        :param init:
        :param return_models:
        :return:
        """
        self.print("Success rate")
        nsuccess = 0
        models = []
        wresid = 0
        for i_shot in range(self.nshots):
            if init:
                nrots = len(self.prob_rots[i_shot])
                pvals = np.ones(nrots) / float(nrots)
            else:
                pvals = self.shot_P_dr[i_shot]

            #i_max_p = np.argmax(pvals)
            #rot_ind_with_max_p = self.prob_rots[i_shot][i_max_p]
            psum = 0
            model = np.zeros_like(self.shots[0])
            for i_rot, p in enumerate(pvals):
                if p > 0.001:
                    rot_ind = self.prob_rots[i_shot][i_rot]
                    #model += self.L.trilinear_interpolation(rot_ind_with_max_p) * p
                    model += self.L.trilinear_interpolation(rot_ind) * p
                    psum += p
            model = utils.errdiv(model, psum)
            data = self.shots[i_shot]
            sel = data > 0
            c = pearsonr(data[sel], model[sel])[0]
            if c > self.success_corr:
                nsuccess += 1
            if return_models:
                models.append(model)
            wresid += np.sum((model.ravel()-data.ravel())[self.qs_inbounds]**2)
        wresid = COMM.bcast(COMM.reduce(wresid))
        #self.all_Wresid.append(wresid)
        nsuccess = COMM.reduce(nsuccess)
        if COMM.rank==0:
            success_rate = float(nsuccess) / self.nshot_tot * 100.
            print0("[ EmcStep%d ] SUCCESS RATE: %d/%d (%.2f %%); Wresid=%f" % (self.i_emc, nsuccess , self.nshot_tot, success_rate, wresid))
        if return_models:
            return models


def determine_rank_with_most_inserts(emc):
    total_inserts_this_rank = len(np.hstack(emc.prob_rots))
    total_inserts_each_rank = COMM.gather(total_inserts_this_rank)
    rank_with_most = None
    if COMM.rank==0:
        rank_with_most = np.argmax(total_inserts_each_rank)
    rank_with_most = COMM.bcast(rank_with_most)
    return rank_with_most, total_inserts_this_rank


def print_gpu_usage_across_ranks(L, logger=None):
    """

    :param L: lerpy instance (see emc_ext)
    :param logger: python logger
    :return:
    """
    free_mem = L.get_gpu_mem()/1024**3
    dev_id = L.dev_id
    all_data = COMM.gather([dev_id, free_mem])
    if COMM.rank==0:
        gb = groupby(sorted(all_data, key=lambda x:x[0]), key=lambda x:x[0])
        gpu_mem_info = {dev: [i for _,i in list(vals)] for dev, vals in gb}
        seen_hosts = set()
        for dev in gpu_mem_info:
            for free_mem in gpu_mem_info[dev]:
                host = socket.gethostname()
                if (dev, host) in seen_hosts:
                    continue
                if logger is not None:
                    logger.debug("device %d  on host %s has %.2f GB  free" % (dev,host, free_mem))
                else:
                    print("device %d  on host %s has %.2f GB  free" % (dev, host, free_mem))
                seen_hosts = seen_hosts.union({(dev,host)})


def print_cpu_mem_usage(logger=None):
    memMB = get_memory_usage()/1024
    host = socket.gethostname()
    all_data = COMM.gather([host, memMB])
    if COMM.rank==0:
        gb = groupby(sorted(all_data, key=lambda x:x[0]), key=lambda x:x[0])
        cpu_mem_info = {host: [i for _,i in list(vals)] for host, vals in gb}
        for host in cpu_mem_info:
            for peak_mem in cpu_mem_info[host]:
                if logger is not None:
                    logger.debug("host %s has used %.2f GB of memory" % (host, peak_mem))
                else:
                    print("host %s has used %.2f GB of memory" % (host, peak_mem))
                break


def ensure_same(den):
    """
    :param den:  numpy ndarray
    """
    req = COMM.isend(den, dest=0, tag=COMM.rank)
    if COMM.rank==0:
        for i_rank in range(COMM.size):
            other_rank_den = COMM.recv(source=i_rank, tag=i_rank)
            assert np.allclose(den, other_rank_den)
    req.wait()
