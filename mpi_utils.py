from mpi4py import MPI
import time
from copy import deepcopy
from simemc import utils
import sympy
from simemc import const
from scipy.stats import pearsonr
import os
import h5py
from simtbx.diffBragg.refiners.parameters import Parameters, RangedParameter
from scipy.optimize import minimize
import glob
import numpy as np
from simtbx.diffBragg import mpi_logger
from scipy.optimize import fmin_l_bfgs_b

from dxtbx.model import ExperimentList
from simtbx.diffBragg.utils import image_data_from_expt
import pylab as plt


from simemc.compute_radials import RadPros
import logging

COMM = MPI.COMM_WORLD


def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)


def print0f(*args, **kwargs):
    kwargs['flush'] = True
    if COMM.rank==0:
        print(*args, **kwargs)


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


def load_emc_input(input_dir, dt=None, max_num=None):
    RENORM = 169895.59872560613 / 100
    inputNames_Shots = None
    if COMM.rank==0:
        all_inputs = glob.glob("%s/emc_input*.h5" % input_dir)
        # tally up total num images
        inputNames_Shots = []
        for f in all_inputs:
            h= h5py.File(f, 'r')
            expt_names = h['expts'][()]
            N = len(expt_names)
            inputNames_Shots += list(zip([f]*N, list(range(N))))
        if max_num is not None:
            assert max_num <= len(inputNames_Shots)
            inputNames_Shots = inputNames_Shots[:max_num]
        ntot = len(inputNames_Shots)
        printR("Found %d total expst to load" % ntot, flush=True)
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
        name_to_h5[fname]['renorm'] = 1e2  # TODO add renorm to hdf5
        name_to_h5[fname]["correction"] = h['omega'][()]*h["polar"][()]

    n_to_load = len(inputNames_Shots_rank)
    for fname, i_shot in inputNames_Shots_rank:
        h5 = name_to_h5[fname]['handle']

        # get the data
        expt_name = h5["expts"][i_shot]
        El = ExperimentList.from_file(expt_name)
        data = image_data_from_expt(El[0])

        # get the background
        radial = h5["background"][i_shot]
        background = RadPros.expand_background_1d_to_2d(
            radial,
            img_sh=name_to_h5[fname]["img_sh"],
            all_Qbins=name_to_h5[fname]["Qbins"]
        )

        # get the correction
        data *= name_to_h5[fname]["correction"]
        data *= RENORM
        rot_inds = h5["probable_rot_inds"][i_shot].astype(np.int32)

        if dt is not None:
            data = data.astype(dt)
        data = np.ascontiguousarray(data).ravel()
        n_to_load = n_to_load -1
        yield data, background, rot_inds, fname, i_shot, n_to_load



def setup_profile_log_files(logdir, name='simemc.profile', overwrite=True):
    """params: PHIL params, see simtbx.diffBragg.hopper phil string"""
    make_dir(logdir)

    mpi_logger._make_logger(name,
                 os.path.join(logdir, mpi_logger.HOST+"-profile"),
                 level=logging.INFO,
                 overwrite=overwrite,
                 formatter=logging.Formatter(mpi_logger.SIMPLE_FORMAT))



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


class EMC:

    def __init__(self, L, shots, prob_rots, min_p=0, outdir=None, beta=1,
                 symmetrize=True, whole_punch=True, img_sh=None, shot_scales=None,
                 refine_scale_factors=False, ave_signal_level=1):
        """
        run emc, the resulting density is stored in the L (emc.lerpy) object
        see call to this method in ests/test_emc_iteration
        :param L: instance of emc.lerpy in insert mode
        :param shots: array of images (Nshot, Npanel , slowDim , fastDim) or (Nshot, Npanel x slowDim x fastDim)
        :param prob_rots: list of array containing indices of probable orientations
        :param beta: float , controls the probability distribution for orientations. lower beta makes more orientations probable
        :param symmetrize: bool, update W after each iteration according to the space group symmetry operators
        :param whole_punch: bool, if True, values in W far away from Bragg peaks are set to 0 after each iterations
        :param shot_scales: list of floats, one per shot, same length as shots and prob_rots. Scale factor applied to
            tomograms for each shot
        :param refine_scale_factors: bool, update scale factors with 3d density
        :return: optimized density. Also L instance should be modified in-place
        """
        self.ave_signal_level=ave_signal_level
        self.L = L
        self.shots = shots
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
        self.Wprev = L.densities()
        self.whole_punch = whole_punch
        self.symmetrize = symmetrize
        self.shot_P_dr = []
        self.outdir = outdir
        self.beta = beta
        self.all_Wresid = []
        self.nshot_tot = COMM.bcast(COMM.reduce(self.nshots))
        self.success_corr = 0.2
        self.i_emc = 0
        self.save_model_freq = 10
        self.qs_inbounds = utils.qs_inbounds(self.L.qvecs.reshape((-1,3)), const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
        self.apply_density_rules()
        self.ave_time_per_iter = 0
        self.refine_scale_factors = refine_scale_factors
        self.update_scale_factors = False  # if the specific iteration is updating the per-shot scale factors
        self.update_density = True  # if the specific iteration is updating the density
        for i,s in enumerate(self.shots):
            self.shot_sums[i] = s[self.qs_inbounds].sum()

        if outdir is not None:
            make_dir(outdir)
            if COMM.rank==0:
                plt.figure(1)
                self.ax = plt.gca()
                init_density_file = os.path.join(self.outdir, "Winit.h5")
                self.save_h5(init_density_file)
            #self.plot_models(init=True)

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
            P_dr_sum = 0
            for i_shot, _, P_dr in self.finite_rot_inds.iter_record(rot_ind):
                W_rt += self.shots[i_shot]*P_dr
                P_dr_sum += P_dr*self.shot_scales[i_shot]
            W_rt = utils.errdiv(W_rt, P_dr_sum)
            self.L.trilinear_insertion(rot_ind, W_rt)

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
                P_dr_sum = 0
                for source, P_dr, tag in recv_info:
                    assert source != COMM.rank, "though supported we dont permit self sending"
                    self.print("get from rank %d, tag=%d" % (source, tag))
                    shot_data = COMM.recv(source=source, tag=tag)
                    shot_img, shot_scale = shot_data
                    self.print("GOT from rank %d, tag=%d" % (source, tag))
                    W_rt += shot_img.ravel()*P_dr
                    P_dr_sum += P_dr*shot_scale

                self.print("Done recv %d /%d" %(i_recv+1, len(rot_inds_to_recv)))
                assert rot_ind in self.finite_rot_inds # this is True by definition if you read the RotInds class method above
                for i_data, rank, P_dr in self.finite_rot_inds.iter_record(rot_ind):
                    W_rt += self.shots[i_data] * P_dr
                    P_dr_sum += P_dr*self.shot_scales[i_data]
                W_rt = utils.errdiv(W_rt, P_dr_sum)
                self.L.trilinear_insertion(rot_ind, W_rt)
        for req in send_req:
            req.wait()
        self.print("Done with tomogram send/recv")

    def reduce_density(self):
        self.print("Reducing density")
        den = self.L.densities()
        wts = self.L.wts()
        den = COMM.bcast(COMM.reduce(den))
        wts = COMM.bcast(COMM.reduce(wts))
        den = utils.errdiv(den, wts)
        self.print("MEAN DENSITY:", den.mean())
        self.L.update_density(den)

        self.apply_density_rules()

    def apply_density_rules(self):
        den = self.L.densities()
        if self.symmetrize:
            if COMM.rank==0:
                den = utils.symmetrize(den).ravel()
            den = COMM.bcast(den)

        if self.whole_punch:
            den = utils.whole_punch_W(den)
        self.L.update_density(den)

    def prep_for_insertion(self):
        self.L.toggle_insert()
        self.Wprev = self.L.densities()

    def do_emc(self, num_iter):
        self.i_emc = 0
        iter_times = []
        while self.i_emc < num_iter:
            t = time.time()

            self.update_scale_factors = (self.i_emc % 2 == 1) and self.refine_scale_factors
            self.update_density = not self.update_scale_factors

            self.finite_rot_inds = utils.RotInds()  # empty dictionary container for storing finite rot in info
            self.shot_P_dr = []
            log_R_per_shot = self.log_R_dr()
            for self.i_shot in range(self.nshots):
                log_R_dr = log_R_per_shot[self.i_shot]
                self.P_dr = self.get_P_dr(log_R_dr)
                self.shot_P_dr.append(self.P_dr)
                #assert np.any(self.P_dr > self.min_p)
                self.add_records()

            if self.update_scale_factors:
                updater = ScaleUpdater(self)
                maxiters = self.nshots*COMM.size*50
                #updater.update(bfgs=True, maxiter=maxiters, reparam=True, max_scale=1e6)
                updater.update(analytical=True)
                self.shot_scales *= self.ave_signal_level / np.mean(self.shot_scales)
                self.i_emc += 1
                t = time.time()-t
                iter_times.append(t)
                self.ave_time_per_iter = np.mean(iter_times)
                continue

            self.distribute_rot_ind_records()
            self.L.toggle_insert()
            self.insert_one_rankers()
            self.insert_multi_rankers()
            self.reduce_density()
            #self.plot_models(init=False)

            self.write_to_outdir()

            self.i_emc += 1
            t = time.time()-t
            iter_times.append(t)
            self.ave_time_per_iter = np.mean(iter_times)
            #if self.i_emc % self.save_model_freq==0 or self.i_emc==num_iter:
            #   self.plot_models()

    def log_R_dr(self, deriv=False):
        shot_log_R_dr = []
        shot_deriv_logR = []
        for i_shot, (img, rot_inds, scale_factor) in enumerate(zip(self.shots, self.prob_rots, self.shot_scales)):
            self.print("Getting Rdr %d / %d (scale=%f)" % ( i_shot+1, self.nshots, scale_factor))
            self.L.copy_image_data(img.ravel())
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
        kwargs["flush"] = True
        update_s = "fixed W" if self.update_scale_factors else "fixed PHI"
        print0("[Rnk%d; EmcStp %d; sec/it=%f; %s]" % (COMM.rank, self.i_emc, self.ave_time_per_iter, update_s), *args, **kwargs)

    def write_to_outdir(self):
        self.print("Write to outputdir")
        if self.outdir is None:
            return
        make_dir(self.outdir)
        if COMM.rank==0:
            density_file = os.path.join(self.outdir, "Witer%d.h5" % (self.i_emc+1))
            self.save_h5(density_file)

            #plt.figure(1)
            #self.ax.clear()
            #self.ax.plot(np.arange(len(self.all_Wresid)), self.all_Wresid , marker='x')
            #self.ax.set_xlabel("iteration", fontsize=11)
            #self.ax.set_ylabel("$|W'-W|$",fontsize=11)
            #self.ax.set_yscale("log")
            #self.ax.grid(which='both', axis='y', ls='--')
            #figname = os.path.join(self.outdir, "w_convergence%d.png" % (self.i_emc+1))
            #plt.savefig(figname)
            ##plt.draw()
            ##plt.pause(0.1)

    def save_h5(self, density_file):
        den = self.L.densities()
        with h5py.File(density_file, "w") as out_h5:
            out_h5.create_dataset("Wprime",data=den.reshape((const.NBINS, const.NBINS, const.NBINS)), compression="lzf")

    def success_rate(self, init=False, return_models=False):
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
        self.all_Wresid.append(wresid)
        nsuccess = COMM.reduce(nsuccess)
        if COMM.rank==0:
            success_rate = float(nsuccess) / self.nshot_tot * 100.
            print0("[ EmcStep%d ] SUCCESS RATE: %d/%d (%.2f %%); Wresid=%f" % (self.i_emc, nsuccess , self.nshot_tot, success_rate, wresid))
        if return_models:
            return models

    def plot_models(self, init=False, pid=0):
        # TODO: generalize to Npanel ?
        self.print("Plotting models")
        models = self.success_rate(init=init, return_models=True)
        if self.outdir is None:
            return
        fig, axs = plt.subplots(nrows=2, ncols=4)
        fig.set_size_inches((9.81,4.8))
        N = min(4, self.nshots)
        for i in range(N):
            mod_img = models[i].reshape(self.img_sh)[pid]
            dat_img = self.shots[i].reshape(self.img_sh)[pid]
            imgs = mod_img, dat_img

            img_ax = axs[0,i], axs[1,i]
            for ax,img in zip(img_ax, imgs):
                m = img.mean()
                s = img.std()
                ax.imshow(img, vmin=m-s, vmax=m+s)

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(length=0)
                ax.grid(ls='--', lw=0.75)
                ax.set_aspect('auto')
            axs[0,0].set_ylabel("models")
            axs[1,0].set_ylabel("data")

        if init:
            figname = os.path.join(self.outdir, "model_agreement_init_rank%d.png" % COMM.rank )
            plt.suptitle("model versus data (init)")
        else:
            figname = os.path.join(self.outdir, "model_agreement_iter%d_rank%d.png" % (self.i_emc+1, COMM.rank) )
            plt.suptitle("model versus data (iter=%d)" % (self.i_emc+1))
        plt.savefig(figname)


class ScaleUpdater:

    def __init__(self, emc):
        self.emc = emc
        self.params = None
        self.reparam = True
        self.shot_names = ["rank%d-shot%d" % (COMM.rank, i_shot) for i_shot in range(self.emc.nshots)]
        all_shot_names = COMM.reduce(self.shot_names)
        shot_name_xpos = None
        if COMM.rank==0:
            shot_name_xpos = {name: i for i, name in enumerate(all_shot_names)}
        self.shot_name_xpos = COMM.bcast(shot_name_xpos)

    def update(self, bfgs=True, maxiter=None, reparam=True, max_scale=1e6, analytical=False):
        if analytical:
            for i_shot, shot_sum in enumerate(self.emc.shot_sums):
                P_dr = self.emc.shot_P_dr[i_shot]
                new_scale = np.sum(P_dr) * shot_sum
                new_scale_norm = 0
                for i_rot, rot_ind in enumerate(self.emc.prob_rots[i_shot]):
                    W_ir = self.emc.L.trilinear_interpolation(rot_ind)
                    Wsum = W_ir[self.emc.qs_inbounds].sum()
                    new_scale_norm += Wsum*P_dr[i_rot]
                if new_scale_norm > 0:
                    new_scale /= new_scale_norm
                    self.emc.shot_scales[i_shot] = new_scale
                    self.emc.print("New scale (shot %d) =%f" % ( i_shot, new_scale))
                else:
                    self.emc.print("New scale (shot %d) had norm=0 so not updating" % (i_shot))
            return

        init_shot_scales = np.zeros(len(self.shot_name_xpos))
        self.params = Parameters()
        self.reparam = reparam
        for scale, name in zip(self.emc.shot_scales, self.shot_names):
            p = RangedParameter(init=scale, minval=1e-7, maxval=max_scale, name=name)
            self.params.add(p)
            xpos = self.shot_name_xpos[name]
            init_shot_scales[xpos] = scale

        x0 = np.ones(len(self.shot_name_xpos))
        bounds = None
        if not reparam:
            x0 = init_shot_scales
            bounds = [(1e-7,max_scale)] * len(init_shot_scales)
        if bfgs:
            out = fmin_l_bfgs_b(self, x0=x0, fprime=self.jac, bounds=bounds)
            xopt = out[0]
        else:
            out = minimize(self, x0=x0, method="Nelder-Mead", options={'maxiter': maxiter})
            xopt = out.x

        new_scales = []
        for i, name in enumerate(self.shot_names):
            p = self.params[name]
            xpos = self.shot_name_xpos[name]
            xval = xopt[xpos]
            if reparam:
                new_scale = p.get_val(xval)
            else:
                new_scale = xval
            self.emc.shot_scales[i] = new_scale
            new_scales.append(new_scale)

    def jac(self, x, *args):
        return self.gra

    def __call__(self, x, *args):
        f, self.gra = self.target(x)
        return f

    def target(self, x):
        """
        returns the functional and its gradient , w.r.t. x
        :param x: refiner parameters (all shot scale factors across all MPI ranks)
        :return: 2-tuple of (float, np.ndarray) (same length as x )
        """
        emc = self.emc
        functional = 0
        grad = np.zeros(len(x))

        scale_on_rank = []
        for name in self.shot_names:
            p = self.params[name]
            xpos = self.shot_name_xpos[name]
            xval = x[xpos]
            if self.reparam:
                scale = p.get_val(xval)
            else:
                scale = xval
            scale_on_rank.append(scale)
        scale_on_rank = np.array(scale_on_rank)
        Q_per_shot, dQ_per_shot = utils.compute_log_R_dr(
            emc.L, emc.shots, emc.prob_rots, scale_on_rank, deriv=True, verbose=COMM.rank == 0)

        for i_shot, (Q, dQ) in enumerate(zip(Q_per_shot, dQ_per_shot)):

            Q = np.array(Q)
            dQ = np.array(dQ)

            P_dr = emc.shot_P_dr[i_shot]
            grad_term = P_dr*dQ

            functional += (P_dr*Q).sum()
            gsum = grad_term.sum()
            name = self.shot_names[i_shot]
            p = self.params[name]
            xpos = self.shot_name_xpos[name]
            if self.reparam:
                g = p.get_deriv(x[xpos], gsum)
            else:
                g = gsum
            grad[xpos] = g

        grad = COMM.bcast(COMM.reduce(grad))
        functional = COMM.bcast(COMM.reduce(functional))
        # running a minimizer so return the negative loglike and its gradient
        return -functional, -grad
