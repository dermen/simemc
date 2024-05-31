from mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
from scipy.optimize import minimize, line_search
from simtbx.diffBragg.refiners.parameters import Parameters, RangedParameter
from simemc import utils, mpi_utils
import logging


class Updater:

    def __init__(self, emc):
        self.LOGGER = logging.getLogger(utils.LOGNAME)
        self.LOGGER.debug("Instantiating updater base class")
        self.emc = emc
        self.f = None
        self.g = None  # gradient of refinement parameters
        self.iter_num = 0
        self.xprev = None
        self.params = None
        self.reparam = True
        self.shot_names = ["rank%d-shot%d" % (COMM.rank, i_shot) for i_shot in range(self.emc.nshots)]
        all_shot_names = COMM.reduce(self.shot_names)
        shot_name_xpos = None
        if COMM.rank == 0:
            shot_name_xpos = {name: i for i, name in enumerate(all_shot_names)}
        self.shot_name_xpos = COMM.bcast(shot_name_xpos)

    def __call__(self, x, *args):
        self.f, self.g = self.target(x)
        return self.f

    def jac(self, x, *args):
        assert self.g is not None
        return self.g

    def target(self, x):
        return None, None

    def check_convergence(self, x):
        pass


class DensityUpdater(Updater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: assert emc.L is in sparse mode
        assert self.emc.L.is_in_sparse_mode
        #mpi_utils.print_gpu_usage_across_ranks(self.emc.L)
        COMM.barrier()
        self.min_prob = 1e-5
        self.prev_iter_F = np.inf

    def update(self, how="line_search", lbfgs_maxiter=60):
        """

        :return: optimized density, returns None for rank >0
        """
        xstart = None
        if COMM.rank==0:
            dens_start = self.emc.L.densities()

            self.LOGGER.debug("density start assert")
            assert np.all(dens_start >= 0)

            is_zero = dens_start == 0
            if np.any(is_zero):
                self.LOGGER.debug("WARNING!!!!!!! Density is 0 in some places")
                min_pos_val = min(1e-7, dens_start[~is_zero].min())
                dens_start[is_zero] = min_pos_val

            self.LOGGER.debug("Reparameterizing density")
            xstart = np.sqrt((dens_start + 1)**2 -1)

        xstart = mpi_utils.bcast_large(xstart, verbose=True, comm=COMM)

        if how=="line_search":
            f, g = self.target(xstart)
            pk = -g
            out = line_search(self, myfprime=self.jac, xk=xstart, pk=pk, gfk=g, maxiter=50)
            self.emc.print("")
            self.emc.print("Line search finished", out)
            alpha = out[0]
            self.emc.L.update_density(xstart + alpha*pk)

        elif how == "lbfgs":
            self.prev_iter_F = np.inf  # for print purposes only, we keep track of the target functional
            self.LOGGER.debug("Beginning density optimization process")
            out = minimize(self, xstart, method="L-BFGS-B", jac=self.jac, callback=None,
                           options={"maxiter": lbfgs_maxiter})
            self.LOGGER.debug("Minimization has terminated in %d iterations (final value=%10.7g, msg=%s, success=%s)" % (out.nit,out.fun, out.message, out.success))
            self.emc.print("")
            self.emc.L.update_reparameterized_density(out.x)
        else:
            raise NotImplementedError("method %s not supported" % how)

    def target(self, x):
        emc = self.emc
        # x is reparameterized so we need to operate on it
        # according to
        #      x -> np.sqrt(x**2+ 1) -1
        # this operation is done in-place on the GPU

        emc.L.update_reparameterized_density(x)

        emc.L.reset_density_derivs()

        functional = 0

        #mpi_utils.print_gpu_usage_across_ranks(self.emc.L, logger=self.LOGGER)
        #mpi_utils.print_cpu_mem_usage(logger=self.LOGGER)

        self.LOGGER.debug("Compute func/grads for %d shots" % emc.nshots)
        for i_shot in range(emc.nshots):
            print_s = "Maximization iter %d ( %d/ %d) (last F=%10.7g)" % (self.iter_num+1, i_shot+1, emc.nshots, self.prev_iter_F)
            self.LOGGER.debug(print_s)
            self.emc.print(print_s, end="\r", flush=True)
            P_dr = emc.shot_P_dr[i_shot]
            is_finite_prob = np.array(P_dr) >= self.min_prob

            self.LOGGER.debug("Copying image data  iter%d (%d / %d)" % (self.iter_num+1, i_shot+1, emc.nshots))
            emc.L.copy_image_data(emc.shots[i_shot], emc.shot_mask, emc.shot_background[i_shot])
            self.LOGGER.debug("Done Copying image data shot  iter%d (%d / %d)" % (self.iter_num+1, i_shot+1, emc.nshots))

            finite_rot_inds = emc.prob_rots[i_shot][is_finite_prob]  # TODO : verify type is np.ndarray and avoid the extra call to np.array
            finite_P_dr = P_dr[is_finite_prob]
            self.LOGGER.debug("Dens deriv iter%d (%d / %d)" % (self.iter_num+1, i_shot+1, emc.nshots))
            emc.L.dens_deriv(finite_rot_inds, finite_P_dr, verbose=False, shot_scale_factor=emc.shot_scales[i_shot],
                             reset_derivs=False, return_grad=False)
            self.LOGGER.debug("Done Dens deriv iter%d (%d / %d)" % (self.iter_num+1, i_shot+1, emc.nshots))

            log_Rdr = np.array(emc.L.get_out())
            self.LOGGER.debug("Done get log Rdr iter%d (%d / %d)" % (self.iter_num+1, i_shot+1, emc.nshots))

            functional += (finite_P_dr*log_Rdr).sum()

        self.LOGGER.debug("Done with rank grad/functional (iter %d)" % (self.iter_num+1))
        COMM.barrier()
        self.LOGGER.debug("Reducing grad")
        emc.L.allreduce_density_derivs(COMM)
        COMM.barrier()
        self.LOGGER.debug("functional")
        functional = COMM.reduce(functional)
        functional = COMM.bcast(functional)

        # Because we reparameterized, we must scale the grad -> dW/dx *grad
        self.LOGGER.debug("Scaling the gradient (iter %d)" % (self.iter_num+1))
        minus_grad = emc.L.reparameterized_densities_gradient(x)
        self.LOGGER.debug("Bcasting the Scaled gradient (iter %d)" % (self.iter_num+1))
        emc_s = "Done with emc iter num: %d (F=%f)" \
                % (self.iter_num+1, functional)
        self.LOGGER.debug(emc_s)

        # running a minimizer, so return the negative loglike and its gradient
        self.iter_num += 1
        self.prev_iter_F = -functional
        return -functional, minus_grad


class ScaleUpdater(Updater):
    """
    used to update per-shot scale factors during EMC
    """
    def __init__(self, *args, **kwargs):
        """

        :param emc: instance of EMC class
        """
        super().__init__(*args, **kwargs)
        self.min_prob = 1e-5


    def update(self, bfgs=True, maxiter=None, reparam=True, max_scale=1e6, analytical=False):
        """

        :param bfgs: use L-BFGS
        :param maxiter: max number iterations, (only applies to Nelder-mead, if bfgs=False, and analytical=False
        :param reparam: apply reparmeterization restraints (only applies if analytical=False)
        :param max_scale: maximum allowed scale factor (only applies if analytical=False)
        :param analytical: use the analytical update rule (all other parameters are ignore if this one is True)
        :return: it analytical is True, then return boolean flags specifying whether a scale factor was updated, list, same length as emc.shot_scales
        """
        if analytical:
            rank_with_most_inserts, total_inserts = mpi_utils.determine_rank_with_most_inserts(self.emc)
            i_insert = 0
            new_scales = []
            mean_scale = 0
            median_scale = 0
            stdev_scale = 0
            mask_in_bounds = self.emc.shot_mask[self.emc.qs_inbounds]
            scale_changed = []
            for i_shot, shot_sum in enumerate(self.emc.shot_sums):
                P_dr = self.emc.shot_P_dr[i_shot]
                new_scale = np.sum(P_dr) * shot_sum
                new_scale_norm = 0
                for i_rot, rot_ind in enumerate(self.emc.prob_rots[i_shot]):
                    #TODO skip over  rots with very small probs
                    if COMM.rank==rank_with_most_inserts:
                        perc = i_insert / total_inserts * 100.
                        print("Updating scale factors %1.2f %% -- New scales mean=%f, median=%f, stdev=%f" % (perc, mean_scale, median_scale, stdev_scale), end="\r", flush=True)
                    W_ir = self.emc.L.trilinear_interpolation(rot_ind)
                    W_ir_inbounds = W_ir[self.emc.qs_inbounds]
                    Wsum = W_ir_inbounds[mask_in_bounds].sum()
                    new_scale_norm += Wsum*P_dr[i_rot]
                    i_insert += 1
                if new_scale_norm > 0:
                    new_scale /= new_scale_norm
                    self.emc.shot_scales[i_shot] = new_scale
                    new_scales.append(new_scale)
                    scale_changed.append(True)
                else:
                    new_scales.append(self.emc.shot_scales[i_shot])
                    scale_changed.append(False)
                    #print("WARNING: rank=%d, New scale (shot %d) had norm=0 so not updating" % (COMM.rank, i_shot))
                mean_scale = np.mean(new_scales)
                median_scale = np.median(new_scales)
                stdev_scale = np.std(new_scales)
            if COMM.rank == rank_with_most_inserts:
                print("\n")
            return scale_changed

        # else if not analytical, use L-BFGS:
        init_shot_scales = np.zeros(len(self.shot_name_xpos))
        self.params = Parameters()
        self.reparam = reparam
        for scale, name in zip(self.emc.shot_scales, self.shot_names):
            p = RangedParameter(init=scale, minval=0, maxval=max_scale, name=name)
            self.params.add(p)
            xpos = self.shot_name_xpos[name]
            init_shot_scales[xpos] = scale

        x0 = np.ones(len(self.shot_name_xpos))
        bounds = None
        if not self.reparam:
            x0 = init_shot_scales
            bounds = [(1e-7, max_scale)] * len(init_shot_scales)
        if bfgs:
            try:
                out = minimize(self, x0=x0, jac=self.jac, method="L-BFGS-B", bounds=bounds,
                               callback=None, options={'maxiter': 100})
                xopt = out.x
            except StopIteration:
                xopt = self.xprev
        else:
            out = minimize(self, x0=x0, method="Nelder-Mead", options={'maxiter': maxiter}, callback=self.check_convergence)
            xopt = out.x

        new_scales = []
        for i, name in enumerate(self.shot_names):
            p = self.params[name]
            xpos = self.shot_name_xpos[name]
            xval = xopt[xpos]
            if self.reparam:
                new_scale = p.get_val(xval)
            else:
                new_scale = xval
            self.emc.shot_scales[i] = new_scale
            new_scales.append(new_scale)

    def get_new_scales(self, x, reparam=True):
        new_scales = []
        for i, name in enumerate(self.shot_names):
            p = self.params[name]
            xpos = self.shot_name_xpos[name]
            xval = x[xpos]
            if reparam:
                new_scale = p.get_val(xval)
            else:
                new_scale = xval
            new_scales.append(new_scale)
        return np.array(new_scales)

    def check_convergence(self, x):
        if self.iter_num == 0:
            self.xprev = x
            self.iter_num += 1
            return False

        current_scales = self.get_new_scales(x, self.reparam)
        prev_scales = self.get_new_scales(self.xprev, self.reparam)
        perc_diff = np.abs(current_scales - prev_scales ) / prev_scales
        all_perc_diff = COMM.gather(perc_diff)
        is_converged = None
        if COMM.rank==0:
            all_perc_diff = np.hstack(all_perc_diff)
            ave_perc_diff = np.mean(all_perc_diff) * 100
            n_above_1perc = sum(all_perc_diff > 0.01)
            is_converged = n_above_1perc == 0

            self.emc.print("it=%d,Ave%%-diff=%1.2f%%. Num.shots with %%-diff>1%% = %d/%d. Converged=%s"
                           % (self.iter_num+1, ave_perc_diff, n_above_1perc, self.emc.nshot_tot, is_converged))
        is_converged = COMM.bcast(is_converged)

        self.xprev = x
        self.iter_num += 1
        if is_converged:
            raise StopIteration()

    def target(self, x):
        """
        returns the functional and its gradient , w.r.t. x
        :param x: refiner parameters (all shot scale factors across all MPI ranks)
        :return: 2-tuple of (float, np.ndarray same length as x )
        """
        x = COMM.bcast(x)  # should always be the same here, but noticed drift in density updater...
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
        #Q_per_shot, dQ_per_shot = utils.compute_log_R_dr(
        #    emc.L, emc.shots, emc.prob_rots, scale_on_rank, emc.shot_mask, bg=emc.shot_background, deriv=1)

        for i_shot in range(emc.nshots):  # , (Q, dQ) in enumerate(zip(Q_per_shot, dQ_per_shot)):
            P_dr = emc.shot_P_dr[i_shot]
            is_finite_prob = np.array(P_dr) >= self.min_prob
            emc.L.copy_image_data(emc.shots[i_shot], emc.shot_mask, emc.shot_background[i_shot])

            finite_rot_inds = emc.prob_rots[i_shot][is_finite_prob]  # TODO : verify type is np.ndarray and avoid the extra call to np.array
            finite_P_dr = P_dr[is_finite_prob]

            scale_factor = scale_on_rank[i_shot]
            emc.L.equation_two(finite_rot_inds, False, scale_factor)
            Q = np.array(emc.L.get_out())

            emc.L.equation_two(finite_rot_inds, False, scale_factor, deriv=1)
            dQ = np.array(emc.L.get_out())

            grad_term = finite_P_dr*dQ

            functional += (finite_P_dr*Q).sum()
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
