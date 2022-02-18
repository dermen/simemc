
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os
import h5py
import time
import numpy as np
from collections import Counter
from simemc import loading_dragonfly_data

#from reborn.misc.interpolate import trilinear_insertion
from simemc import utils
from simemc import mpi_utils
print0 = mpi_utils.print0  # rank0 print
print0f = mpi_utils.print0f  # rank0 print with stdout flush
printR = mpi_utils.printR  # all rank print
printRf = mpi_utils.printRf  # all rank print with stdout flush
from simemc.emc import lerpy
from simemc import const
import pylab as plt

def dragon_loader(loader, dt, quat_f):
    n = loader.N_frames_tot
    inds_each_rank = np.array_split(np.arange(n), COMM.size)
    inds_to_load = inds_each_rank[COMM.rank]
    #inds_to_load = np.arange(COMM.rank*10, COMM.rank*10+10)
    frames = loader.load_frames(inds_to_load[0], inds_to_load[-1]+1, check_mem=False)
    data = frames['pad_data']
    #r_e2 = 7.940787682024162e-30
    #SA = frames['solid_angles'] * r_e2 * frames["J0"]
    SA = 1
    n_to_load = data.shape[0]

    rotMats, _ = utils.load_quat_file(quat_f)
    rot_inds = np.arange(len(rotMats)).astype(np.int32)
    printRf("Will load %d shots (%d-%d)" % ( data.shape[0], inds_to_load[0], inds_to_load[-1]) )

    for i in range(data.shape[0]):
        n_to_load = n_to_load - 1
        d = data[i] / SA
        if d.dtype != dt:
            d = d.astype(dt)
        yield d, [0], rot_inds, None, None, n_to_load


def main():
    max_iter = 100
    output_dir = "dragon_out5"
    loader = loading_dragonfly_data.Loader('sim.ini')

    mpi_utils.make_dir(output_dir)  # only use rank0 to makedir, if it doesnt already exist
    quat_file = "quatgrid/c-quaternion9.bin"
    rotMats, rotMatWeights = utils.load_quat_file(quat_file)
    Rdr_getter = lerpy()

    maxRotInds = len(rotMats)+1
    img_sh = 1,150,150
    maxNumQ = int(np.product(img_sh)) # boost doesnt like np.int64, consider putting a converter in trilerp_ext
    num_data_pix = maxNumQ
    nbin=const.NBINS
    dens_sh = (nbin,nbin,nbin)
    corners, deltas = utils.corners_and_deltas(dens_sh, const.X_MIN_DRAGON, const.X_MAX_DRAGON)

    Wcurr = np.random.random(dens_sh)*100
    all_Wresid = []

    num_dev = 8
    beta=1 #1e-3
    dev_id = COMM.rank % num_dev
    # TODO warn if more than X ranks are using the same device

    qx,qy,qz = utils.load_qmap("/global/cscratch1/sd/dermen/lyso/simemc/sim_geo_qmap.npy")
    qcoords = np.vstack((qx, qy, qz)).T

    print0("Allocating for %d rotmats and %d pixels" % (rotMats.shape[0], num_data_pix))
    Rdr_getter.allocate_lerpy(
        dev_id, rotMats, Wcurr, int(maxNumQ),
        tuple(corners), tuple(deltas), qcoords,
        maxRotInds, int(num_data_pix))

    dt = np.float32 if Rdr_getter.size_of_cudareal==4 else np.float64
    data_loader = dragon_loader(loader, dt, quat_file)
    verbose = False
    rank_data = {}
    rank_bg = {}
    rank_rot_inds = {}
    actual_max_rot_inds =0
    sum_data = np.zeros(num_data_pix, dt)
    num_shots = 0
    for i, (d, bg, rot_inds, h5name, dset_idx,num_remain) in enumerate(data_loader):
        print0f("data max=%.3f, bkgrnd max=%.3f, remaining shots to load: %d" % (np.max(d), np.max(bg), num_remain))
        rank_data[i] = d
        rank_rot_inds[i] = rot_inds
        actual_max_rot_inds = max(len(rot_inds), actual_max_rot_inds)
        rank_bg[i] = bg
        sum_data += d
        num_shots += 1

    del loader
    ##########################
    # get the intial model:
    ###########################
    print0f("Getting initial model")
    tWinit = time.time()
    tot_num_shots = COMM.reduce(num_shots)
    all_sum_data = None
    if COMM.rank==0:
        all_sum_data = np.zeros_like(sum_data)
    COMM.Reduce(sum_data, all_sum_data)
    if COMM.rank==0:
        all_sum_data = all_sum_data / tot_num_shots
    all_sum_data = COMM.bcast(all_sum_data)
    # insert at each orientation
    Rdr_getter.toggle_insert()
    for i_rot in range(rotMats.shape[0]):
        if i_rot % COMM.size != COMM.rank:
            continue
        perc = float(i_rot) / rotMats.shape[0] * 100.
        print0f("[Getting initial W (%1.2f %%)]" % perc)
        Rdr_getter.trilinear_insertion(i_rot, all_sum_data)
    dens = Rdr_getter.densities()
    wts = Rdr_getter.wts()
    all_dens = all_wts = None
    if COMM.rank==0:
        all_dens = np.zeros_like(dens)
        all_wts = np.zeros_like(wts)
    COMM.Reduce(dens, all_dens)
    COMM.Reduce(wts, all_wts)
    if COMM.rank==0:
        with np.errstate(divide='ignore', invalid='ignore'):
            Wcurr = np.nan_to_num(all_dens / all_wts)
    Wcurr = COMM.bcast(Wcurr)
    Rdr_getter.update_density(Wcurr)
    tWinit = time.time()-tWinit
    print0f("Took %.4f sec to get initial W" % tWinit)
    #############################
    # Done getting initial model
    #############################

    print0f("computing actual max rot inds")
    actual_max_rot_inds = COMM.gather(actual_max_rot_inds)
    if COMM.rank==0:
        actual_max_rot_inds = max(actual_max_rot_inds)
        print0f("Maximum number of rot inds=%d" % actual_max_rot_inds)
    actual_max_rot_inds = COMM.bcast(actual_max_rot_inds)
    assert actual_max_rot_inds <= maxRotInds, ("increase maxRotInds to be %d" % actual_max_rot_inds)
    COMM.barrier()

    num_iter = 0
    num_data = len(rank_data)
    while num_iter < max_iter:
        print0f("Beginning iteration %f" % time.time())
        finite_rot_inds = utils.RotInds()
        titer = time.time()
        for i_data in rank_data:
            t = time.time()
            # get R_dr for each of the shots probable rotation indices
            d = rank_data[i_data]
            rot_inds = rank_rot_inds[i_data]
            Rdr_getter.copy_image_data(d)
            Rdr_getter.equation_two(rot_inds, verbose=verbose)
            log_R_dr = np.array(Rdr_getter.get_out())
            R_dr = np.exp(log_R_dr)

            # compute the normalized probs (P_dr)
            wts_dr = rotMatWeights[rot_inds]  # TODO cache these wts per shot
            wR = wts_dr*R_dr**beta
            P_dr = utils.errdiv(wR , np.sum(wR))
            #P_dr[P_dr < prob_thresh] = 0
            #Psum = P_dr.sum()
            #if Psum > 0:
            #    P_dr /= Psum
            t = time.time()-t

            perc = (i_data+1) / float(num_data) *100.
            print0f("[ITER %d, %.4fs (%2.1f%%)]" % (num_iter+1, t,perc), "P_dr range: %.4f-%.4f (num P_dr >0.05: %d, >0: %d) "
                    %( P_dr.mean(), P_dr.max(), np.sum(P_dr > 0.05), np.sum(P_dr > 0)))

            for rot_ind, prob in zip(rot_inds, P_dr):
                #if prob == 0:
                #    continue
                if rot_ind not in finite_rot_inds:
                    finite_rot_inds[rot_ind] = {}
                    finite_rot_inds[rot_ind]["i_data"] = []
                    finite_rot_inds[rot_ind]["P_dr"] = []
                    finite_rot_inds[rot_ind]["rank"] = []
                finite_rot_inds[rot_ind]["i_data"].append(i_data)
                finite_rot_inds[rot_ind]["P_dr"].append(prob)
                finite_rot_inds[rot_ind]["rank"].append(COMM.rank)

        print0f("computing unique rot inds")
        unique_rot_inds = list(finite_rot_inds.keys())
        unique_rot_inds_all_ranks = COMM.reduce(unique_rot_inds)
        rot_inds_on_multiple_ranks = None
        if COMM.rank==0:
            C = Counter(unique_rot_inds_all_ranks)
            rot_inds_on_multiple_ranks = set([rot_ind for rot_ind,count in C.items() if count > 1])
            print0f(len(rot_inds_on_multiple_ranks), maxRotInds)
            time.sleep(2)
        rot_inds_on_multiple_ranks = COMM.bcast(rot_inds_on_multiple_ranks)
        print0f("Computed unique rot inds")

        Rdr_getter.toggle_insert()

        ###########################################################################
        # First, contributions from rot indices that exist on one rank only
        # For these, no MPI communication is necessary to get the tomogram updates
        ###########################################################################
        t = time.time()
        #rots_on_one_rank = rot_inds_global.on_one_rank
        for i_rot_ind, rot_ind in enumerate(finite_rot_inds):
            if rot_ind in rot_inds_on_multiple_ranks:
                continue
            print0f("Finite rot ind %d / %d" %(i_rot_ind, len(finite_rot_inds)))
            W_rt = np.zeros(num_data_pix, dtype=dt)
            P_dr_sum = 0
            for i_data, P_dr in zip(finite_rot_inds[rot_ind]['i_data'], finite_rot_inds[rot_ind]['P_dr']):
                W_rt += rank_data[i_data]*P_dr
                P_dr_sum += P_dr
            W_rt = utils.errdiv(W_rt , P_dr_sum)

            # insert
            Rdr_getter.trilinear_insertion(rot_ind, W_rt)

        tfinite_rot = time.time()-t
        COMM.barrier()
        print0f("done with Finite rot inds (%.4f sec)" % tfinite_rot)
        
        # sum up (reduce) all insertions thus far
        rank_Wprime = Rdr_getter.densities()
        rank_wts = Rdr_getter.wts()

        Wprime = wts = None
        if COMM.rank==0:
            Wprime = np.zeros_like(rank_Wprime)
            wts = np.zeros_like(rank_wts)

        COMM.Reduce(rank_Wprime, Wprime)
        COMM.Reduce(rank_wts, wts)

        ###############################################################
        # Second, if a rot index was on multiple ranks,
        # we mpi reduce it across ranks and add its contribution to density on rank0
        ###############################################################
        t2 = time.time()

        Rdr_getter.toggle_insert()
        for i_rot_ind, rot_ind in enumerate(rot_inds_on_multiple_ranks):
            perc = float(i_rot_ind)  / len(rot_inds_on_multiple_ranks)  * 100
            print0f("[ITER %d: Reducing W_rt across ranks (%1.2f %%)]"
                    % (num_iter+1, perc))
            if rot_ind in finite_rot_inds:
                rank_W_rt = np.zeros(num_data_pix, dtype=dt)
                rank_P_dr_sum = 0
                for i_data, P_dr in zip(finite_rot_inds[rot_ind]['i_data'], finite_rot_inds[rot_ind]['P_dr']):
                    rank_W_rt += rank_data[i_data]*P_dr
                    rank_P_dr_sum += P_dr
            else:
                #TODO ensure rank_data[0] exists, maybe safer to use tomogram shape here to init zeros
                rank_W_rt = np.zeros_like(rank_data[0])
                rank_P_dr_sum = 0

            W_rt = None
            if COMM.rank==0:
                W_rt = np.zeros_like(rank_W_rt)
            COMM.Reduce(rank_W_rt, W_rt)
            P_dr_sum = COMM.reduce(rank_P_dr_sum)

            if COMM.rank==0:
                W_rt  = utils.errdiv(W_rt, P_dr_sum)
                Rdr_getter.trilinear_insertion(rot_ind, W_rt)
        t2 = time.time()-t2
        print0f("Took %f sec to reduce tomograms across ranks" % t2)

        ##############################################
        # Finally, reduce /broadcast / normalize Wprime
        ##############################################
        if COMM.rank==0:
            Wprime += Rdr_getter.densities()
            wts += Rdr_getter.wts()
            Wprime = utils.errdiv(Wprime, wts)
        Wprime = COMM.bcast(Wprime)
        Rdr_getter.update_density(Wprime)

        t = time.time()
        Wresid = np.sum((Wcurr.ravel() - Wprime.ravel())**2)
        density_file = os.path.join(output_dir, "Witer%d.h5" % (num_iter+1))
        if COMM.rank==0:
            with h5py.File(density_file, "w") as out_h5:
                out_h5.create_dataset("Wprime",data=Wprime.reshape((const.NBINS, const.NBINS, const.NBINS)), compression="lzf")
        t = time.time()-t
        print0f("Took %.4f sec to write current density to file %s" %(t, density_file))

        Wcurr = Wprime.copy()
        num_iter += 1
        titer = time.time()-titer
        print0f("Iter %d: Wresid=%1.2e took %.4f sec" % (num_iter, Wresid, titer))
        if COMM.rank==0:
            all_Wresid.append(Wresid)
            plt.clf()
            plt.plot(np.arange(num_iter), all_Wresid , marker='x')
            plt.xlabel("iteration", fontsize=11)
            plt.ylabel("$|W'-W|$",fontsize=11)
            plt.gca().set_yscale("log")
            plt.gca().grid(which='both', axis='y', ls='--')
            figname = os.path.join(output_dir, "w_convergence%d.png" % num_iter)
            plt.savefig(figname)


if __name__=="__main__":
    main()
