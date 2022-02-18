
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os
import h5py
import time
import numpy as np
#from simemc import loading_dragonfly_data

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


def main():
    max_iter = 100
    input_dir = "../1600sim_noBG/emc_input"
    output_dir = "../1600sim_noBG/emc_output_sym"
    #input_dir = "dummie_input2"
    #output_dir = "dummie_output2"

    mpi_utils.make_dir(output_dir)  # only use rank0 to makedir, if it doesnt already exist
    quat_file = "quatgrid/c-quaternion70.bin"

    #loader = loading_dragonfly_data.Loader('sim.ini')
    #quat_file = loader.quats_file
    binary=True

    rotMats, rotMatWeights = utils.load_quat_file(quat_file, binary)
    Rdr_getter = lerpy()
    prob_thresh = 0 # any P_dr with value less than prob_thresh will be set to 0 ..

    maxRotInds = 10000 # TODO check max rotInds
    img_sh = 1, 2527, 2463
    maxNumQ = int(np.product(img_sh)) # boost doesnt like np.int64, consider putting a converter in trilerp_ext
    num_data_pix = maxNumQ
    nbin=const.NBINS
    corners, deltas = utils.corners_and_deltas((nbin,nbin,nbin), const.X_MIN, const.X_MAX)

    Wcurr = utils.get_W_init().ravel()
    Wcurr = np.random.random(Wcurr.shape)+1
    all_Wresid = []

    num_dev = 8
    beta=1 #1e-3
    dev_id = COMM.rank % num_dev
    # TODO warn if more than X ranks are using the same device

    qx, qy, qz = utils.load_qmap("../qmap.npy")
    qcoords = np.vstack((qx, qy, qz)).T

    print0("Allocating")
    Rdr_getter.allocate_lerpy(
        dev_id, rotMats, Wcurr, int(maxNumQ),
        tuple(corners), tuple(deltas), qcoords,
        maxRotInds, int(num_data_pix))

    dt = np.float32 if Rdr_getter.size_of_cudareal==4 else np.float64
    data_loader = mpi_utils.load_emc_input(input_dir, dt)
    verbose = False
    rank_data = {}
    rank_bg = {}
    rank_rot_inds = {}
    rank_h5name = {}
    rank_dset_idx = {}
    max_rot_inds =0
    for i, (d, bg, rot_inds, h5name, dset_idx,num_remain) in enumerate(data_loader):
        printRf("data max=%.3f, bkgrnd max=%.3f, remaining shots to load: %d" % (d.max(), bg.max(), num_remain))
        rank_data[i] = d
        rank_rot_inds[i] = rot_inds
        max_rot_inds = max(len(rot_inds), max_rot_inds)
        rank_bg[i] = bg
        rank_h5name[i] = h5name
        rank_dset_idx[i] = dset_idx

    max_rot_inds = COMM.gather(max_rot_inds)
    if COMM.rank==0:
        max_rot_inds = max(max_rot_inds)
        print0f("Maximum number of rot inds=%d" % max_rot_inds)
    max_rot_inds = COMM.bcast(max_rot_inds)
    assert max_rot_inds < maxRotInds, ("increase maxRotInds to be %d" % max_rot_inds)
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
            with np.errstate(divide='ignore', invalid='ignore'):
                wR = wts_dr*R_dr**beta
                P_dr = np.nan_to_num(wR / np.sum(wR))
                #P_dr[P_dr < prob_thresh] = 0
                #Psum = P_dr.sum()
                #if Psum > 0:
                #    P_dr /= Psum
            t = time.time()-t

            perc = (i_data+1) / float(num_data) *100.
            print0f("[ITER %d, %.4fs (%2.1f%%)]" % (num_iter+1, t,perc), "P_dr range: %.4f-%.4f (num P_dr >0.05: %d, >0: %d) "
                    %( P_dr.mean(), P_dr.max(), np.sum(P_dr > 0.05), np.sum(P_dr > 0)))

            for rot_ind, prob in zip(rot_inds, P_dr):
                if prob == 0:
                    continue
                if rot_ind not in finite_rot_inds:
                    finite_rot_inds[rot_ind] = {}
                    finite_rot_inds[rot_ind]["i_data"] = []
                    finite_rot_inds[rot_ind]["P_dr"] = []
                    finite_rot_inds[rot_ind]["rank"] = []
                finite_rot_inds[rot_ind]["i_data"].append(i_data)
                finite_rot_inds[rot_ind]["P_dr"].append(prob)
                finite_rot_inds[rot_ind]["rank"].append(COMM.rank)

        rot_inds_each_rank = COMM.gather(finite_rot_inds)
        rot_inds_global = None
        if COMM.rank == 0:
            rot_inds_global = utils.RotInds()
            for frir in rot_inds_each_rank:
               rot_inds_global.merge(frir)
        rot_inds_global = COMM.bcast(rot_inds_global)

        ##############################
        # compute the updated density
        ##############################
        #Wprime = np.zeros((256,256,256))
        #Wprime_wts = np.zeros((256,256,256))
        Rdr_getter.toggle_insert() # sets density to 0, creates a wts array, same len as densities and also sets to 0

        ###########################################################################
        # First, contributions from rot indices that exist on one rank only
        # For these, no MPI communication is necessary to get the tomogram updates
        ###########################################################################
        t = time.time()
        rots_on_one_rank = rot_inds_global.on_one_rank
        for i_rot_ind, rot_ind in enumerate(finite_rot_inds):
            printRf("Finite rot ind %d / %d" %(i_rot_ind, len(finite_rot_inds)))
            if rot_ind not in rots_on_one_rank:
                continue
            W_rt = np.zeros(num_data_pix)
            P_dr_sum = 0
            for i_data, P_dr in zip(finite_rot_inds[rot_ind]['i_data'], finite_rot_inds[rot_ind]['P_dr']):
                W_rt += rank_data[i_data]*P_dr
                P_dr_sum += P_dr
            W_rt /= P_dr_sum

            # insert
            Rdr_getter.trilinear_insertion(rot_ind, W_rt)
            #U = rotMats[rot_ind]
            #qcoords_rot = np.dot(U.T, qcoords.T).T
            ## TODO does the selection change
            #kji = np.floor((qcoords_rot - corners) / deltas)
            #out_of_bounds = np.logical_or(kji < 0, kji > const.NBINS - 2)
            #all_inbounds = ~np.any(out_of_bounds, axis=1)
            #trilinear_insertion(
            #    Wprime, Wprime_wts,
            #    vectors=np.ascontiguousarray(qcoords_rot[all_inbounds]),
            #    insert_vals=W_rt.ravel()[all_inbounds],
            #    x_min=const.X_MIN, x_max=const.X_MAX)
        tfinite_rot = time.time()-t
        COMM.barrier()
        print0f("done with Finite rot inds (%.4f sec)" % tfinite_rot)

        ###############################################################
        # Second, if a rot index was on multiple ranks,
        # a random rank is assigned to compute the tomogram update
        ###############################################################
        send_to = recv_from = None
        if COMM.rank==0:
            send_to, recv_from = rot_inds_global.tomogram_sends_and_recvs()
        send_to = COMM.bcast(send_to)
        recv_from = COMM.bcast(recv_from)
        # send_to and recv_from are dicts whose keys are COMM.rank

        send_req = []
        if COMM.rank in send_to:
            for dest, i_data, tag in send_to[COMM.rank]:
                req = COMM.isend(rank_data[i_data], dest=dest, tag=tag)
                send_req.append(req)

        if COMM.rank in recv_from:
            rot_inds_to_recv = recv_from[COMM.rank]['rot_inds']
            rot_inds_recv_info = recv_from[COMM.rank]['comms_info']
            for i_recv, (rot_ind, recv_info) in enumerate(zip(rot_inds_to_recv, rot_inds_recv_info)):
                printRf("recv %d (%d / %d)" % (rot_ind, i_recv+1, len(rot_inds_to_recv)))
                W_rt = np.zeros(num_data_pix)
                P_dr_sum = 0
                for source, P_dr, tag in recv_info:
                    assert source != COMM.rank, "though supported we dont permit self sending"
                    printRf("get from rank %d, tag=%d" % (source, tag))
                    shot_data = COMM.recv(source=source, tag=tag)
                    printRf("GOT from rank %d, tag=%d" % (source, tag))
                    W_rt += shot_data*P_dr
                    P_dr_sum += P_dr

                printRf("Done recv %d /%d" %(i_recv+1, len(rot_inds_to_recv)))
                assert rot_ind in finite_rot_inds # this is True by definition if you read the RotInds class method above
                for i_data, P_dr in zip(finite_rot_inds[rot_ind]['i_data'], finite_rot_inds[rot_ind]['P_dr']):
                    W_rt += rank_data[i_data] * P_dr
                    P_dr_sum += P_dr
                W_rt /= P_dr_sum

                # insert (same as above , consider making a method somehow)
                Rdr_getter.trilinear_insertion(rot_ind, W_rt)
                #U = rotMats[rot_ind]
                #qcoords_rot = np.dot(U.T, qcoords.T).T
                ## TODO does the selection change
                #kji = np.floor((qcoords_rot - corners) / deltas)
                #out_of_bounds = np.logical_or(kji < 0, kji > const.NBINS - 2)
                #all_inbounds = ~np.any(out_of_bounds, axis=1)
                #trilinear_insertion(
                #    Wprime, Wprime_wts,
                #    vectors=np.ascontiguousarray(qcoords_rot[all_inbounds]),
                #    insert_vals=W_rt.ravel()[all_inbounds],
                #    x_min=const.X_MIN, x_max=const.X_MAX)
        for req in send_req:
            req.wait()
        print0f("Done with tomogram send/recv")

        ##############################################
        # Finally, reduce /broadcast / normalize Wprime
        ##############################################
        Wprime = Rdr_getter.densities()
        Wprime_wts = Rdr_getter.wts()
        #Wprime2 = COMM.bcast(COMM.reduce(Wprime2))
        #Wprime2_wts = COMM.bcast(COMM.reduce(Wprime2_wts))
        COMM.barrier()  # barrier before reduce, so we can time just the reduce
        t = time.time()
        print0f("Reduce Wprime")
        Wprime = COMM.reduce(Wprime)
        tred = time.time()-t
        t = time.time()
        print0("Broadcast Wprime")
        Wprime = COMM.bcast(Wprime)
        tbro = time.time()-t
        print0f("Red: %.4f sec; Bro %.4f sec" %(tred, tbro))

        COMM.barrier()  # barrier before reduce, so we can time the reduce
        Wprime_wts = COMM.bcast(COMM.reduce(Wprime_wts))
        with np.errstate(divide='ignore', invalid='ignore'):
            Wprime = np.nan_to_num(Wprime / Wprime_wts)

        if apply_symmetry:
            print0f("symmetrizing")
            if COMM.rank==0:
                Wprime = utils.symmetrize(Wprime, reshape=False)
            Wprime = COMM.bcast(Wprime)

        # copy new density to the GPU for the next iteration
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
            plt.ylabel("$||W`-W||$",fontsize=11)
            plt.gca().set_yscale("log")
            figname = os.path.join(output_dir, "w_convergence%d.png" % num_iter)
            plt.savefig(figname)


if __name__=="__main__":
    main()
