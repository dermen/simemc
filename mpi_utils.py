from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os
import h5py
import glob
import numpy as np
from simtbx.diffBragg import mpi_logger
import logging

from dxtbx.model import ExperimentList
from simtbx.diffBragg.utils import image_data_from_expt


from simemc.compute_radials import RadPros
import logging


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



def print_profile(stats, timed_methods, loggerName='simemc.profile'):
    #PROFILE_LOGGER = logging.getLogger(loggerName)
    for method in stats.timings.keys():
        filename, header_ln, name = method
        if name not in timed_methods:
            continue
        info = stats.timings[method]
        printR("\n")
        printR("FILE: %s" % filename)
        if not info:
            #PROFILE_LOGGER.info("<><><><><><><><><><><><><><><><><><><><><><><>")
            #PROFILE_LOGGER.info("METHOD %s : Not profiled because never called" % (name))
            #PROFILE_LOGGER.info("<><><><><><><><><><><><><><><><><><><><><><><>")
            continue
        unit = stats.unit

        line_nums, ncalls, timespent = zip(*info)
        fp = open(filename, 'r').readlines()
        total_time = sum(timespent)
        header_line = fp[header_ln-1][:-1]
        #PROFILE_LOGGER.info(header_line)
        #PROFILE_LOGGER.info("TOTAL FUNCTION TIME: %f ms" % (total_time*unit*1e3))
        #PROFILE_LOGGER.info("<><><><><><><><><><><><><><><><><><><><><><><>")
        #PROFILE_LOGGER.info("%5s%14s%9s%10s" % ("Line#", "Time", "%Time", "Line" ))
        #PROFILE_LOGGER.info("%5s%14s%9s%10s" % ("", "(ms)", "", ""))
        #PROFILE_LOGGER.info("<><><><><><><><><><><><><><><><><><><><><><><>")
        print(header_line)
        printR("TOTAL FUNCTION TIME: %f ms" % (total_time*unit*1e3))
        printR("<><><><><><><><><><><><><><><><><><><><><><><>")
        printR("%5s%14s%9s%10s" % ("Line#", "Time", "%Time", "Line" ))
        printR("%5s%14s%9s%10s" % ("", "(ms)", "", ""))
        printR("<><><><><><><><><><><><><><><><><><><><><><><>")
        for i_l, l in enumerate(line_nums):
            frac_t = timespent[i_l] / total_time * 100.
            line = fp[l-1][:-1]
            #PROFILE_LOGGER.info("%5d%14.2f%9.2f%s" % (l, timespent[i_l]*unit*1e3, frac_t, line))
            printR("%5d%14.2f%9.2f%s" % (l, timespent[i_l]*unit*1e3, frac_t, line))


def do_emc(L, shots, prob_rots):
    """
    run emc, the resulting density is stored in the L (emc.lerpy) object
    see call to this method in ests/test_emc_iteration
    :param L: instance of emc.lerpy in insert mode
    :param shots: array of images (Nshot, Npanel, slowDim, fastDim)
    :param prob_rots: list of array containing indices of probable orientations
    :return: optimized density. Also L instance should be modified in-place
    """
    raise NotImplementedError()
