from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os
import h5py
import glob
import numpy as np

from dxtbx.model import ExperimentList
from simtbx.diffBragg.utils import image_data_from_expt


from simemc.compute_radials import RadPros


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


def load_emc_input(input_dir, dt=None):
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
