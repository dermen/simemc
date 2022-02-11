
from mpi4py import MPI
COMM = MPI.COMM_WORLD
if COMM.rank==0:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("nshot", type=int, help="number of shots to simulate")
    parser.add_argument("outdir", type=str, help="Path to an output folder, will be created if non-existent")
    parser.add_argument("--ndev", default=1, type=int, help="number of GPUs per compute node")
    parser.add_argument("--sparse", action="store_true",
                        help="Generate sparse patterns with few spots, only matters if the flag `--no-water` is no present")
    parser.add_argument("--no-water", dest="no_water",action="store_true", help="Dont include background")
    parser.add_argument("--no-calib-noise", dest="no_calib", action="store_true", help="No per-pixel gain errors")
    args = parser.parse_args()
else:
    args = None
args = COMM.bcast(args)

import numpy as np
import os
from simemc import sim_utils


if __name__=="__main__":
    np.random.seed(COMM.rank)

    ###########################
    # variables
    NUM_DEV=args.ndev
    num_shots = args.nshot
    if args.sparse:
        XTAL_SIZE = 0.02  # mm
    else:
        XTAL_SIZE = 0.002
    calib_noise_percent = 0 if args.no_calib else 3
    add_background= not args.no_water
    #######################################

    OUTDIR = os.path.join( args.outdir, "rank%d" % COMM.rank)
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    COMM.barrier()

    Famp = sim_utils.get_famp()

    water = 0
    if add_background and COMM.rank==0:
        if COMM.rank==0:
            print("Simulating water scattering...")
            water = sim_utils.get_water_scattering()
        water = COMM.bcast(water)

    dev_id = COMM.rank %NUM_DEV
    SIM = sim_utils.get_noise_sim(calib_noise_percent)

    for i_shot in range(num_shots):
        if i_shot % COMM.size != COMM.rank:
            continue
        ROT_CRYSTAL = sim_utils.random_crystal()

        print("Simulating shot %d / %d on device %d " % (i_shot+1, num_shots, dev_id), flush=True)

        outfile = os.path.join(OUTDIR, "shot%d.cbf" % i_shot)

        sim_utils.synthesize_cbf(
            SIM, ROT_CRYSTAL, Famp,
            dev_id, XTAL_SIZE, outfile, water)

    sim_utils.delete_noise_sim(SIM)
