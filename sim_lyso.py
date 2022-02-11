# coding: utf-8
from dials.array_family import flex
import numpy as np
from dxtbx.model import ExperimentList
from simtbx.nanoBragg import utils
from mpi4py import MPI
COMM = MPI.COMM_WORLD
from simtbx.diffBragg import utils as db_utils
from simtbx.nanoBragg import nanoBragg
from simtbx.modeling.forward_models import diffBragg_forward
from simtbx.nanoBragg.tst_nanoBragg_multipanel import cryst as CRYSTAL
from scipy.spatial.transform import Rotation
import os
import time
import sys
np.random.seed(COMM.rank)

NUM_DEV=1
total_flux=1e12
beam_size=0.01
# to retrieve the PDB, run `iotbx.fetch_pdb 4bs7` from cmdline
PDB = "4bs7.pdb"
num_shots = 5
XTAL_SIZE = 0.02  # mm

OUTDIR = os.path.join( sys.argv[1], "rank%d" % COMM.rank)
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
COMM.barrier()



El = ExperimentList.from_file("geom.expt", False)
DETECTOR = El.detectors()[0]
DETECTOR = db_utils.strip_thickness_from_detector(DETECTOR)

BEAM = El.beams()[0]
ave_wave = BEAM.get_wavelength()
energies = [utils.ENERGY_CONV / ave_wave]

Famp = db_utils.get_complex_fcalc_from_pdb(PDB, dmin=1.4, dmax=50)
Famp = Famp.as_amplitude_array()


fast, slow = DETECTOR[0].get_image_size()
img_sh = slow, fast

num_en = len(energies)
fluxes = [total_flux]

water = None
if COMM.rank==0:
    print("Simulating water scattering...")
    water = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size,
        Fbg_vs_stol=None, sample_thick_mm=1, density_gcm3=1, molecular_weight=18)
    
water = COMM.bcast(water)
water = np.reshape(water, img_sh)
#water_bkgrnd = flex.double(np.load("water.npy").reshape(img_sh))
#water_bkgrnd = np.load("/global/cfs/cdirs/m3992/dermen/water.npy").reshape(img_sh)
#bg = [water_bkgrnd]

dev_id = COMM.rank %NUM_DEV 
SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
SIM.beamsize_mm = beam_size
SIM.detector_calibration_noise_pct = 3
SIM.exposure_s = 1
SIM.flux = total_flux
SIM.adc_offset_adu =0
SIM.detector_psf_kernel_radius_pixels = 5
SIM.detector_psf_fwhm_mm = 0
SIM.quantum_gain = 1
SIM.readout_noise_adu = 0

for i_shot in range(num_shots):
    if i_shot % COMM.size != COMM.rank: 
        continue
    randU = Rotation.random()
    randU = randU.as_matrix()
    CRYSTAL.set_U(randU.ravel())
    CRYSTAL = COMM.bcast(CRYSTAL)
    #if COMM.rank==0:
    print("Simulating shot %d / %d on device %d " % (i_shot+1, num_shots, dev_id), flush=True)

    img = diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies, 
        fluxes,
        oversample=1, Ncells_abc=(30,30,40),
        mos_dom=1, mos_spread=0, beamsize_mm=beam_size, device_Id=dev_id,
        show_params=False, crystal_size_mm=XTAL_SIZE, printout_pix=None,
        verbose=0, default_F=0, interpolate=0, profile="gauss",
        mosaicity_random_seeds=None,
        show_timings=False, #COMM.rank==0,
        nopolar=False, diffuse_params=None)
    
    if len(img.shape)==3:
        img = img[0]

    img_with_bg = img+water
    
    SIM.raw_pixels *= 0
    SIM.raw_pixels += flex.double((img_with_bg).ravel())
    SIM.add_noise()
    #img_w_noise = SIM.raw_pixels.as_numpy_array().reshape(img_sh)

    outfile = os.path.join(OUTDIR, "shot%d.cbf" % i_shot)
    SIM.to_cbf(outfile)
    np.savez(os.path.join(OUTDIR, "shot%d" % i_shot),
             A=CRYSTAL.get_A(), B=CRYSTAL.get_B(), U=CRYSTAL.get_U)

SIM.free_all()
del SIM
