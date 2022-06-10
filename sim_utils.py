"""
NOTE:These methods are not intended to be general, and are specifically
for simulating lsyozyme scattering modeled after the experiment reported in
https://doi.org/10.1107/S205225251800903X
"""
from copy import deepcopy
import numpy as np
import os
from scipy.spatial.transform import Rotation

from dials.array_family import flex
from simtbx.modeling.forward_models import diffBragg_forward
from simtbx.nanoBragg import nanoBragg
from simtbx.diffBragg import utils as db_utils
from simtbx.nanoBragg import utils as nb_utils
from simtbx.diffBragg import hopper_utils

from simemc import sim_const as SC


def random_crystal(rand_state=None):
    Cryst = deepcopy(SC.CRYSTAL)
    randU = Rotation.random(random_state=rand_state)
    randU = randU.as_matrix()
    Cryst.set_U(randU.ravel())
    return Cryst


def get_famp():
    PDB = "4bs7.pdb"  # for P43212
    #PDB = "5k2d.pdb"  # for C2221
    # to retrieve the PDB, run `iotbx.fetch_pdb 4bs7` from cmdline
    if not os.path.exists(PDB):
        raise OSError("Download 4bs7.pdb using `iotbx.fecth_pdb 4bs7`")
    Famp = db_utils.get_complex_fcalc_from_pdb(PDB, dmin=SC.DMIN, dmax=SC.DMAX)
    Famp = Famp.as_amplitude_array()
    return Famp

def synthesize_cbf(
        noise_sim, CRYSTAL, Famp,
        dev_id, xtal_size, outfile=None, background=0,
        poly_perc=None):
    """

    :param noise_sim: nanoBragg instance, output of get_noise_sim
    :param CRYSTAL: dxtbx crystal, output of random_crystal
    :param Famp: structure factor, output of get_famp
    :param dev_id: gpu device Id
    :param xtal_size: size of crystal in mm (0.02 mm)
    :param outfile: write a cbf with this name
    :param background: scattering to add to the simulated img
    :return: optionally returns a numpy array, or else None
    """

    if poly_perc is not None:
        en0 = db_utils.ENERGY_CONV /SC.BEAM.get_wavelength()
        fwhm = en0*poly_perc*0.01
        energies, fluxes = hopper_utils.generate_gauss_spec(
            en0, fwhm, res=2,nchan=200, total_flux=SC.TOTAL_FLUX, 
            as_spectrum=False)
        K = SC.TOTAL_FLUX / fluxes.mean()
        fluxes *= K
    else:
        fluxes = [SC.TOTAL_FLUX]
        energies = [nb_utils.ENERGY_CONV / SC.BEAM.get_wavelength()]
    img = diffBragg_forward(
        CRYSTAL, SC.DETECTOR, SC.BEAM, Famp, energies, fluxes,
        oversample=SC.OVERSAMPLE, Ncells_abc=SC.NCELLS_ABC,
        mos_dom=SC.MOS_DOMS, mos_spread=SC.MOS_SPREAD, beamsize_mm=SC.BEAMSIZE,
        device_Id=dev_id,
        show_params=False, crystal_size_mm=xtal_size, printout_pix=None,
        verbose=0, default_F=0, interpolate=0, profile=SC.PROFILE,
        mosaicity_random_seeds=None,
        show_timings=False,
        nopolar=False, diffuse_params=None)

    if len(img.shape)==3:
        img = img[0]

    img_with_bg = img + background

    noise_sim.raw_pixels *= 0
    noise_sim.raw_pixels += flex.double((img_with_bg).ravel())
    noise_sim.add_noise()

    if outfile is not None:
        noise_sim.to_cbf(outfile)
        np.savez(outfile+".npz",
                 A=CRYSTAL.get_A(), B=CRYSTAL.get_B(), U=CRYSTAL.get_U())
    return noise_sim.raw_pixels.as_numpy_array()


def get_water_scattering():
    water = nb_utils.sim_background(
        SC.DETECTOR, SC.BEAM,
        wavelengths=[SC.BEAM.get_wavelength()],
        wavelength_weights=[1],
        total_flux=SC.TOTAL_FLUX, pidx=0,
        beam_size_mm=SC.BEAMSIZE, Fbg_vs_stol=None,
        sample_thick_mm=SC.WATER_PATH,
        density_gcm3=1, molecular_weight=18)

    fast, slow = SC.DETECTOR[0].get_image_size()
    img_sh = slow, fast
    water = np.reshape(water, img_sh)
    return water


def delete_noise_sim(noise_sim):
    noise_sim.free_all()
    del noise_sim


def get_noise_sim(calib_noise_percent):
    noise_sim = nanoBragg(detector=SC.DETECTOR, beam=SC.BEAM)
    noise_sim.beamsize_mm = SC.BEAMSIZE
    noise_sim.detector_calibration_noise_pct = calib_noise_percent
    noise_sim.exposure_s = 1
    noise_sim.calib_seed=0
    noise_sim.seed=0
    noise_sim.flux = SC.TOTAL_FLUX
    noise_sim.adc_offset_adu =0
    noise_sim.detector_psf_kernel_radius_pixels = 5
    noise_sim.detector_psf_fwhm_mm =0
    noise_sim.quantum_gain = 1
    noise_sim.readout_noise_adu = 0
    return noise_sim
