
import os
import numpy as np
from simemc import utils, sim_utils, sim_const
from simemc.compute_radials import RadPros

phil_str="""
spotfinder.threshold.algorithm=dispersion
spotfinder.threshold.dispersion.gain=1
spotfinder.threshold.dispersion.kernel_size=[3,3]
spotfinder.threshold.dispersion.sigma_strong=3
spotfinder.threshold.dispersion.sigma_background=6
spotfinder.filter.min_spot_size=4
"""


def test_radial_background():
    water_img = sim_utils.get_water_scattering()

    qmap = utils.calc_qmap(sim_const.DETECTOR, sim_const.BEAM)
    qx,qy,qz = map(lambda x: x.ravel(), qmap)
    np.random.seed(0)
    C = sim_utils.random_crystal()
    print(C.get_U())

    SIM = sim_utils.get_noise_sim(0)
    Famp = sim_utils.get_famp()
    img = sim_utils.synthesize_cbf(
        SIM, C, Famp,
        dev_id=0,
        xtal_size=0.002, outfile=None, background=water_img, just_return_img=True )
    img = np.array([img])
    water_img = np.array([water_img])
    print("IMG MEAN=", img.mean())  # check mean to make sure its same each time script is run

    # detect peaks
    phil_file = "_test_radial_background.phil"
    with open(phil_file, "w") as o:
        o.write(phil_str)
    R = utils.refls_from_sims(img, sim_const.DETECTOR, sim_const.BEAM, phil_file=phil_file)
    os.remove(phil_file)

    refGeom = {"D": sim_const.DETECTOR, "B": sim_const.BEAM}
    radProMaker = RadPros(refGeom, numBins=1000)
    radProMaker.polarization_correction()
    radProMaker.solidAngle_correction()
    correction = radProMaker.POLAR * radProMaker.OMEGA
    correction /= correction.mean()

    # apply polarization/solid angle corrections to image
    img_uncorrected = img.copy()
    water_img_uncorrected = water_img.copy()

    img *= correction
    water_img *= correction

    rp_water = radProMaker.makeRadPro(
        data_pixels=water_img,
        strong_refl=R,
        apply_corrections=False, use_median=False)

    bg_water_img = radProMaker.expand_background_1d_to_2d(rp_water, radProMaker.img_sh, radProMaker.all_Qbins)
    diff_water_img = water_img - bg_water_img

    rp_water_diff = radProMaker.makeRadPro(
        data_pixels=diff_water_img,
        strong_refl=R,
        apply_corrections=False, use_median=False)
    assert np.allclose(rp_water_diff, 0)

    rp_water_uncor = radProMaker.makeRadPro(
        data_pixels=water_img_uncorrected,
        strong_refl=R,
        apply_corrections=False, use_median=False)

    bg_water_img_uncor = radProMaker.expand_background_1d_to_2d(rp_water_uncor, radProMaker.img_sh, radProMaker.all_Qbins)
    diff_water_img_uncor = water_img_uncorrected - bg_water_img_uncor

    # sum along slow-scan indices 1000-1500, these traces should have signifcant structure
    # if polarization and solid angle were not corrected for
    vals = diff_water_img[0, 1000:1500].mean(axis=0)
    vals_uncor = diff_water_img_uncor[0, 1000:1500].mean(axis=0)
    # the magnitude of these structures should be at least 10x greated in the uncorrected images
    factor = np.abs(vals_uncor).mean() / np.abs(vals).mean()
    assert factor > 10


    rp_img = radProMaker.makeRadPro(
        data_pixels=img,
        strong_refl=R,
        apply_corrections=False, use_median=False)
    bg_img = radProMaker.expand_background_1d_to_2d(rp_img, radProMaker.img_sh, radProMaker.all_Qbins)
    diff_img = img - bg_img

    rp_diff = radProMaker.makeRadPro(
        data_pixels=diff_img,
        strong_refl=R,
        apply_corrections=False, use_median=False)

    assert np.allclose(rp_diff,0)


if __name__=="__main__":
    test_radial_background()
