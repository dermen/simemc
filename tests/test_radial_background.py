import numpy as np
from simemc import utils, const, sim_utils, sim_const
from simemc.compute_radials import RadPros
import pytest
import os

phil_str="""
spotfinder.threshold.algorithm=dispersion
spotfinder.threshold.dispersion.gain=1
spotfinder.threshold.dispersion.kernel_size=[3,3]
spotfinder.threshold.dispersion.sigma_strong=3
spotfinder.threshold.dispersion.sigma_background=6
spotfinder.filter.min_spot_size=4
"""


@pytest.mark.skip(reason="in development")
def test_background():
    water_img = sim_utils.get_water_scattering()
    gpu_device = 0
    num_rot_mats = 10000000
    #maxRotInds = 10000
    maxRotInds = 5
    max_num_strong_spots = 1000

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
    print("IMG MEAN=", img.mean())  # check mean to make sure its same each time script is run

    # detect peaks
    phil_file = "_test_radial_background.phil"
    with open(phil_file, "w") as o:
        o.write(phil_str)
    R = utils.refls_from_sims(np.array([img]), sim_const.DETECTOR, sim_const.BEAM, phil_file=phil_file)
    os.remove(phil_file)

    refGeom = {"D": sim_const.DETECTOR, "B": sim_const.BEAM}
    radProMaker = RadPros(refGeom, numBins=1500)
    radProMaker.polarization_correction()
    radProMaker.solidAngle_correction()
    correction = radProMaker.POLAR * radProMaker.OMEGA
    correction /= correction.mean()

    img *= correction[0]
    water_img *= correction[0]
    radPro_median = radProMaker.makeRadPro(
        data_pixels=np.array([water_img]),
        strong_refl=R,
        apply_corrections=False, use_median=True)

    radPro_mean = radProMaker.makeRadPro(
        data_pixels=np.array([water_img]),
        strong_refl=R,
        apply_corrections=False, use_median=False)

    bgImage_median = radProMaker.expand_background_1d_to_2d(radPro_median, radProMaker.img_sh, radProMaker.all_Qbins)

    bgImage_mean = radProMaker.expand_background_1d_to_2d(radPro_mean, radProMaker.img_sh, radProMaker.all_Qbins)


if __name__=="__main__":
    test_background()
