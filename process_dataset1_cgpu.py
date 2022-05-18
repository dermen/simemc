
import sys
import os
from mpi4py import MPI
from copy import deepcopy
COMM = MPI.COMM_WORLD

import numpy as np
from scipy.ndimage import median_filter as mf
from dials.array_family import flex
import time

from simtbx.modeling.forward_models import diffBragg_forward
from dxtbx.model import Crystal
from simemc import mpi_utils, utils
from simemc import const
from simemc.compute_radials import RadPros
from simtbx.diffBragg import utils as db_utils
from dxtbx.model import ExperimentList
from simemc.emc import lerpy


outdir=sys.argv[1]
highRes = 4
ndevice = 8
mf_filter_sh=8,8
datadir = os.environ["CSCRATCH"] + "/dataset_1"
input_file =    datadir + "/exp_ref.txt"
ref_geom_file = datadir + "/split_0000.expt"
mask_file =     datadir + "/test_mask.pkl"
quat_file = "/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules/simemc/quatgrid/c-quaternion120.bin"
ave_ucell = 68.48, 68.48, 104.38, 90,90,90
hcut=0.025
min_pred=3
niter=100
num_radial_bins = 1000
maxN = 2

mpi_utils.make_dir(outdir)

ucell_man = db_utils.manager_from_params(ave_ucell)
Brecip = ucell_man.B_recipspace

GeoExpt = ExperimentList.from_file(ref_geom_file,False)
BEAM = GeoExpt[0].beam
DET = db_utils.strip_thickness_from_detector(GeoExpt[0].detector)
assert BEAM is not None
assert DET is not None
MASK = db_utils.load_mask(mask_file)
DEV_ID = COMM.rank % ndevice
this_ranks_imgs, this_ranks_refls = mpi_utils.mpi_load_exp_ref(input_file, maxN=maxN)

print0 = mpi_utils.print0f
print0("Creating radial profile maker!")
refGeom = {"D": DET, "B": BEAM}
radProMaker = RadPros(refGeom, numBins=num_radial_bins, maskFile=MASK)
radProMaker.polarization_correction()
radProMaker.solidAngle_correction()


correction = radProMaker.POLAR * radProMaker.OMEGA
correction /= correction.mean()
for i_img in range(len(this_ranks_imgs)):
    this_ranks_imgs[i_img] *= correction

rots, wts = utils.load_quat_file(quat_file)

for i,R in enumerate(this_ranks_refls):
    Q = db_utils.refls_to_q(R, DET, BEAM)
    dspace = 1 / np.linalg.norm(Q, axis=1)
    sel = flex.bool(dspace >= highRes)
    Rsel = R.select(sel)
    assert len(Rsel) >= 3
    this_ranks_refls[i] = Rsel

this_ranks_prob_rot = utils.get_prob_rot(DEV_ID, this_ranks_refls, rots,
                                         Bmat_reference=Brecip, hcut=hcut, min_pred=min_pred,
                                        verbose=COMM.rank==0, detector=DET,beam=BEAM)

# fit background to each image; estimate signal level per image
this_ranks_bgs = []
this_ranks_signal_levels = []
for i_img, (img, R) in enumerate(zip(this_ranks_imgs, this_ranks_refls)):
    t = time.time()
    radial_pro, bg_mask = radProMaker.makeRadPro(
        data_pixels=img,
        strong_refl=R,
        apply_corrections=False, use_median=True, return_mask=True)

    img_background = radProMaker.expand_radPro(radial_pro)
    img_filled = img.copy()
    img_filled[~bg_mask] = img_background[~bg_mask]
    bg = mf(img_filled[0], mf_filter_sh)

    this_ranks_bgs.append(np.array([bg]))
    t = time.time()-t

    signal_level = utils.signal_level_of_image(R, img)
    if signal_level ==0:
        print("WARNING, shot has 0 signal level")
    bg_signal_level = utils.signal_level_of_image(R, this_ranks_bgs[-1])
    assert bg_signal_level <= signal_level
    this_ranks_signal_levels.append(signal_level-bg_signal_level)

    # TODO subtract bg signal level ? 
    print0("Done with background image %d / %d (Took %f sec to fit bg) (signal=%f, bg_sig=%f)" % (i_img+1, len(this_ranks_imgs), t, signal_level, bg_signal_level))


all_ranks_signal_levels = COMM.reduce(this_ranks_signal_levels)
ave_signal_level = None
if COMM.rank==0:
    ave_signal_level = np.mean(all_ranks_signal_levels)
ave_signal_level = COMM.bcast(ave_signal_level)
# let the initial density estimate be constant gaussians (add noise?)
Wstart = utils.get_W_init(ucell_p=ave_ucell)
Wstart /= Wstart.max()
Wstart *= ave_signal_level

# get the emc trilerp instance
qmap = utils.calc_qmap(DET, BEAM)
qx,qy,qz = map(lambda x: x.ravel(), qmap)
qcoords = np.vstack([qx,qy,qz]).T

Winit = np.zeros(const.DENSITY_SHAPE, np.float32)
corner, deltas = utils.corners_and_deltas(const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
maxRotInds = 1000000  # TODO make this a property of lerpy, and catch if trying to pass more rot inds

num_pix = MASK.size
L = lerpy()
rots = rots.astype(L.array_type)
Winit = Winit.astype(L.array_type)
qcoords = qcoords.astype(L.array_type)

L.allocate_lerpy(DEV_ID, rots.ravel(), Winit.ravel(), num_pix,
                 corner, deltas, qcoords.ravel(),
                 maxRotInds, num_pix)

# convert the type of images to match the lerpy instance array_type (prevents annoying warnings)
for i, img in enumerate(this_ranks_imgs):
    this_ranks_imgs[i] = img.astype(L.array_type).ravel()
    this_ranks_bgs[i] = this_ranks_bgs[i].astype(L.array_type).ravel()

# set the starting density
L.toggle_insert()
L.update_density(Wstart)
if COMM.rank==0:
    np.save(os.path.join(outdir, "Starting_density_relp"), Wstart)

init_shot_scales = np.ones(len(this_ranks_imgs))

# mask pixels that are outside the shell
# TODO: mask pixels not in bounds
inbounds = utils.qs_inbounds(qcoords, const.DENSITY_SHAPE, const.X_MIN, const.X_MAX)
inbounds = inbounds.reshape(this_ranks_imgs[0].shape)
print0("INIT SHOT SCALES:", init_shot_scales)
SHOT_MASK = inbounds*MASK.ravel()

class Model:
    def __init__(self):
        self.fluxes = [1e12]
        self.energies = [db_utils.ENERGY_CONV/BEAM.get_wavelength()]
        self.Famp = db_utils.make_miller_array("P43212", ave_ucell)
        hall = self.Famp.space_group_info().type().hall_symbol()
        ucell_man = db_utils.manager_from_params(ave_ucell)
        Bmat = np.reshape(ucell_man.B_realspace, (3, 3))
        a1, a2, a3 = map(tuple, [Bmat[:, 0], Bmat[:, 1], Bmat[:, 2]])
        cryst_descr = {'__id__': 'crystal',
                       'real_space_a': a1,
                       'real_space_b': a2,
                       'real_space_c': a3,
                       'space_group_hall_symbol': hall}

        self.CRYST= Crystal.from_dict(cryst_descr)

    def sanity_check_model(self, umat):
        C= deepcopy(self.CRYST)
        umat = tuple(umat.ravel())
        C.set_U(umat)
        model = diffBragg_forward(
            C, DET, BEAM, self.Famp, self.energies, self.fluxes,
            oversample=1, Ncells_abc=(20,20,20),
            mos_dom=1, mos_spread=0, beamsize_mm=0.001,
            device_Id=0,
            show_params=False, crystal_size_mm=0.005, printout_pix=None,
            verbose=0, default_F=0, interpolate=0, profile="gauss",
            mosaicity_random_seeds=None,
            show_timings=False,
            nopolar=False, diffuse_params=None)
        return model

M = Model()

for i,img in enumerate(this_ranks_imgs):
    R = this_ranks_refls[i]
    x, y, z = R['xyzobs.px.value'].parts()

    x-=0.5
    y-=0.5
    models =[]
    new_group = True
    prev_model = None
    for rot_id in this_ranks_prob_rot[i]:

        umat = rots[rot_id]
        model = M.sanity_check_model(umat)
        if prev_model is None:
            prev_model = model
            new_group = True
        else:
            if np.allclose(prev_model, model):
                new_group = False
            else:
                new_group = True
                prev_model = model
        if new_group:

            Rpred = utils.refls_from_sims(model, DET, BEAM, thresh=1e-4)
            utils.label_strong_reflections(Rpred, R, 1)
            nclose = np.sum(Rpred['is_strong'])
            Rclose = Rpred.select(Rpred["is_strong"])
            xobs, yobs, _ = Rclose["xyzobs.px"] .parts()
            xcal, ycal,_ = Rclose["xyzcal.px"].parts()

            pred_dist = np.mean(np.sqrt((xobs-xcal)**2 + (yobs-ycal)**2))
            print("model %d, nclose=%d, predoffset=%.4f pix" %(rot_id, nclose, pred_dist ))

    from IPython import embed;embed()



# make the mpi emc object
emc = mpi_utils.EMC(L, this_ranks_imgs, this_ranks_prob_rot,
                    shot_mask=SHOT_MASK,
                    shot_background=this_ranks_bgs,
                    min_p=1e-5,
                    outdir=outdir,
                    beta=1,
                    shot_scales=init_shot_scales,
                    refine_scale_factors=True,
                    ave_signal_level=ave_signal_level,
                    scale_update_method="bfgs",
                    density_update_method="lbfgs",
                    ucell_p=ave_ucell)

# run emc for specified number of iterations
emc.do_emc(niter)
