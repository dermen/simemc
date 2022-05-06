# coding: utf-8
import h5py
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("W", type=str, nargs="+", help="Witer files")
parser.add_argument("--mtz", type=str, default=None)
args = parser.parse_args()

from simtbx.diffBragg import utils as db_utils
from simemc import sim_utils
from simemc import utils
import numpy as np


Fmtz_map = None
if args.mtz is not None:
    from iotbx.reflection_file_reader import any_reflection_file
    Fmtz = any_reflection_file(args.mtz).as_miller_arrays()[0]
    Fmtz_map = {hkl: val for hkl,val in zip(Fmtz.indices(), Fmtz.data())}

Fcalc = sim_utils.get_famp()
Famp = Fcalc.as_amplitude_array()
dspace = Famp.d_spacings()
FI = Famp.as_intensity_array()
Fmap = {hkl: val for hkl,val in zip(FI.indices(), FI.data())}
dspace_map = {hkl: val for hkl,val in zip(dspace.indices(), dspace.data())}

from pylab import *
hcommon = None
gt_vals = None
from scipy.stats import pearsonr
for wname in args.W:
    try:
        W = h5py.File(wname, "r")['Wprime'][()]
    except OSError:
        W = np.load(wname)[()].reshape((256,256,256))
    h,I = utils.integrate_W(W)
    dataMap = {hkl:val for hkl,val in zip(list(map(tuple,h)), I)}
    if hcommon is None:
        hcommon = set(dataMap.keys()).intersection(Fmap.keys())
        if Fmtz_map is not None:
            hcommon = hcommon.intersection(Fmtz_map.keys())
            mtz_vals = [Fmtz_map[h] for h in hcommon]
        gt_vals = np.array([Fmap[h] for h in hcommon])
    else:
        assert set(dataMap.keys()).intersection(Fmap.keys()) == hcommon
    vals = array([dataMap[h] for h in hcommon])
    dspace = array([dspace_map[h] for h in hcommon])
    nbin = 10
    res_bins = linspace(dspace.min()-1e-6, dspace.max()+1e-6, nbin)
    res_bin_id = np.digitize(dspace, res_bins)
    corr_at_res = []
    res_bin_centers = (res_bins[1:] + res_bins[:-1])*0.5
    for bin_id in range(1,nbin):
        is_in_bin = res_bin_id==bin_id
        vals_in_bin = vals[is_in_bin]
        gt_vals_in_bin = gt_vals[is_in_bin]
        corr_at_res.append(pearsonr(vals_in_bin, gt_vals_in_bin)[0])

    c = pearsonr(vals, gt_vals)[0]
    print(wname, c)
    plot(gt_vals, vals,'.', label=wname + " --- CC=%.4f" % c)


#np.savez("corr_vals", gt=gt_vals, emc=vals, mtz=mtz_vals)
if Fmtz_map is not None:
    c = pearsonr(mtz_vals, gt_vals)[0]
    plot( gt_vals, mtz_vals, '.', label="%s; --- CC= %.4f" % (args.mtz, c))
xlabel("ground truth $|F|^2$", fontsize=14)
ylabel("analysis $|F|^2$", fontsize=14)
gca().legend(markerscale=2, prop={"size":12})
gca().grid(which='both', ls='--', lw=0.5)
gca().set_xscale('log')
gca().set_yscale('log')
gca().tick_params(labelsize=12)
subplots_adjust(left=.15, bottom=.15,right=.95, top=.91)

figure()
plot( res_bin_centers, corr_at_res, 's')


show()

