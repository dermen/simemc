# coding: utf-8
import h5py
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument("--gtmerge", type=str)
parser.add_argument("W", nargs="+")
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
FI = Famp.as_intensity_array()
Fmap = {hkl: val for hkl,val in zip(FI.indices(), FI.data())}

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
        gt_vals = [Fmap[h] for h in hcommon]
    else:
        assert set(dataMap.keys()).intersection(Fmap.keys()) == hcommon
    vals = [dataMap[h] for h in hcommon]
    c = pearsonr(vals, gt_vals)[0]
    print(wname, c)
    plot(gt_vals, vals,'.', label=wname + " --- CC=%.4f" % c)

if Fmtz_map is not None:
    c = pearsonr(mtz_vals, gt_vals)[0]
    plot( gt_vals, mtz_vals, '.', label="%s; --- CC= %.4f" % (args.mtz, c))
xlabel("ground truth $|F|^2$")
ylabel("analysis $|F|^2$")
gca().legend()
gca().set_xscale('log')
gca().set_yscale('log')
show()

