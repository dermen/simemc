import h5py
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("W", type=str, nargs="+", help="Witer files")
parser.add_argument("mtz", type=str, default=None)
args = parser.parse_args()

from iotbx.reflection_file_reader import any_reflection_file
from pylab import *
from scipy.stats import pearsonr
from simemc import utils



Fmtz = any_reflection_file(args.mtz).as_miller_arrays()[0]
ucell_p = Fmtz.unit_cell().parameters()
Fmtz_map = {hkl: val for hkl,val in zip(Fmtz.indices(), Fmtz.data())}
D = Fmtz.d_spacings()
dspace_map = {hkl:val for hkl, val in zip(D.indices(), D.data())}
nbin = 10

hcommon = None
for wname in args.W:
    try:
        W = h5py.File(wname, "r")['Wprime'][()]
    except OSError:
        W = np.load(wname)[()].reshape((256,256,256))
    # TODO store ucell in W
    h,I = utils.integrate_W(W, ucell_p=ucell_p)
    dataMap = {hkl:val for hkl,val in zip(list(map(tuple,h)), I)}
    if hcommon is None:
        hcommon = set(dataMap.keys()).intersection(Fmtz_map.keys())
        mtz_vals = array([Fmtz_map[h] for h in hcommon])
    else:
        assert set(dataMap.keys()).intersection(Fmtz_map.keys()) == hcommon
    vals = array([dataMap[h] for h in hcommon])
    dspace = array([dspace_map[h] for h in hcommon])
    res_bins = linspace(dspace.min()-1e-6, dspace.max()+1e-6, nbin)
    res_bin_id = np.digitize(dspace, res_bins)
    corr_at_res = []
    res_bin_centers = (res_bins[1:] + res_bins[:-1])*0.5
    for bin_id in range(1,nbin):
        is_in_bin = res_bin_id==bin_id
        vals_in_bin = vals[is_in_bin]
        mtz_vals_in_bin = mtz_vals[is_in_bin]
        corr_at_res.append(pearsonr(vals_in_bin, mtz_vals_in_bin)[0])

    c = pearsonr(vals, mtz_vals)[0]
    print(wname, c)
    plot(vals, mtz_vals,'.', label=wname + " --- CC=%.4f" % c)

xlabel("EMC $|F|^2$", fontsize=14)
ylabel("cctbx $|F|^2$", fontsize=14)
gca().legend(markerscale=2, prop={"size":12})
gca().grid(which='both', ls='--', lw=0.5)
gca().set_xscale('log')
gca().set_yscale('log')
gca().tick_params(labelsize=12)
subplots_adjust(left=.15, bottom=.15,right=.95, top=.91)

figure()
plot( res_bin_centers, corr_at_res, 's')


show()

