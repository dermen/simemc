import h5py
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("W", type=str, nargs="+", help="Witer files")
parser.add_argument("mtz", type=str, default=None)
args = parser.parse_args()

from iotbx.reflection_file_reader import any_reflection_file
from pylab import *
from scipy.stats import spearmanr, pearsonr
from simemc import utils

dens_dim=256
max_q=0.25
highRes=1/max_q
CC = pearsonr
symbol = "P43212"
ucell_p = 68.48, 68.48, 104.38, 90,90,90

Fmtz = any_reflection_file(args.mtz).as_miller_arrays()[0]
Fmtz = Fmtz.as_amplitude_array()
Fmtz = Fmtz.resolution_filter(d_min=highRes)
#ucell_p = Fmtz.unit_cell().parameters()
Fmtz_map = {hkl: val for hkl,val in zip(Fmtz.indices(), Fmtz.data())}
D = Fmtz.d_spacings()
dspace_map = {hkl:val for hkl, val in zip(D.indices(), D.data())}
nbin = 15

hcommon = None
wnames = np.array(args.W)
idx = [0 if "init" in w else int(w.split("Witer")[1].split(".")[0]) for w in wnames]
idx = np.array(idx)
print(idx)
order = np.argsort(idx)
wnames = wnames[order]
idx = idx[order]
all_cc_in_res_bin = []
cc_per_wname = []
for wname in wnames:
    try:
        W = h5py.File(wname, "r")['Wprime'][()]
    except OSError:
        W = np.load(wname)[()].reshape([dens_dim]*3)
    # TODO store ucell in W
    h,I = utils.integrate_W(W, dens_dim, max_q, ucell_p, symbol)
    I = np.array(I)
    is_pos = I >0
    I = I[is_pos]
    h = np.array(h)[is_pos]
    #assert np.all(I > 0)
    I = np.sqrt(I)
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
        corr_at_res.append(CC(vals_in_bin, mtz_vals_in_bin)[0])
    all_cc_in_res_bin.append(corr_at_res)

    c = CC(vals, mtz_vals)[0]
    print(wname, c)
    plot(vals, mtz_vals,'.', label=wname + " --- CC=%.4f" % c)
    cc_per_wname.append(c)

xlabel("EMC $|F|^2$", fontsize=14)
ylabel("cctbx $|F|^2$", fontsize=14)
gca().legend(markerscale=2, prop={"size":14})
gca().grid(which='both', ls='--', lw=0.5)
gca().set_xscale('log')
gca().set_yscale('log')
gca().tick_params(labelsize=12)
subplots_adjust(left=.15, bottom=.15,right=.95, top=.91)
gca().tick_params(labelsize=12)

figure()
plot( res_bin_centers, corr_at_res, 's' , color='tomato')
xlabel("resolution ($\AA$)", fontsize=13)
ylabel("CC (cctbx vs EMC)", fontsize=13)
gca().tick_params(labelsize=12)
gca().grid(1, ls='--', lw=0.5)

if len(wnames) > 1:

    figure()
    plot(idx, cc_per_wname)

    xlabel("EMC iteration", fontsize=13)
    ylabel("CC (cctbx vs EMC)")

    figure()
    if 0 in idx:  # dont display imshow for first iteration
        all_cc_in_res_bin = all_cc_in_res_bin[1:]
        idx = idx[1:]
    ncc = len(all_cc_in_res_bin)
    imshow(all_cc_in_res_bin, extent=(highRes,res_bin_centers[-1], 0,ncc), cmap='hot')
    
    gca().set_yticklabels([str(i) for i in idx[::-1]])
    gca().set_yticks(arange(ncc)+0.5)
    colorbar()
    xlabel("resolution ($\AA$)", fontsize=13)
    ylabel("EMC iteraton")


show()

