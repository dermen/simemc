import h5py
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("W", type=str, nargs="+", help="Witer files")
parser.add_argument("ref", type=str, default=None, help="can be an MTZ file or a PDB")
parser.add_argument("--nbin", type=int, default=10)
parser.add_argument("--toggle", action="store_true")
args = parser.parse_args()

from iotbx.reflection_file_reader import any_reflection_file
from pylab import *
from scipy.stats import spearmanr, pearsonr
from simemc import utils

max_q=0.25
highRes=1/max_q
CC = pearsonr
symbol = "P43212"
ucell_p = 68.48, 68.48, 104.38, 90,90,90

from simtbx.diffBragg import utils as db_utils
if args.ref.endswith("mtz"):
    Fref = any_reflection_file(args.ref).as_miller_arrays()[0]
    if Fref.is_xray_intensity_array():
        Fref = Fref.as_amplitude_array()
else:
    assert args.ref.endswith("pdb")
    Fref = db_utils.get_complex_fcalc_from_pdb(args.ref)
    Fref = Fref.as_amplitude_array()

# get the unit cell from the first file:
H5 = h5py.File(args.W[0], "r")
ucell = tuple(H5['ucell'][()])
dens_dim= H5["Wprime"] .shape[0]

Fref = Fref.resolution_filter(d_min=highRes)
#ucell_p = Fref.unit_cell().parameters()
Fref_map = {hkl: val for hkl,val in zip(Fref.indices(), Fref.data())}
D = Fref.d_spacings()
dspace_map = {hkl:val for hkl, val in zip(D.indices(), D.data())}
nbin = args.nbin

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
    print(dens_dim, max_q)
    h,I = utils.integrate_W(W, dens_dim, max_q, ucell_p, symbol)
    ma = utils.integrate_W(W, dens_dim, max_q, ucell_p, symbol, kernel_iters=1, conn=1).as_amplitude_array().resolution_filter(d_min=1/max_q)
    ma = ma.average_bijvoet_mates()
    I = np.array(I)
    is_pos = I >0
    I = I[is_pos]
    h = np.array(h)[is_pos]
    #assert np.all(I > 0)
    I = np.sqrt(I)
    dataMap = {hkl: val for hkl,val in zip(list(map(tuple,h)), I)}
    dataMap = {hkl:val for hkl,val in zip(ma.indices(), ma.data())}
    F2 = any_reflection_file("small_cxis_merge_mark0/iobs_all.mtz").as_miller_arrays()[0].as_amplitude_array()
    F2map = {h: v for h, v in zip(F2.indices(), F2.data())}
    if hcommon is None:
        hcommon = set(dataMap).intersection(Fref_map, F2map)
        mtz_vals = array([Fref_map[h] for h in hcommon])
    else:
        assert set(dataMap.keys()).intersection(Fref_map.keys()) == hcommon
    print("HCOMMON=%d" %len(hcommon))
    if args.toggle:
        dataMap = F2map
    vals = array([dataMap[h] for h in hcommon])
    dspace = array([dspace_map[h] for h in hcommon])
    res_bins = [d[0] for d in np.array_split(np.sort(dspace), nbin)] + [dspace.max()]
    res_bin_id = np.digitize(dspace, res_bins)
    corr_at_res = []
    res_bin_centers = [] #(res_bins[1:] + res_bins[:-1])*0.5
    for bin_id in range(1,nbin):
        is_in_bin = res_bin_id==bin_id
        nn = np.sum(is_in_bin)
        if nn<2:
            continue
        vals_in_bin = vals[is_in_bin]
        mtz_vals_in_bin = mtz_vals[is_in_bin]
        corr_at_res.append(CC(vals_in_bin, mtz_vals_in_bin)[0])
        med_d = np.median(dspace[is_in_bin])
        print("res=%.3f, num in bin= %d" % (med_d, nn ))
        res_bin_centers.append(  med_d)
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
plot(res_bin_centers, corr_at_res, 's' , color='tomato')
np.save("data", [res_bin_centers, corr_at_res])
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

