# coding: utf-8
from argparse import ArgumentParser
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument("input", type=str , help="simemc output folder" )
parser.add_argument("iterNum", type=int, help="simemc output file iter number")
parser.add_argument("output", type=str, help="will write a file output.pkl and dump expts and refls into a folder output/")
parser.add_argument("quatN", type=int, help="int specifying simemc quaternion file used")
parser.add_argument("--j", type=int, default=4, help="number of processes")
args = parser.parse_args()
import glob
    
from dials.array_family import flex
from dxtbx.model import ExperimentList
import numpy as np
import os
from simemc import utils
from dxtbx.model.crystal import CrystalFactory
from cctbx import crystal, sgtbx
from copy import deepcopy
import pandas

from simemc import integrate, refine, index




def get_med_offset(R):
    x,y,_ = R['xyzobs.px.value'].parts()
    x2,y2,_ = R['xyzcal.px'].parts()
    return np.median(np.sqrt((x-x2)**2 + (y-y2)**2))

OUT = args.output
if OUT.endswith("/"):
    OUT = OUT[:-1]

if not os.path.exists(OUT):
    os.makedirs(OUT)
else:
    assert os.path.isdir(OUT)

fnames = glob.glob('%s/prob_rots/rank*/*iter%d*npz' % (args.input, args.iterNum))
print(len(fnames))

def main(RANK, SIZE):

    PHIL = os.environ["DD4"] + "/proc_small_cxis_ucell.phil"
    MASK = os.environ["DD4"] + "/test_mask.pkl"
    A,B = utils.load_quat_file(os.environ["MODZ"] + "/simemc/quatgrid/c-quaternion%d.bin" % args.quatN)

    UCELL_A = 68.5
    UCELL_B = 68.5
    UCELL_C = 104.8
    HALL='-P 4 2'
    crystal_dict = {
        '__id__': 'crystal',
        'real_space_a': (UCELL_A, 0.0, 0.0),
        'real_space_b': (0.0, UCELL_B, 0.0),
        'real_space_c': (0.0, 0.0, UCELL_C),
        'space_group_hall_symbol': HALL}
    Cbase = CrystalFactory.from_dict(crystal_dict)
        
    mats = [np.reshape(o.r().as_double(),(3,3)) for o in sgtbx.space_group_info("P43212").group().all_ops()]

    seen_xtal = 0
    residuals = []
    for i_f,f in enumerate(fnames):
      if i_f % SIZE != RANK: 
        continue
      data = dict(np.load(f, allow_pickle=True))  
      nshot = len(data['names'])
      for i_shot in range(nshot):
        prob = data["Pdr"][i_shot]
        rot_inds = data['rots'][i_shot]
        name = data['names'][i_shot]
        ngood = sum(prob > 1e-2)
        if ngood not  in [1,4]:
            continue
        if ngood==1:
            assert np.allclose(prob[prob>0],1)
            good_rots = [A[i] for i in rot_inds[prob>0]]
            
        if ngood==4:
            if not np.allclose( prob[prob>0], .25):
                continue
            v = np.ones(3)/np.sqrt(3)
            good_rots = [A[i] for i in rot_inds[prob > 0]]
            u = np.matmul(good_rots[0].T,v)
            for i in range(1,4):
                equivs = 0
                for M in mats:
                    Mu = np.matmul(M,good_rots[i].T)
                    u2 = np.matmul(Mu, v)
                    
                    if np.allclose( u, u2):
                        equivs+=1
                assert equivs==1
                # if probabilities are equal, action of rotations should be equal
        Umat = good_rots[0]#.T
        El = ExperimentList.from_file(name,True)
        C = deepcopy(Cbase)
        
        assert len(El)==1
        C.set_U(tuple(Umat.ravel()))
        El[0].crystal = C
        assert np.allclose(El[0].crystal.get_U(), Umat.ravel())
        if "strong" in name:
            refl_name = name.replace(".expt", ".refl")
        else: #"refined" in name:
            refl_name = name.replace("refined.expt", "strong.refl")
        assert os.path.exists(refl_name)
        R = flex.reflection_table.from_file(refl_name)
        El[0].scan = None
        El[0].goniometer = None
        try:
            El_idx, R_idx = index.index(PHIL, El, R)
            El_ref, R_ref = refine.refine(PHIL, El_idx, R_idx)
            El_int, R_int = integrate.integrate(
                            PHIL, El_idx, R_idx,MASK,False)

        except Exception as err:
            print("Failed to index/integrate!", str(err))
            continue

        basename = os.path.splitext(os.path.basename(name))[0]
        new_name_template = "%s/%s"% (OUT,basename) + "_%s.expt" 

        med = get_med_offset(R_idx)
        residuals.append( (med, len(R_idx) ))
        for el,r,tag in zip(
                [El_idx, El_ref, El_int],
                [R_idx, R_ref, R_int],
                ["indexed", "refined", "integrated"]):
            nn = new_name_template % tag
            el.as_file(nn)
            r.as_file(nn.replace(".expt", ".refl"))

        seen_xtal += 1
        if RANK==0:
            print("Done with rank %d / %d (shot %d / %d), saved %d integrated refls" 
                % (i_f+1, len(fnames), i_shot+1, nshot, len(R_int)), flush=True)
        
    return residuals 


results = Parallel(n_jobs=args.j)(delayed(main)(jid, args.j) for jid in range(args.j))
residuals = []
for r in resultrs:
    residuals += r

meds, nrefs = zip(*residuals)
df = pandas.DataFrame({"pred_offset":meds, 
                    "nref": nrefs})
df.to_pickle("%s.pkl" % OUT)
from pylab import *
hexbin(df.pred_offset, df.nref, cmap='gnuplot', xscale='log', yscale='log')
xlabel("pred offset (pixels)", fontsize=14)
ylabel("# of refls", fontsize=14)
title("Median[pref offset]=%.3f pix; Median[# ref]=%.1f" % (df.pred_offset.median(), df.nref.median()), fontsize=14)
show()

