Timer unit: 1e-06 s

Total time: 10.5021 s
File: /mnt/home1/dermen/xtal_gpu3/modules/simemc/compute_radials.py
Function: makeRadPro at line 127

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   127                                               @profile
   128                                               def makeRadPro(self, data_pixels=None, data_expt=None, strong_refl=None, strong_params=None,
   129                                                           apply_corrections=True, use_median=True):
   130                                                   """
   131                                                   Create a 1d radial profile of the background pixels in the image
   132                                                   :param data_pixels: image pixels same shape as detector model 
   133                                                   :param data_expt: filename of an experiment list containing an imageset
   134                                                   :param strong_refl: filename of a strong spots reflection table
   135                                                   :param strong_params:  phil params for dials.spotfinder 
   136                                                   :param apply_corrections: if True, correct for polarization and solid angle
   137                                                   :param use_median: compute radial median profile, as opposed to radial mean profile
   138                                                   :return: radial profile as a numpy array
   139                                                   """
   140         5          5.0      1.0      0.0          if data_expt is not None:
   141                                                       data_El = ExperimentList.from_file(data_expt)
   142                                                       iset = data_El[0].imageset
   143                                                       # load the pixels
   144                                                       data = iset.get_raw_data(0)
   145                                                       if not isinstance(data, tuple):
   146                                                           data = (data,)
   147                                                       data = np.array([d.as_numpy_array() for d in data])
   148                                                   else:
   149         5          5.0      1.0      0.0              assert data_pixels is not None
   150         5          4.0      0.8      0.0              data = data_pixels
   151                                           
   152         5          3.0      0.6      0.0          if strong_refl is None:
   153                                                       assert strong_params is not None
   154                                                       all_peak_masks = [~dials_find_spots(data[pid], self.mask[pid], strong_params)\
   155                                                                           for pid in range(len(data))]
   156                                                   else:
   157         5       6304.0   1260.8      0.1              all_peak_masks = ~strong_spot_mask(strong_refl, self.detector)
   158                                           
   159         5         10.0      2.0      0.0          if apply_corrections:
   160                                                       data /= (self.POLAR*self.OMEGA)
   161         5     212159.0  42431.8      2.0          bin_labels = self.all_Qbins.copy()
   162         5       5723.0   1144.6      0.1          combined_mask = np.logical_and(all_peak_masks, self.mask)
   163         5       8802.0   1760.4      0.1          bin_labels[~combined_mask] = 0
   164         5        229.0     45.8      0.0          with np.errstate(divide='ignore', invalid='ignore'):
   165         5          4.0      0.8      0.0              if use_median:
   166         5   10268824.0 2053764.8     97.8                  radPro = ndimage.median(data, bin_labels,self._index)
   167                                                       else:
   168                                                           radPro = ndimage.mean(data, bin_labels,self._index)
   169         5         14.0      2.8      0.0          return radPro

Total time: 20.1134 s
File: bg_and_probOri.py
Function: main at line 30

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    30                                           @profile
    31                                           def main():
    32         1          5.0      5.0      0.0      hcut = 0.12
    33         1          2.0      2.0      0.0      min_pred = 4
    34         1         10.0     10.0      0.0      outdir = args.outdir
    35         1          2.0      2.0      0.0      quat_file = args.quat
    36         1          1.0      1.0      0.0      input_file=args.input
    37         1          7.0      7.0      0.0      num_gpu_dev = args.ndev
    38         1          1.0      1.0      0.0      max_num_strong_spots = 1000
    39         1          2.0      2.0      0.0      num_process = args.num
    40         1          2.0      2.0      0.0      qmin = 1/40.
    41         1          1.0      1.0      0.0      qmax = 1/4.
    42                                           
    43                                               # constants
    44         1          2.0      2.0      0.0      img_sh = 2527, 2463
    45         1          1.0      1.0      0.0      numQ = 256
    46         1          2.0      2.0      0.0      num_radial_bins = 500
    47         1         36.0     36.0      0.0      gpu_device = COMM.rank % num_gpu_dev
    48                                           
    49         1         48.0     48.0      0.0      mpi_utils.make_dir(outdir)
    50         1         28.0     28.0      0.0      outfile = os.path.join(outdir, "emc_input%d.h5" %COMM.rank)
    51                                           
    52         1    2180312.0 2180312.0     10.8      rotMats, rotMatWeights = utils.load_quat_file(quat_file)
    53         1     178579.0 178579.0      0.9      expt_names, refl_names = utils.load_expt_refl_file(input_file)
    54         1          4.0      4.0      0.0      if num_process is not None:
    55         1         31.0     31.0      0.0          expt_names = expt_names[:num_process]
    56         1         68.0     68.0      0.0          refl_names = refl_names[:num_process]
    57                                           
    58         1         36.0     36.0      0.0      shot_numbers = np.arange(len(expt_names))
    59         1        162.0    162.0      0.0      shot_num_rank = np.array_split(np.arange(len(expt_names)), COMM.size)[COMM.rank]
    60                                           
    61         1          2.0      2.0      0.0      Qx = Qy = Qz = None
    62         1          1.0      1.0      0.0      detector = beam = None
    63         1          2.0      2.0      0.0      if COMM.rank==0:
    64         1          5.0      5.0      0.0          if args.geom is not None:
    65         1       7565.0   7565.0      0.0              dummie_expt = ExperimentList.from_file(args.geom, False)[0]
    66                                                   else:
    67                                                       dummie_expt = ExperimentList.from_file(expt_names[0], False)[0]
    68         1          7.0      7.0      0.0          detector = dummie_expt.detector
    69         1       4213.0   4213.0      0.0          detector = strip_thickness_from_detector(detector)
    70         1          5.0      5.0      0.0          beam = dummie_expt.beam
    71         1     800462.0 800462.0      4.0          qmap = utils.calc_qmap(detector, beam)
    72         1         61.0     61.0      0.0          Qx,Qy,Qz = map(lambda x: x.ravel(), qmap)
    73         1        773.0    773.0      0.0      detector = COMM.bcast(detector)
    74         1        189.0    189.0      0.0      beam = COMM.bcast(beam)
    75                                           
    76         1     126322.0 126322.0      0.6      Qx = COMM.bcast(Qx)
    77         1     124391.0 124391.0      0.6      Qy = COMM.bcast(Qy)
    78         1     125335.0 125335.0      0.6      Qz = COMM.bcast(Qz)
    79         1     154570.0 154570.0      0.8      Qmag = np.sqrt( Qx**2 + Qy**2 + Qz**2)
    80                                           
    81         1        363.0    363.0      0.0      qbins = np.linspace( -qmax, qmax, numQ + 1)
    82         1      21181.0  21181.0      0.1      sel = np.logical_and(Qmag > qmin, Qmag < qmax)
    83         1      30001.0  30001.0      0.1      qXYZ = Qx[sel], Qy[sel], Qz[sel]
    84                                           
    85         1        113.0    113.0      0.0      print0("Found %d experiment files total, dividing across ranks" % len(expt_names), flush=True)
    86                                               # TODO: this script assumes a single panel image format, generalizing is trivial, but should be done
    87                                           
    88                                               # make the probable orientation identifier
    89         1         22.0     22.0      0.0      O = probable_orients()
    90         1     954715.0 954715.0      4.7      O.allocate_orientations(gpu_device, rotMats.ravel(), max_num_strong_spots)
    91                                           
    92         1          3.0      3.0      0.0      radProMaker = None
    93         1          1.0      1.0      0.0      correction = None
    94                                           
    95                                               # NOTE: assume one knows the unit cell:
    96         1         70.0     70.0      0.0      O.Bmatrix = CRYSTAL.get_B()
    97                                           
    98                                           
    99         1      30835.0  30835.0      0.2      with h5py.File(outfile, "w") as OUT:
   100         1          6.0      6.0      0.0          num_shots = len(shot_num_rank)  # number of shots to load on this rank
   101         2        619.0    309.5      0.0          prob_rot_dset = OUT.create_dataset(
   102         1          2.0      2.0      0.0              name="probable_rot_inds", shape=(num_shots,),
   103         1         23.0     23.0      0.0              dtype=h5py.vlen_dtype(int))
   104         2        192.0     96.0      0.0          bg_dset = OUT.create_dataset(
   105         1          2.0      2.0      0.0              name="background", shape=(num_shots,num_radial_bins))
   106         6         56.0      9.3      0.0          for i_f, i_shot in enumerate(shot_num_rank):
   107         5         15.0      3.0      0.0              expt_f = expt_names[i_shot]
   108         5         10.0      2.0      0.0              refl_f = refl_names[i_shot]
   109                                           
   110         5      89837.0  17967.4      0.4              El = ExperimentList.from_file(expt_f, True)
   111         5     560318.0 112063.6      2.8              data = image_data_from_expt(El[0])
   112                                           
   113         5       4251.0    850.2      0.0              R = flex.reflection_table.from_file(refl_f)
   114         5       3057.0    611.4      0.0              R.centroid_px_to_mm(El)
   115         5       1216.0    243.2      0.0              R.map_centroids_to_reciprocal_space(El)
   116                                           
   117                                                       ##########################
   118                                                       # Get the background image
   119                                                       ##########################
   120         5         10.0      2.0      0.0              if radProMaker is None:
   121         1         64.0     64.0      0.0                  print0("Creating radial profile maker!", flush=True)
   122                                                           # TODO: add support for per-shot wavelength
   123         1          2.0      2.0      0.0                  refGeom = {"D": detector, "B": beam}
   124         1    1715432.0 1715432.0      8.5                  radProMaker = RadPros(refGeom, numBins=num_radial_bins)
   125         1    1065050.0 1065050.0      5.3                  radProMaker.polarization_correction()
   126         1     431645.0 431645.0      2.1                  radProMaker.solidAngle_correction()
   127         1      32808.0  32808.0      0.2                  correction = radProMaker.POLAR * radProMaker.OMEGA
   128         1       8079.0   8079.0      0.0                  correction /= np.mean(correction)
   129                                           
   130         5         21.0      4.2      0.0              t = time.time()
   131         5      34112.0   6822.4      0.2              data *= correction
   132        10   10525861.0 1052586.1     52.3              radialProfile = radProMaker.makeRadPro(
   133         5          9.0      1.8      0.0                      data_pixels=data,
   134         5          7.0      1.4      0.0                      strong_refl=R,
   135         5          9.0      1.8      0.0                      apply_corrections=False, use_median=True)
   136         5         36.0      7.2      0.0              tbg = time.time()-t
   137                                           
   138                                                       ####################################
   139                                                       # Get the probable orientations list
   140                                                       ####################################
   141         5          9.0      1.8      0.0              t = time.time()
   142         5        396.0     79.2      0.0              qvecs = R['rlp'].as_numpy_array()
   143         5          9.0      1.8      0.0              verbose_flag = False #COMM.rank==0
   144         5     584617.0 116923.4      2.9              prob_rot = O.orient_peaks(qvecs.ravel(), hcut, min_pred, verbose_flag)
   145         5         36.0      7.2      0.0              tori = time.time()-t
   146                                           
   147                                                       ### Save stuff
   148         5       3225.0    645.0      0.0              prob_rot_dset[i_f] = prob_rot
   149         5       1636.0    327.2      0.0              bg_dset[i_f] = radialProfile
   150        15        330.0     22.0      0.0              print0("(%d/%d) bkgrnd est. took %.4f sec, prob. ori. est. %.4f sec . %d prob ori from %d strong spots."
   151        10         23.0      2.3      0.0                     % (i_f+1, num_shots, tbg, tori, len(prob_rot), len(qvecs)), flush=True)
   152                                           
   153         1        421.0    421.0      0.0          OUT.create_dataset("background_img_sh", data=radProMaker.img_sh)
   154         1      31493.0  31493.0      0.2          OUT.create_dataset("all_Qbins", data=radProMaker.all_Qbins)
   155         1      26148.0  26148.0      0.1          OUT.create_dataset("polar", data=radProMaker.POLAR)
   156         1      28553.0  28553.0      0.1          OUT.create_dataset("omega", data=radProMaker.OMEGA)
   157         1      37263.0  37263.0      0.2          OUT.create_dataset("correction", data=correction)
   158         1         16.0     16.0      0.0          Es = [expt_names[i] for i in shot_num_rank]
   159         1          5.0      5.0      0.0          Rs = [refl_names[i] for i in shot_num_rank]
   160         3     184302.0  61434.0      0.9          for dset_name, lst in [("expts", Es), ("refls", Rs)]:
   161         2        625.0    312.5      0.0              dset = OUT.create_dataset(dset_name, shape=(num_shots,), dtype=h5py.string_dtype(encoding="utf-8"))
   162         2        961.0    480.5      0.0              dset[:] = lst

