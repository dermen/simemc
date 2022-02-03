from dxtbx.model import ExperimentList
from dials.array_family import flex
import numpy as np
import os

# TODO generalize for variable wavelength
# NOTES: dxtbx radial average (which wraps xfel's radial average , a C++ extension module)
# could also work, but I noticed the beam center is forced to be an integer in that implementation
# Probably doesnt matter much for this work..


class RadPros:

    def __init__(self, refGeom, maskFile, numBins=500):
        """
        :param refGeom: points to an experiment containing a beam and detector
        :param numBins: number of radial bins for the radial profiles
        """

        self.numBins = numBins
        # reference geometry!
        El = ExperimentList.from_file(refGeom)
        self.detector = El[0].detector
        fastDim, slowDim = self.detector[0].get_image_size()
        self.panel_sh = slowDim, fastDim

        mask = np.load(maskFile)
        if len(mask.shape) == 2:
            mask = np.array([mask])
        self.mask = mask

        # not generalized yet for thick detectors
        assert self.detector[0].get_mu() == 0

        self.beam = El[0].beam
        self.unit_s0 = self.beam.get_unit_s0()
        self._setupQbins()

        # geom correction containers (panel_id -> 2D numpy.array)
        self.POLAR = {}
        self.OMEGA = {}
        for pid in range(len(self.detector)):
            self.POLAR[pid] = np.ones(self.panel_sh)
            self.OMEGA[pid] = np.ones(self.panel_sh)

    def _setupQbins(self):
        Qmags = {}
        self.DIFFRACTED = {}
        self.AIRPATH ={}
        for pid in range(len(self.detector)):
            FAST = np.array(self.detector[pid].get_fast_axis())
            SLOW = np.array(self.detector[pid].get_slow_axis())
            ORIG = np.array(self.detector[pid].get_origin())

            Ypos, Xpos = np.indices(self.panel_sh)
            px = self.detector[pid].get_pixel_size()[0]
            Ypos = Ypos* px
            Xpos = Xpos*px

            SX = ORIG[0] + FAST[0]*Xpos + SLOW[0]*Ypos
            SY = ORIG[1] + FAST[1]*Xpos + SLOW[1]*Ypos
            SZ = ORIG[2] + FAST[2]*Xpos + SLOW[2]*Ypos
            self.AIRPATH[pid] = np.sqrt(SX**2 + SY**2 + SZ**2)   # units of mm

            Snorm = np.sqrt(SX**2 + SY**2 + SZ**2)

            SX /= Snorm
            SY /= Snorm
            SZ /= Snorm

            self.DIFFRACTED[pid] = np.array([SX, SY, SZ])

            QX = (SX - self.unit_s0[0]) / self.beam.get_wavelength()
            QY = (SY - self.unit_s0[1]) / self.beam.get_wavelength()
            QZ = (SZ - self.unit_s0[2]) / self.beam.get_wavelength()
            Qmags[pid] = np.sqrt(QX**2 + QY**2 + QZ**2)

            #unit_QX = QX / Qmags_dials
            #unit_QY = QY / Qmags_dials
            #unit_QZ = QZ / Qmags_dials

            #Qvecs_dials = np.zeros( (self.panel_sh)+(3,) )
            #Qvecs_dials[:, :, 0] = unit_QX
            #Qvecs_dials[:, :, 1] = unit_QY
            #Qvecs_dials[:, :, 2] = unit_QZ

        minQ = min([q.min() for q in Qmags.values()])
        maxQ = max([q.max() for q in Qmags.values()])

        self.bins = np.linspace(minQ-1e-6, maxQ+1e-6, self.numBins)

        self.bin_cent = (self.bins[:-1] + self.bins[1:])*.5
        self.all_Qbins = {}
        for pid in Qmags:
            self.all_Qbins[pid] = np.digitize(Qmags[pid], self.bins)-1

    def makeRadPro(self, data_expt, strong_refl):
        """
        :param data_expt: filename of an experiment list containing an imageset
        :param strong_refl: filename of a strong spots reflection table
        :return: radial profile as a numpy array
        """
        data_El = ExperimentList.from_file(data_expt)
        R = flex.reflection_table.from_file(strong_refl)
        iset = data_El[0].imageset
        # load the pixels
        data = iset.get_raw_data(0)
        if not isinstance(data, tuple):
            data = (data,)

        binCounts = np.zeros(self.numBins-1)
        binWeights = np.zeros(self.numBins-1)

        for pid in range(len(self.detector)):
            photons = data[pid].as_numpy_array().astype(np.float32)
            photons /= (self.POLAR[pid]*self.OMEGA[pid])
            pix_mask = self.mask[pid]

            # mask out the strong pixels
            refls_pid = R.select(R['panel'] == pid)
            peak_mask = np.ones_like(pix_mask)
            for x1, x2, y1, y2, _, _ in refls_pid['bbox']:
                peak_mask[y1:y2, x1:x2] = False

            combined_mask = np.logical_and(peak_mask,pix_mask)

            qbin_idx = self.all_Qbins[pid][combined_mask].ravel()
            np.add.at(binCounts, qbin_idx, 1)
            np.add.at(binWeights, qbin_idx, photons[combined_mask].ravel())

        radPro = np.nan_to_num(binWeights / binCounts)
        return radPro

    def solidAngle_correction(self):
        """vectorized solid angle correction for every pixel; follows nanoBragg implementation"""
        sq_pixel_size = self.detector[0].get_pixel_size()[0]**2
        for pid in range(len(self.detector)):
            close_distance = self.detector[pid].get_distance()
            airPathCubed = np.power(self.AIRPATH[pid], 3)
            omega_pixel = sq_pixel_size * close_distance / airPathCubed
            self.OMEGA[pid] = omega_pixel

    def polarization_correction(self):
        """
        vectorized polarization correction for each detector panel
        This is the same expression used in simtbx.nanoBragg based off of the Kahn paper
        Sets the .POLAR attribute, container for the polarization correction as a 2D numpy array, one per panel
        """
        incident = np.array(self.unit_s0)
        pol_axis = np.array(self.beam.get_polarization_normal())
        kahn_factor = self.beam.get_polarization_fraction()
        B_in = np.cross(pol_axis, incident)
        E_in = np.cross(incident, B_in)

        for pid in self.DIFFRACTED:
            d = self.DIFFRACTED[pid]
            sx, sy, sz = map( lambda x: x.ravel(), d)
            diffracted = np.vstack((sx, sy, sz))
            cos2theta = np.dot(incident, diffracted)
            cos2theta_sqr = cos2theta * cos2theta;
            sin2theta_sqr = 1 - cos2theta_sqr;

            psi = 0
            if kahn_factor != 0.0:
                kEi = np.dot(diffracted.T, E_in)
                kBi = np.dot(diffracted.T, B_in)
                psi = -np.arctan2(kBi, kEi);
            polar = 0.5 * (1.0 + cos2theta_sqr - kahn_factor * np.cos(2 * psi) * sin2theta_sqr)
            self.POLAR[pid] = np.reshape(polar, d[0].shape)


if __name__ == "__main__":
    import pylab as plt
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    import time
    import glob
    import h5py


    fnames = glob.glob("all_data[1-8]_spots4/*imported.expt")
    img_nums = [int(f.split("_")[-2]) for f in fnames]
    if COMM.rank==0:
        print("Found %d expts" % len(fnames))


    refGeo = "../make-detector/geom_+x_-y.expt"
    maskFile = "/data/dermen/better_mask_test_data.npy"
    rp = RadPros(refGeo, maskFile)
    rp.polarization_correction()
    rp.solidAngle_correction()
    num_radials = len(fnames)
    with h5py.File("ALL_data_spots4_radials.h5", "w", driver="mpio", comm=COMM) as OUT:
        #OUT.atomic = True
        if COMM.rank==0:
            print("Making dataset")
        radials_dset = OUT.create_dataset("radials", 
                shape=(num_radials, rp.numBins-1), 
                dtype=np.float32)

        tall = time.time()
        fname_idx_per_rank = np.array_split(np.arange(num_radials), COMM.size)[COMM.rank]

        for ii in range(num_radials):
            if ii % COMM.size != COMM.rank: continue

            data_expt = fnames[ii] 
            strong_refl = data_expt.replace("imported.expt", "strong.refl")
            #assert os.path.exists( strong_refl)

            #data_expt = 'test_data_spots4/idx-lysozyme2_test_%06d_imported.expt' % img_num
            #strong_refl = 'test_data_spots4/idx-lysozyme2_test_%06d_strong.refl' % img_num

            t = time.time()
            with np.errstate(divide='ignore', invalid='ignore'):
                radPro = rp.makeRadPro(data_expt, strong_refl)
            radials_dset[ii] = radPro
            t = time.time()-t

            if COMM.rank==0:
                print("Done with img %s (%d / %d) in %.3f sec" % (data_expt, ii+1, num_radials, t))

    COMM.barrier()
    tall = time.time()-tall
    if COMM.rank==0:
        print("Took %.3f sec total" % tall)
