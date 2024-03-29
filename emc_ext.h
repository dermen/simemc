#ifndef CUDA_EMC_H
#define CUDA_EMC_H

#include <Eigen/Dense>
#include<Eigen/StdVector>
#include <iostream>
#include <mpi.h>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <sys/time.h>

typedef double CUDAREAL; // FIXME CUDAREAL as float causes failures
typedef Eigen::Matrix<CUDAREAL,3,1> VEC3;
typedef Eigen::Matrix<CUDAREAL,3,3> MAT3;
typedef std::vector<VEC3,Eigen::aligned_allocator<VEC3> > eigVec3_vec;

#define MPI_CUDAREAL sizeof(CUDAREAL)==sizeof(double) ? MPI_DOUBLE: MPI_FLOAT

namespace bp = boost::python;
namespace np = boost::python::numpy;

struct lerpy {
  int* sparse_lookup=NULL;
  int num_sparse_lookup=-1;
  int* unmasked_inds=NULL;
  CUDAREAL* unmasked_vals=NULL;
  int num_unmasked=-1;
  bool alwaysResetDeriv = true;
  bool close_rotMats_handle = false;
  bool free_rotMats = true;
  int mpi_rank=-1; // MPI rank, has to be set manually from python
  int num_sym_op = 0; // number of symmetry operators
  MAT3* rotMatsSym=NULL; // symmetry operators
  MAT3* rotMats=NULL; // orientation grid operators
  CUDAREAL* densities=NULL; // values in each voxel (shape is total number of voxels)
  CUDAREAL* densities_gradient=NULL;
  CUDAREAL* wts=NULL; // trilinear insertion weights
  CUDAREAL* data=NULL; // typically these will be detector pixel vaues (for inserting into the density)
  CUDAREAL* Pdr=NULL;  // probability of a rotation "r" for a given pattern "d"
  std::vector<CUDAREAL> Pdr_host;
  bool* mask=NULL; // same length as data array, these are the trusted values (e.g. from a hot pixel mask)
  bool* is_peak_in_density=NULL;
  CUDAREAL* background=NULL; // same length as the background array, these are the background scattering values
  int numDataPixels; // size of the detector array (typically, though the symmetrize_density method hijacks this attribute
  VEC3* qVecs=NULL; // qvecs, one for every detector pixel (typically, though the symmetrize_density method hijacks this attribute
  CUDAREAL* out=NULL; // output buffer  (see do_a_lerp function body)
  CUDAREAL* out_equation_two=NULL; // output buffer  (see do_a_lerp function body)
  bp::list outList; // output  buffer  (see do_a_lerp function body)
  int numQ; // number of q copied to the allocated qvector array
  int maxNumQ;  //size of the allocated qvector array
  int numRot; // number of rotation matrices stored on device (these correspond to the EMC grid)
  int numDens=-1; // number of voxels in density
  int* rotInds; // indices corresponding to the rotation matrix grid stored on the gpu
  int nx,ny,nz; //
  int numBlocks, blockSize; // cuda-specific variables
  int device;  // id of the gpu to use (integer)
  int maxNumRotInds; // size of the allocated array for storing rotation matrices (max number of rot mats on gpu)
  VEC3 corner; // lowest corner of the density
  VEC3 delta; // size of a voxel
  // TODO remove these default values, they should be set during class construction
  int densDim=256; // number of voxels along one dimension in the density
  double maxQ=0.25; // maximum q in inverse angstroms
  CUDAREAL shot_scale=1; // scale factor for the current shot corresponding to the info stored in data/mask/background
  CUDAREAL tomogram_wt=1;  // specify a weight for each tomogram (experimental)
  bool use_poisson_stats=true; // if False, a Gaussian random variable is used to describe the pixel measurements
  CUDAREAL sigma_r_sq=0.25;  // variance model for each pixel (e.g. dark noise variance)
  bool is_allocated=false; // whether device arrays have been allocated
};

void set_sparse_lookup(lerpy& gpu, np::ndarray& is_peak_in_density);
void set_device(lerpy& gpu);
void reparameterize_density_gradients(lerpy& gpu);
void convert_reparameterized_densities(lerpy& gpu);
void malloc_relp_mask(lerpy& gpu);
void relp_mask_to_device(lerpy& gpu, np::ndarray& relp_mask);
void malloc_unmasked_inds(lerpy& gpu);

void prepare_for_lerping(lerpy& gpu, np::ndarray& Umats,
                         np::ndarray& qvectors, bool use_IPC);

void copy_image_data_to_device(lerpy& gpu, np::ndarray& shot_data, np::ndarray& shot_mask, np::ndarray& shot_background);
void densities_to_device(lerpy& gpu, np::ndarray& new_densities);

void set_threads_blocks(lerpy& gpu, int N);  // sets the blocksize
void do_after_kernel();
void sym_ops_to_dev(lerpy& gpu, np::ndarray& rot_mats);
void symmetrize_density(lerpy& gpu, np::ndarray& _q_cent);


// fills the gpu.out array with interpolated values, 1 for each qvec
void do_a_lerp(lerpy& gpu,
               std::vector<int>& rot_inds,
               bool verbose, int task);
void to_dev_memcpy( CUDAREAL* dev_ptr,  CUDAREAL* host_ptr, int N);
void from_dev_memcpy( CUDAREAL* dev_ptr,  CUDAREAL* host_ptr, int N);
void from_dev_memcpy_int( int* dev_ptr,  int* host_ptr, int N);
void reset_dens_deriv(lerpy& gpu);
void toggle_insert_mode(lerpy& gpu);
void free_lerpy(lerpy& gpu);
void update_masked_density_gpu(lerpy& gpu, np::ndarray& new_vals);


struct gpuOrient {
//  flags for the cleanup method to know whether to close te handle or free mem (mem only allocated on one rank)
    bool close_rotMats_handle = false;
    bool free_rotMats = true;
    int mpi_rank=-1; // TODO add a setter for this in emc_ext
    MAT3* rotMats=NULL;
    VEC3* qVecs=NULL;
    //double* qVecs=NULL;
    bool* out=NULL;
    int max_numQ;
    int numRot;
    int numBlocks, blockSize;
    int device=-1;
    bp::list probable_rot_inds;
    MAT3 Bmat; // upper diagonal Bmatrix (reciprocal space same convention as dxtbx Crystal.get_B())
};

void orientPeaks(gpuOrient& gpu,
                 np::ndarray& qvecs,
                 CUDAREAL hcut,
                 int minpred, bool verbose);
void setup_orientMatch(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray& Umats, bool alloc);
void setup_orientMatch_IPC(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray& Umats, int numRot, MPI_Comm COMM);
void free_orientMatch(gpuOrient& gpu);

/*
The following methods are defined in general.cu
*/
size_t get_gpu_mem();
void copy_umats(MAT3*& mats, np::ndarray& Umats, int numRot);


#endif
