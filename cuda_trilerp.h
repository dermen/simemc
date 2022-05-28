#ifndef CUDA_EMC_H
#define CUDA_EMC_H

#include <Eigen/Dense>
typedef double CUDAREAL;
typedef Eigen::Matrix<CUDAREAL,3,1> VEC3;
typedef Eigen::Matrix<CUDAREAL,3,3> MAT3;
#include <iostream>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <sys/time.h>

namespace bp = boost::python;
namespace np = boost::python::numpy;

struct lerpy {
  MAT3* rotMats=NULL;
  CUDAREAL* densities=NULL;
  CUDAREAL* densities_gradient=NULL;
  CUDAREAL* wts=NULL;
  CUDAREAL* data=NULL;
  CUDAREAL* Pdr=NULL;
  std::vector<CUDAREAL> Pdr_host;
  bool* mask=NULL;
  bool* is_peak_in_density=NULL;
  CUDAREAL* background=NULL;
  int numDataPixels;
  VEC3* qVecs=NULL;
  CUDAREAL* out=NULL;
  CUDAREAL* out_equation_two=NULL;
  bp::list outList;
  int numQ;
  int maxNumQ;
  int numRot;
  int numDens;
  int* rotInds;
  int nx,ny,nz;
  int numBlocks, blockSize;
  int device;
  int maxNumRotInds;
  VEC3 corner;
  VEC3 delta;
  int densDim=258;
  double maxQ=0.25;
  CUDAREAL shot_scale=1;
  CUDAREAL tomogram_wt=1;
  bool use_poisson_stats=true; // if False, a Gaussian random variable is used to describe the pixel measurements
  CUDAREAL sigma_r_sq=0.25;  // variance model for each pixel (e.g. dark noise variance)
};

void relp_mask_to_device(lerpy& gpu, np::ndarray& relp_mask);

void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities,
                         np::ndarray qvectors);

void shot_data_to_device(lerpy& gpu, np::ndarray& shot_data, np::ndarray& shot_mask, np::ndarray& shot_background);
void densities_to_device(lerpy& gpu, np::ndarray& new_densities);

// fills the gpu.out array with interpolated values, 1 for each qvec
void do_a_lerp(lerpy& gpu,
               std::vector<int>& rot_inds,
               bool verbose, int task);

void toggle_insert_mode(lerpy& gpu);

void free_lerpy(lerpy& gpu);



struct gpuOrient {

    MAT3* rotMats=NULL;
    VEC3* qVecs=NULL;
    //double* qVecs=NULL;
    bool* out=NULL;
    int max_numQ;
    int numRot;
    int numBlocks, blockSize;
    int device;
    bp::list probable_rot_inds;
    MAT3 Bmat; // upper diagonal Bmatrix (reciprocal space same convention as dxtbx Crystal.get_B())
};

void orientPeaks(gpuOrient& gpu,
                 np::ndarray qvecs,
                 CUDAREAL hcut,
                 int minpred, bool verbose);
void setup_orientMatch(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray Umats, bool alloc);
void free_orientMatch(gpuOrient& gpu);


#endif
