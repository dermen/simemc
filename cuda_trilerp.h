#ifndef CUDA_EMC_H
#define CUDA_EMC_H

#include <Eigen/Dense>
typedef float CUDAREAL;
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
  CUDAREAL* densities;
  CUDAREAL* data;
  int numDataPixels;
  VEC3* qVecs=NULL;
  CUDAREAL* out=NULL;
  CUDAREAL* out2=NULL;
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
};

void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities,
                         np::ndarray qvectors);

void shot_data_to_device(lerpy& gpu, np::ndarray& shot_data);

// fills the gpu.out array with interpolated values, 1 for each qvec
void do_a_lerp(lerpy& gpu,
               std::vector<int>& rot_inds,
               //np::ndarray qvecs,
               bool verbose, int task);


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
};

void orientPeaks(gpuOrient& gpu,
                 np::ndarray qvecs,
                 CUDAREAL hcut,
                 int minpred, bool verbose);
void setup_orientMatch(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray Umats, bool alloc);
void free_orientMatch(gpuOrient& gpu);

#endif