
#include <Eigen/Dense>
typedef Eigen::Matrix<float,3,1> VEC3;
typedef Eigen::Matrix<float,3,3> MAT3;
#include <iostream>
#include <boost/python/numpy.hpp>
#include <stdio.h>
#include <sys/time.h>


namespace bp = boost::python;
namespace np = boost::python::numpy;

struct lerpy {
  MAT3* rotMats=NULL;
  double* densities;
  VEC3* qVecs=NULL;
  double* out=NULL;
  bp::list outList;
  int numQ;
  int maxNumQ;
  int numRot;
  int numDens;
  int nx,ny,nz;
  int numBlocks, blockSize;
  int device;
  VEC3 corner;
  VEC3 delta;
};

void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities,
                         bp::tuple corner, bp::tuple delta);

// fills the gpu.out array with interpolated values, 1 for each qvec
void do_a_lerp(lerpy& gpu,
               np::ndarray qvecs,
               bool verbose);
