
#include "cuda_trilerp.h"

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ __inline__ int get_densities_index(int i,int j,int k, int nx, int ny, int nz);


__global__ void trilinear_interpolation(const double* __restrict__ densities,
                                        VEC3*vectors, double* out, int num_qvec,
                                        int nx, int ny, int nz,
                                        double cx, double cy, double cz,
                                        double dx, double dy, double dz);


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void error_msg(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        printf("%s: CUDA error message: %s\n", msg, cudaGetErrorString(err));
        exit(err);
    }
}


void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities, bp::tuple corner, bp::tuple delta){
    gpu.numRot = Umats.shape(0)/9;
    gpu.numDens = densities.shape(0);
   // TODO asserts on len of corner and delta (must be 3)
    gpu.corner[0] = bp::extract<double>(corner[0]);
    gpu.corner[1] = bp::extract<double>(corner[1]);
    gpu.corner[2] = bp::extract<double>(corner[2]);

    gpu.delta[0] = bp::extract<double>(delta[0]);
    gpu.delta[1] = bp::extract<double>(delta[1]);
    gpu.delta[2] = bp::extract<double>(delta[2]);

    gpuErr(cudaSetDevice(gpu.device));
    gpuErr(cudaMallocManaged((void **)&gpu.rotMats, gpu.numRot*sizeof(MAT3)));
    gpuErr(cudaMallocManaged((void **)&gpu.densities, gpu.numDens*sizeof(double)));
    gpuErr(cudaMallocManaged((void **)&gpu.out, gpu.maxNumQ*sizeof(double)));
    gpuErr(cudaMallocManaged((void **)&gpu.qVecs, gpu.maxNumQ*sizeof(VEC3)));

    MAT3 Umat; // orientation matrix
    for (int i_rot=0; i_rot < gpu.numRot; i_rot ++){
        int i= i_rot*9;
        float uxx = float(bp::extract<double>(Umats[i]));
        float uxy = float(bp::extract<double>(Umats[i+1]));
        float uxz = float(bp::extract<double>(Umats[i+2]));
        float uyx = float(bp::extract<double>(Umats[i+3]));
        float uyy = float(bp::extract<double>(Umats[i+4]));
        float uyz = float(bp::extract<double>(Umats[i+5]));
        float uzx = float(bp::extract<double>(Umats[i+6]));
        float uzy = float(bp::extract<double>(Umats[i+7]));
        float uzz = float(bp::extract<double>(Umats[i+8]));
        Umat << uxx, uxy, uxz,
                uyx, uyy, uyz,
                uzx, uzy, uzz;
        gpu.rotMats[i_rot] = Umat;
    }

    for (int i=0; i < gpu.numDens; i++){
        gpu.densities[i] = bp::extract<double>(densities[i]);
    }
}

void do_a_lerp(lerpy& gpu, np::ndarray qvecs, bool verbose) {
    double time;
    struct timeval t1, t2;//, t3 ,t4;

    gettimeofday(&t1, 0);

    int numQ = qvecs.shape(0) / 3;

    // optional size of each device block, else default to 128
    char *threads = getenv("ORIENT_THREADS_PER_BLOCK");
    if (threads == NULL)
        gpu.blockSize = 128;
    else
        gpu.blockSize = atoi(threads);
    gpu.numBlocks = (numQ + gpu.blockSize - 1) / gpu.blockSize;

    // copies over qvectors
    if (verbose)printf("Copying over %d q vectors\n", numQ);
    for (int i_q = 0; i_q < numQ; i_q++) {
        int i = i_q * 3;
        float qx = float(bp::extract<double>(qvecs[i]));
        float qy = float(bp::extract<double>(qvecs[i + 1]));
        float qz = float(bp::extract<double>(qvecs[i + 2]));
        VEC3 Q(qx, qy, qz);
        gpu.qVecs[i_q] = Q;
    }
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Pre-kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);
    // run the kernel
    trilinear_interpolation<<<gpu.numBlocks, gpu.blockSize>>>
            (gpu.densities, gpu.qVecs, gpu.out, numQ,
             256, 256, 256,
             gpu.corner[0], gpu.corner[1], gpu.corner[2],
             gpu.delta[0], gpu.delta[1], gpu.delta[2]
            );

    error_msg(cudaGetLastError(), "after kernel call");
    cudaDeviceSynchronize();
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);
    bp::list outList;
    for (int i = 0; i < gpu.maxNumQ; i++)
        outList.append(gpu.out[i]);
    gpu.outList = outList;
    if (verbose){
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Post-kernel time=%f msec\n", time);
    }
}

__device__ __inline__ int get_densities_index(int i,int j,int k, int nx, int ny, int nz)
{
    int idx = i + j*nx + k*nx*ny;
    return idx;
}

/** 
 * this is a CUDA port of the reborn trilinear interpolator written in Fortran:
 *     https://gitlab.com/kirianlab/reborn/-/blob/master/reborn/fortran/density.f90#L16
 */
__global__ void trilinear_interpolation(const double * __restrict__ densities, VEC3 *vectors, double * out, int num_qvec,
                                        int nx, int ny, int nz,
                                        double cx, double cy, double cz,
                                        double dx, double dy, double dz){
  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    double i_f, j_f, k_f;
    double x0,x1,y0,y1,z0,z1;
    double qx,qy,qz;
    int i0, i1, j0, j1, k0, k1;

    VEC3 Q;
    for (int i=tid; i < num_qvec; i += thread_stride){
        Q = vectors[i];
        qx = Q[0];
        qy = Q[1];
        qz = Q[2];

        k_f = (qx - cx) / dx;
        j_f = (qy - cy) / dy;
        i_f = (qz - cz) / dz;
        i0 = int(floor(i_f));
        j0 = int(floor(j_f));
        k0 = int(floor(k_f));
        if (i0 > nz-2 || j0 > ny-2 || k0 > nx-2 )
            continue;
        if(i0 < 0 || j0  < 0 || k0 < 0)
            continue;
        i1 = i0 + 1;
        j1 = j0 + 1;
        k1 = k0 + 1;

        x0 = i_f - i0;
        y0 = j_f - j0;
        z0 = k_f - k0;
        x1 = 1.0 - x0;
        y1 = 1.0 - y0;
        z1 = 1.0 - z0;

        out[i] = __ldg(&densities[get_densities_index(i0, j0, k0, nx, ny, nz)]) * x1 * y1 * z1 +
                 __ldg(&densities[get_densities_index(i1, j0, k0, nx, ny, nz)]) * x0 * y1 * z1 +
                 __ldg(&densities[get_densities_index(i0, j1, k0, nx, ny, nz)]) * x1 * y0 * z1 +
                 __ldg(&densities[get_densities_index(i0, j0, k1, nx, ny, nz)]) * x1 * y1 * z0 +
                 __ldg(&densities[get_densities_index(i1, j0, k1, nx, ny, nz)]) * x0 * y1 * z0 +
                 __ldg(&densities[get_densities_index(i0, j1, k1, nx, ny, nz)]) * x1 * y0 * z0 +
                 __ldg(&densities[get_densities_index(i1, j1, k0, nx, ny, nz)]) * x0 * y0 * z1 +
                 __ldg(&densities[get_densities_index(i1, j1, k1, nx, ny, nz)]) * x0 * y0 * z0;
    }
}
