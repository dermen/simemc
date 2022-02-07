
#include "cuda_trilerp.h"

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//__device__ __inline__ int get_densities_index(int i,int j,int k, int stride_x, int stride_y);
__device__ __inline__ int get_densities_index(int i,int j,int k, int nx, int ny, int nz);


//__global__ void trilinear_interpolation(const CUDAREAL* __restrict__ densities,
//                                        VEC3*vectors, CUDAREAL* out, int num_qvec,
//                                        int nx, int ny, int nz,
//                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
//                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz);

__global__ void trilinear_interpolation_rotate_on_GPU(const CUDAREAL* __restrict__ densities,
                                        VEC3*vectors, CUDAREAL* out, MAT3* rotMats, int * rot_inds, 
                                        int numRot, int num_qvec,
                                        int nx, int ny, int nz,
                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz);


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


void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities, 
                        np::ndarray qvectors){
    gpu.numRot = Umats.shape(0)/9;
    gpu.numQ = qvectors.shape(0)/3;
    gpu.numDens = densities.shape(0);
   // TODO asserts on len of corner and delta (must be 3)

    gpuErr(cudaSetDevice(gpu.device));
    gpuErr(cudaMallocManaged((void **)&gpu.rotMats, gpu.numRot*sizeof(MAT3)));
    gpuErr(cudaMallocManaged((void **)&gpu.densities, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.out, gpu.maxNumQ*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.qVecs, gpu.maxNumQ*sizeof(VEC3)));
    gpuErr(cudaMallocManaged((void **)&gpu.rotInds, gpu.maxNumRotInds*sizeof(int)));

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

    for (int i_q = 0; i_q < gpu.numQ; i_q++) {
        int i = i_q * 3;
        CUDAREAL qx = float(bp::extract<double>(qvectors[i]));
        CUDAREAL qy = float(bp::extract<double>(qvectors[i + 1]));
        CUDAREAL qz = float(bp::extract<double>(qvectors[i + 2]));
        VEC3 Q(qx, qy, qz);
        gpu.qVecs[i_q] = Q;
    }

    for (int i=0; i < gpu.numDens; i++){
        gpu.densities[i] = bp::extract<double>(densities[i]);
    }
}

void do_a_lerp(lerpy& gpu, std::vector<int>& rot_inds, bool verbose) {
    double time;
    struct timeval t1, t2;//, t3 ,t4;

    gettimeofday(&t1, 0);

    //int numQ = qvecs.shape(0) / 3;

    // optional size of each device block, else default to 128
    char *threads = getenv("ORIENT_THREADS_PER_BLOCK");
    if (threads == NULL)
        gpu.blockSize = 128;
    else
        gpu.blockSize = atoi(threads);
    gpu.numBlocks = (gpu.numQ + gpu.blockSize - 1) / gpu.blockSize;

    //// copies over qvectors
    //if (verbose)printf("Copying over %d q vectors\n", numQ);
    //for (int i_q = 0; i_q < numQ; i_q++) {
    //    int i = i_q * 3;
    //    CUDAREAL qx = float(bp::extract<double>(qvecs[i]));
    //    CUDAREAL qy = float(bp::extract<double>(qvecs[i + 1]));
    //    CUDAREAL qz = float(bp::extract<double>(qvecs[i + 2]));
    //    VEC3 Q(qx, qy, qz);
    //    gpu.qVecs[i_q] = Q;
    //}
    int numRotInds = rot_inds.size();
    for (int i=0; i< numRotInds; i++){
        gpu.rotInds[i] = rot_inds[i];
    }
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Pre-kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);
    // run the kernel
    //trilinear_interpolation<<<gpu.numBlocks, gpu.blockSize>>>
    //        (gpu.densities, gpu.qVecs, gpu.out, numQ,
    //         256, 256, 256,
    //         gpu.corner[0], gpu.corner[1], gpu.corner[2],
    //         gpu.delta[0], gpu.delta[1], gpu.delta[2]
    //        );
    
    trilinear_interpolation_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
            (gpu.densities, gpu.qVecs, gpu.out, gpu.rotMats, gpu.rotInds, numRotInds, gpu.numQ,
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
//__device__ __inline__ int get_densities_index(int i,int j,int k, int stride_x, int stride_xy)
{
    int idx = i + j*nx + k*nx*ny;
    return idx;
}

//__global__ void trilinear_interpolation(const CUDAREAL * __restrict__ densities, VEC3 *vectors, CUDAREAL * out, 
//                                        int num_qvec,
//                                        int nx, int ny, int nz,
//                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
//                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz){
//
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    int thread_stride = blockDim.x * gridDim.x;
//    CUDAREAL i_f, j_f, k_f;
//    CUDAREAL x0,x1,y0,y1,z0,z1;
//    CUDAREAL qx,qy,qz;
//    int i0, i1, j0, j1, k0, k1;
//    CUDAREAL I0,I1,I2,I3,I4,I5,I6,I7;
//    CUDAREAL a0,a1,a2,a3,a4,a5,a6,a7;
//    CUDAREAL x0y0, x1y1, x0y1, x1y0;
//
//    VEC3 Q;
//    for (int i=tid; i < num_qvec; i += thread_stride){
//        Q = vectors[i];
//        qx = Q[0];
//        qy = Q[1];
//        qz = Q[2];
//
//        k_f = (qx - cx) / dx;
//        j_f = (qy - cy) / dy;
//        i_f = (qz - cz) / dz;
//        i0 = int(floor(i_f));
//        j0 = int(floor(j_f));
//        k0 = int(floor(k_f));
//        if (i0 > nz-2 || j0 > ny-2 || k0 > nx-2 )
//            continue;
//        if(i0 < 0 || j0  < 0 || k0 < 0)
//            continue;
//        i1 = i0 + 1;
//        j1 = j0 + 1;
//        k1 = k0 + 1;
//
//        x0 = i_f - i0;
//        y0 = j_f - j0;
//        z0 = k_f - k0;
//        x1 = 1.0 - x0;
//        y1 = 1.0 - y0;
//        z1 = 1.0 - z0;
//
//        I0 = __ldg(&densities[get_densities_index(i0, j0, k0, nx, ny, nz)]); 
//        I1 = __ldg(&densities[get_densities_index(i1, j0, k0, nx, ny, nz)]); 
//        I2 = __ldg(&densities[get_densities_index(i0, j1, k0, nx, ny, nz)]); 
//        I3 = __ldg(&densities[get_densities_index(i0, j0, k1, nx, ny, nz)]); 
//        I4 = __ldg(&densities[get_densities_index(i1, j0, k1, nx, ny, nz)]); 
//        I5 = __ldg(&densities[get_densities_index(i0, j1, k1, nx, ny, nz)]); 
//        I6 = __ldg(&densities[get_densities_index(i1, j1, k0, nx, ny, nz)]); 
//        I7 = __ldg(&densities[get_densities_index(i1, j1, k1, nx, ny, nz)]); 
//
//        x0y0 = x0*y0;
//        x1y1 = x1*y1;
//        x1y0 = x1*y0;
//        x0y1 = x0*y1;
//       
//        a0 = x1y1 * z1;
//        a1 = x0y1 * z1;
//        a2 = x1y0 * z1;
//        a3 = x1y1 * z0;
//        a4 = x0y1 * z0;
//        a5 = x1y0 * z0;
//        a6 = x0y0 * z1;
//        a7 = x0y0 * z0;
//
//        //out[i] = fma(I0,a0, 
//        //         fma(I1,a1,
//        //         fma(I2,a2,
//        //         fma(I3,a3,
//        //         fma(I4,a4,
//        //         fma(I5,a5,
//        //         fma(I6,a6,
//        //         fma(I7,a7,0))))))));
//
//        out[i] = I0 * a0 +
//                 I1 * a1 +
//                 I2 * a2 +
//                 I3 * a3 +
//                 I4 * a4 +
//                 I5 * a5 +
//                 I6 * a6 +
//                 I7 * a7;
//        //out[i] = __ldg(&densities[get_densities_index(i0, j0, k0, nx, ny, nz)]) * x1 * y1 * z1 +
//        //         __ldg(&densities[get_densities_index(i1, j0, k0, nx, ny, nz)]) * x0 * y1 * z1 +
//        //         __ldg(&densities[get_densities_index(i0, j1, k0, nx, ny, nz)]) * x1 * y0 * z1 +
//        //         __ldg(&densities[get_densities_index(i0, j0, k1, nx, ny, nz)]) * x1 * y1 * z0 +
//        //         __ldg(&densities[get_densities_index(i1, j0, k1, nx, ny, nz)]) * x0 * y1 * z0 +
//        //         __ldg(&densities[get_densities_index(i0, j1, k1, nx, ny, nz)]) * x1 * y0 * z0 +
//        //         __ldg(&densities[get_densities_index(i1, j1, k0, nx, ny, nz)]) * x0 * y0 * z1 +
//        //         __ldg(&densities[get_densities_index(i1, j1, k1, nx, ny, nz)]) * x0 * y0 * z0;
//    }
//}

__global__ void trilinear_interpolation_rotate_on_GPU(
                                        const CUDAREAL * __restrict__ densities, 
                                        VEC3 *vectors, CUDAREAL * out, 
                                        MAT3* rotMats, int* rot_inds, int numRot, int num_qvec,
                                        int nx, int ny, int nz,
                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    CUDAREAL i_f, j_f, k_f;
    CUDAREAL x0,x1,y0,y1,z0,z1;
    CUDAREAL qx,qy,qz;
    int i0, i1, j0, j1, k0, k1;
    CUDAREAL I0,I1,I2,I3,I4,I5,I6,I7;
    CUDAREAL a0,a1,a2,a3,a4,a5,a6,a7;
    CUDAREAL x0y0, x1y1, x0y1, x1y0;
    int rot_index;
    int i, i_rot;

    VEC3 Q;
    
    for (i=tid; i < num_qvec; i += thread_stride){
        for (i_rot =0; i_rot < numRot; i_rot++){
            rot_index = rot_inds[i_rot];
            Q = rotMats[rot_index]*vectors[i];
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

            //I0 = __ldg(&densities[get_densities_index(i0, j0, k0, stride_x, stride_xy)]); 
            //I1 = __ldg(&densities[get_densities_index(i1, j0, k0, stride_x, stride_xy)]); 
            //I2 = __ldg(&densities[get_densities_index(i0, j1, k0, stride_x, stride_xy)]); 
            //I3 = __ldg(&densities[get_densities_index(i0, j0, k1, stride_x, stride_xy)]); 
            //I4 = __ldg(&densities[get_densities_index(i1, j0, k1, stride_x, stride_xy)]); 
            //I5 = __ldg(&densities[get_densities_index(i0, j1, k1, stride_x, stride_xy)]); 
            //I6 = __ldg(&densities[get_densities_index(i1, j1, k0, stride_x, stride_xy)]); 
            //I7 = __ldg(&densities[get_densities_index(i1, j1, k1, stride_x, stride_xy)]); 
            I0 = __ldg(&densities[get_densities_index(i0, j0, k0, nx, ny, nz)]); 
            I1 = __ldg(&densities[get_densities_index(i1, j0, k0, nx, ny, nz)]); 
            I2 = __ldg(&densities[get_densities_index(i0, j1, k0, nx, ny, nz)]); 
            I3 = __ldg(&densities[get_densities_index(i0, j0, k1, nx, ny, nz)]); 
            I4 = __ldg(&densities[get_densities_index(i1, j0, k1, nx, ny, nz)]); 
            I5 = __ldg(&densities[get_densities_index(i0, j1, k1, nx, ny, nz)]); 
            I6 = __ldg(&densities[get_densities_index(i1, j1, k0, nx, ny, nz)]); 
            I7 = __ldg(&densities[get_densities_index(i1, j1, k1, nx, ny, nz)]); 

            x0y0 = x0*y0;
            x1y1 = x1*y1;
            x1y0 = x1*y0;
            x0y1 = x0*y1;
           
            a0 = x1y1 * z1;
            a1 = x0y1 * z1;
            a2 = x1y0 * z1;
            a3 = x1y1 * z0;
            a4 = x0y1 * z0;
            a5 = x1y0 * z0;
            a6 = x0y0 * z1;
            a7 = x0y0 * z0;

            out[i] = I0 * a0 +
                     I1 * a1 +
                     I2 * a2 +
                     I3 * a3 +
                     I4 * a4 +
                     I5 * a5 +
                     I6 * a6 +
                     I7 * a7;
        }
    }
}
