
#include <cub/cub.cuh>
#include "cuda_trilerp.h"
#include "emc.cuh"

__device__ __inline__ unsigned int get_densities_index(int i,int j,int k, int nx, int ny, int nz);

__global__ void trilinear_interpolation_rotate_on_GPU(const CUDAREAL* __restrict__ densities,
                                        VEC3*vectors, CUDAREAL* out, MAT3 rotMat,
                                        int num_qvec,
                                        int nx, int ny, int nz,
                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz);

__global__ void trilinear_insertion_rotate_on_GPU(
        CUDAREAL * densities,
        CUDAREAL * wts,
        CUDAREAL* insertion_values,
        CUDAREAL tomo_wt,
        VEC3 *vectors,
        MAT3 rotMat, int num_qvec,
        int nx, int ny, int nz,
        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz);


__global__ void EMC_equation_two(const CUDAREAL* __restrict__ densities,
                                 const CUDAREAL* __restrict__ data,
                                 CUDAREAL scale_factor,
                                 VEC3*vectors, CUDAREAL* out_rot,
                                 MAT3* rotMats, int * rot_inds,
                                 int numRot, int num_qvec,
                                 int nx, int ny, int nz,
                                 CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                 CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                 const bool compute_derivative);


void prepare_for_lerping(lerpy& gpu, np::ndarray Umats, np::ndarray densities, 
                        np::ndarray qvectors){
    gpu.numRot = Umats.shape(0)/9;
    gpu.numQ = qvectors.shape(0)/3;
    // TODO global verbose flag
    //printf("Number of Qvectors=%d\n", gpu.numQ);
    gpu.numDens = densities.shape(0);
   // TODO asserts on len of corner and delta (must be 3)

    gpuErr(cudaSetDevice(gpu.device));
    gpuErr(cudaMallocManaged((void **)&gpu.rotMats, gpu.numRot*sizeof(MAT3)));
    gpuErr(cudaMallocManaged((void **)&gpu.densities, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.out, gpu.maxNumQ*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.out_equation_two, gpu.maxNumRotInds*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.qVecs, gpu.maxNumQ*sizeof(VEC3)));
    gpuErr(cudaMallocManaged((void **)&gpu.rotInds, gpu.maxNumRotInds*sizeof(int)));
    gpuErr(cudaMallocManaged((void **)&gpu.data, gpu.numDataPixels*sizeof(CUDAREAL)));

    MAT3 Umat; // orientation matrix
    CUDAREAL* Umats_ptr = reinterpret_cast<CUDAREAL*>(Umats.get_data());
    for (int i_rot=0; i_rot < gpu.numRot; i_rot ++){
        int i= i_rot*9;
        CUDAREAL uxx = *(Umats_ptr+i);
        CUDAREAL uxy = *(Umats_ptr+i+1);
        CUDAREAL uxz = *(Umats_ptr+i+2);
        CUDAREAL uyx = *(Umats_ptr+i+3);
        CUDAREAL uyy = *(Umats_ptr+i+4);
        CUDAREAL uyz = *(Umats_ptr+i+5);
        CUDAREAL uzx = *(Umats_ptr+i+6);
        CUDAREAL uzy = *(Umats_ptr+i+7);
        CUDAREAL uzz = *(Umats_ptr+i+8);
        Umat << uxx, uxy, uxz,
                uyx, uyy, uyz,
                uzx, uzy, uzz;
        gpu.rotMats[i_rot] = Umat.transpose();
    }

    CUDAREAL* qvec_ptr = reinterpret_cast<CUDAREAL*>(qvectors.get_data());
    for (int i_q = 0; i_q < gpu.numQ; i_q++) {
        int i = i_q * 3;
        CUDAREAL qx = *(qvec_ptr +i);
        CUDAREAL qy = *(qvec_ptr +i+1);
        CUDAREAL qz = *(qvec_ptr +i+2);
        VEC3 Q(qx, qy, qz);
        gpu.qVecs[i_q] = Q;
    }

    CUDAREAL* dens_ptr = reinterpret_cast<CUDAREAL*>(densities.get_data());
    for (int i=0; i < gpu.numDens; i++){
        gpu.densities[i] = *(dens_ptr+i);
    }
}

void shot_data_to_device(lerpy& gpu, np::ndarray& shot_data){
    unsigned int num_pix = shot_data.shape(0);
    CUDAREAL* data_ptr = reinterpret_cast<CUDAREAL*>(shot_data.get_data());
    for (int i=0; i < num_pix; i++) {
        gpu.data[i] = *(data_ptr + i);
    }
}

void densities_to_device(lerpy& gpu, np::ndarray& new_densities){
    unsigned int numDens = new_densities.shape(0);
    CUDAREAL* dens_ptr = reinterpret_cast<CUDAREAL*>(new_densities.get_data());
    for (int i=0; i < gpu.numDens; i++){
        gpu.densities[i] = *(dens_ptr+i);
    }
}

void toggle_insert_mode(lerpy& gpu){
    if (gpu.wts==NULL){
        gpuErr(cudaMallocManaged((void **)&gpu.wts, gpu.numDens*sizeof(CUDAREAL)));
    }

    for (int i=0; i < gpu.numDens; i++){
        gpu.wts[i]=0;
        gpu.densities[i]=0;
    }
}


void do_a_lerp(lerpy& gpu, std::vector<int>& rot_inds, bool verbose, int task) {
    double time;
    struct timeval t1, t2;//, t3 ,t4;

    gettimeofday(&t1, 0);

    // optional size of each device block, else default to 128
    char *threads = getenv("ORIENT_THREADS_PER_BLOCK");
    if (threads == NULL)
        gpu.blockSize = 128;
    else
        gpu.blockSize = atoi(threads);
    //gpu.blockSize=blockSize;
    gpu.numBlocks = (gpu.numQ + gpu.blockSize - 1) / gpu.blockSize;

    int numRotInds = rot_inds.size();
    for (int i=0; i< numRotInds; i++){
        gpu.rotInds[i] = rot_inds[i];
        if(task==1 || task==3){
            gpu.out_equation_two[i] = 0;
        }
    }
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Pre-kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);

    /*
     * KERNELS
     */
    if (task==0){
        MAT3 rotMat = gpu.rotMats[gpu.rotInds[0]];
        trilinear_interpolation_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.qVecs, gpu.out,
                 rotMat, gpu.numQ,
                 256, 256, 256,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2]
                );
    }
    else if(task==1 || task==3) {
        if (verbose)printf("Running equation 2!\n");
        EMC_equation_two<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities,  gpu.data, gpu.shot_scale,
                 gpu.qVecs, gpu.out_equation_two,
                 gpu.rotMats, gpu.rotInds, numRotInds, gpu.numQ,
                 256, 256, 256,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2], task==3
                );

    }
    else if (task==2)  {
        if (verbose)printf("Trilinear insertion!\n");
        MAT3 rotMat = gpu.rotMats[gpu.rotInds[0]];
//      NOTE: here gpu.data are the insert values
        trilinear_insertion_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.wts, gpu.data, gpu.tomogram_wt, gpu.qVecs,
                 rotMat, gpu.numQ,
                 256, 256, 256,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2]
                );
   
    }
    //else {
    //    // TODO remove this block of code, dont think its used ...
    //    printf("Symmetrize density!\n");
    //    // zero out the densities
    //    toggle_insert_mode(gpu);

    //    MAT3 rotMat = gpu.rotMats[gpu.rotInds[0]];
//  //    NOTE: here gpu.data are the insert values
    //    trilinear_insertion_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
    //            (gpu.densities, gpu.wts, gpu.data, gpu.qVecs,
    //             rotMat, gpu.numQ,
    //             256, 256, 256,
    //             gpu.corner[0], gpu.corner[1], gpu.corner[2],
    //             gpu.delta[0], gpu.delta[1], gpu.delta[2]
    //            );
    //}
    error_msg(cudaGetLastError(), "after kernel call");
    cudaDeviceSynchronize();
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);
    if (task==1 || task==3){
        bp::list outList;
        for (int i = 0; i < numRotInds; i++)
            outList.append(gpu.out_equation_two[i]);
        gpu.outList = outList;
    }
        
    if (verbose){
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Post-kernel time=%f msec\n", time);
    }
}

void free_lerpy(lerpy& gpu){
    if (gpu.qVecs != NULL)
        gpuErr(cudaFree(gpu.qVecs));
    if (gpu.rotInds!= NULL)
        gpuErr(cudaFree(gpu.rotInds));
    if (gpu.rotMats!= NULL)
        gpuErr(cudaFree(gpu.rotMats));
    if (gpu.data!= NULL)
        gpuErr(cudaFree(gpu.data));
    if (gpu.out!= NULL)
        gpuErr(cudaFree(gpu.out));
    if (gpu.out_equation_two!= NULL)
        gpuErr(cudaFree(gpu.out_equation_two));
    if (gpu.densities!= NULL)
        gpuErr(cudaFree(gpu.densities));
    if (gpu.wts!= NULL)
        gpuErr(cudaFree(gpu.wts));
}

__device__ __inline__ unsigned int get_densities_index(int i,int j,int k, int nx, int ny, int nz)
{
    //int idx = i + j*nx + k*nx*ny;
    unsigned int idx = fma(nx, fma(k,ny,j), i);
    return idx;
}

/**
 * this is mostly a CUDA port of the reborn trilinear interpolator written in Fortran:
 *     https://gitlab.com/kirianlab/reborn/-/blob/master/reborn/fortran/density.f90#L16
 */
__global__ void trilinear_interpolation_rotate_on_GPU(
                                        const CUDAREAL * __restrict__ densities, 
                                        VEC3 *vectors, CUDAREAL * out,
                                        MAT3 rotMat, int num_qvec,
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
    int i;
    VEC3 Q;
    
    for (i=tid; i < num_qvec; i += thread_stride){
        Q = rotMat*vectors[i];
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

/**
 * Insert a tomogram into the density
 *
 * this is mostly a CUDA port of the reborn trilinear insertion written in Fortran:
 *     https://gitlab.com/kirianlab/reborn/-/blob/master/reborn/fortran/density.f90#L16
 */
__global__ void trilinear_insertion_rotate_on_GPU(
        CUDAREAL*  densities,
        CUDAREAL*  wts,
        CUDAREAL* insertion_values,
        CUDAREAL tomo_wt,
        VEC3 *vectors,
        MAT3 rotMat, int num_qvec,
        int nx, int ny, int nz,
        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    CUDAREAL i_f, j_f, k_f;
    CUDAREAL x0,x1,y0,y1,z0,z1;
    CUDAREAL qx,qy,qz;
    int i0, i1, j0, j1, k0, k1;
    CUDAREAL a0,a1,a2,a3,a4,a5,a6,a7;
    CUDAREAL x0y0, x1y1, x0y1, x1y0;
    int i;

    VEC3 Q;
    int idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7;
    CUDAREAL val;
    for (i=tid; i < num_qvec; i += thread_stride){
        val = insertion_values[i];
        Q = rotMat*vectors[i];
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

        x0y0 = x0*y0;
        x1y1 = x1*y1;
        x1y0 = x1*y0;
        x0y1 = x0*y1;

        z1 *= tomo_wt;
        z0 *= tomo_wt;

        a0 = x1y1 * z1;
        a1 = x0y1 * z1;
        a2 = x1y0 * z1;
        a3 = x1y1 * z0;
        a4 = x0y1 * z0;
        a5 = x1y0 * z0;
        a6 = x0y0 * z1;
        a7 = x0y0 * z0;
        idx0 = get_densities_index(i0, j0, k0, nx, ny, nz);
        idx1 = get_densities_index(i1, j0, k0, nx, ny, nz);
        idx2 = get_densities_index(i0, j1, k0, nx, ny, nz);
        idx3 = get_densities_index(i0, j0, k1, nx, ny, nz);
        idx4 = get_densities_index(i1, j0, k1, nx, ny, nz);
        idx5 = get_densities_index(i0, j1, k1, nx, ny, nz);
        idx6 = get_densities_index(i1, j1, k0, nx, ny, nz);
        idx7 = get_densities_index(i1, j1, k1, nx, ny, nz);

        atomicAdd(&densities[idx0], val*a0);
        atomicAdd(&densities[idx1], val*a1);
        atomicAdd(&densities[idx2], val*a2);
        atomicAdd(&densities[idx3], val*a3);
        atomicAdd(&densities[idx4], val*a4);
        atomicAdd(&densities[idx5], val*a5);
        atomicAdd(&densities[idx6], val*a6);
        atomicAdd(&densities[idx7], val*a7);

        atomicAdd(&wts[idx0], a0);
        atomicAdd(&wts[idx1], a1);
        atomicAdd(&wts[idx2], a2);
        atomicAdd(&wts[idx3], a3);
        atomicAdd(&wts[idx4], a4);
        atomicAdd(&wts[idx5], a5);
        atomicAdd(&wts[idx6], a6);
        atomicAdd(&wts[idx7], a7);
    }
}

/*
 * Computes equation (2) in http://dx.doi.org/10.1107/S1600576716008165
 *
 */
__global__ void EMC_equation_two(const CUDAREAL * __restrict__ densities,
                                 const CUDAREAL * __restrict__ data,
                                 CUDAREAL scale_factor,
                                 VEC3 *vectors,
                                 CUDAREAL * out_rot,
                                 MAT3* rotMats, int* rot_inds, int numRot, int num_qvec,
                                 int nx, int ny, int nz,
                                 CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                 CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                 const bool compute_derivative){

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_stride = blockDim.x * gridDim.x;
    CUDAREAL i_f, j_f, k_f;
    CUDAREAL x0,x1,y0,y1,z0,z1;
    CUDAREAL qx,qy,qz;
    int i0, i1, j0, j1, k0, k1;
    CUDAREAL I0,I1,I2,I3,I4,I5,I6,I7;
    CUDAREAL a0,a1,a2,a3,a4,a5,a6,a7;
    CUDAREAL x0y0, x1y1, x0y1, x1y0;
    int i_rot;

    VEC3 Q;
    CUDAREAL K_t;
    CUDAREAL W_rt;
    int t,r;

    CUDAREAL R_dr_thread, R_dr_block;
    typedef cub::BlockReduce<CUDAREAL, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    MAT3 R;

    for (i_rot =0; i_rot < numRot; i_rot++){
        R_dr_thread = 0;
        r = rot_inds[i_rot];
        R = rotMats[r];
        for (t=tid; t < num_qvec; t += thread_stride){
            K_t = __ldg(&data[t]);
            Q = R*vectors[t];
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

            W_rt = fma(I0,a0,
                   fma(I1,a1,
                   fma(I2,a2,
                   fma(I3,a3,
                   fma(I4,a4,
                   fma(I5,a5,
                   fma(I6,a6,
                   fma(I7,a7,0))))))));
            if (compute_derivative) {
                if (W_rt > 0)
                    R_dr_thread += K_t/scale_factor - W_rt;
            }
            else{
                //W_rt *= scale_factor;
                if (W_rt > 0)
                    R_dr_thread += K_t * log(W_rt*scale_factor) - W_rt*scale_factor;
            }

        }
        __syncthreads();
        // reduce R_dr across blocks, store result on thread 0
        R_dr_block = BlockReduce(temp_storage).Sum(R_dr_thread);

        // accumulate across all thread 0 using atomics
        if (threadIdx.x==0)
            atomicAdd(&out_rot[i_rot], R_dr_block);
    }
}
