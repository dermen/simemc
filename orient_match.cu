
#include <mpi.h>
#include "emc_ext.h"
#include "general.cuh"

__global__
void orientMultiply(VEC3*  qVecs, MAT3* rotMats, int numQ,
                    int numRot, bool* out, CUDAREAL hcut,
                    int minPred, MAT3 Bmat);

void set_blocks_threads(gpuOrient& gpu, int numRot);

void free_orientMatch(gpuOrient& gpu){
    if (gpu.rotMats != NULL && gpu.free_rotMats)
        gpuErr(cudaFree(gpu.rotMats));
    if(gpu.qVecs != NULL)
        gpuErr(cudaFree(gpu.qVecs));
    if (gpu.out != NULL)
        gpuErr(cudaFree(gpu.out));
    if (gpu.close_rotMats_handle)
        gpuErr(cudaIpcCloseMemHandle(gpu.rotMats));
}


void set_blocks_threads(gpuOrient& gpu, int numRot){
    // optional size of each device block, else default to 128
    char* diffBragg_threads = getenv("ORIENT_THREADS_PER_BLOCK");
    if (diffBragg_threads==NULL)
        gpu.blockSize=128;
    else
        gpu.blockSize=atoi(diffBragg_threads);
    gpu.numBlocks = (numRot+gpu.blockSize-1)/gpu.blockSize;
}

void setup_orientMatch_IPC(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray& Umats, int numRot, MPI_Comm COMM){

    int rank, size;
    MPI_Comm_size(COMM, &size);
    MPI_Comm_rank(COMM, &rank);

    gpuErr(cudaSetDevice(dev_id));
    gpu.numRot = numRot;
    gpu.max_numQ = maxNumQ;
    gpu.device = dev_id;
    gpuErr(cudaMallocManaged((void **)&gpu.out, numRot*sizeof(bool)));
    gpuErr(cudaMallocManaged((void **)&gpu.qVecs, maxNumQ*sizeof(VEC3)));

    cudaIpcMemHandle_t rotMats_memHand;//[1];
    if (rank==0){
        get_mem_handle(rotMats_memHand, gpu.rotMats, Umats, numRot);
    }
    // broadcast and copy the memoryhandle to gpu.rotMats on other processes
    broadcast_ipc_handle(rotMats_memHand, gpu.rotMats, COMM);

    gpu.free_rotMats = (rank==0) ;
    gpu.close_rotMats_handle = (rank >0) ;

    set_blocks_threads(gpu, numRot);
}


void setup_orientMatch(int dev_id, int maxNumQ, gpuOrient& gpu,
                       np::ndarray& Umats, bool alloc ){
    int numRot = Umats.shape(0)/9;
    if (alloc){
        gpu.numRot = numRot;
        gpu.max_numQ = maxNumQ;
        gpuErr(cudaSetDevice(dev_id));
        gpu.device = dev_id;
        gpuErr(cudaMallocManaged((void **)&gpu.rotMats, numRot*sizeof(MAT3)));
        gpuErr(cudaMallocManaged((void **)&gpu.out, numRot*sizeof(bool)));
        gpuErr(cudaMallocManaged((void **)&gpu.qVecs, maxNumQ*sizeof(VEC3)));
    }
    copy_umats(gpu.rotMats, Umats, numRot);
    set_blocks_threads(gpu, numRot);
}


void orientPeaks(gpuOrient& gpu, np::ndarray& qvecs, CUDAREAL hcut,
                 int minPred, bool verbose){

    double time;
    struct timeval t1, t2;//, t3 ,t4;

    gettimeofday(&t1, 0);
    int numQ = qvecs.shape(0)/3;

    if (verbose)
        printf("Setting cuda device %d\n", gpu.device);
    gpuErr(cudaSetDevice(gpu.device));
    if (numQ > gpu.max_numQ){
        printf("WARNING: re-allocating because maximum num Q vecs was exceeded!! Now maxNumQ =%d (was %d)\n",
               numQ, gpu.max_numQ);
        gpu.max_numQ = numQ;
        if (gpu.qVecs != NULL)
        gpuErr(cudaFree(gpu.qVecs));
        gpuErr(cudaMallocManaged((void **)&gpu.qVecs, gpu.max_numQ*sizeof(VEC3)));
    }

    // copy the Qvectors to the device
    if (verbose)
        printf("Copying over %d qvectors to the GPU\n", numQ);
    CUDAREAL* qptr = reinterpret_cast<CUDAREAL*>(qvecs.get_data());
    for (int i_q=0; i_q < numQ; i_q++){
        int i = i_q*3;
        CUDAREAL qx = *(qptr+i);
        CUDAREAL qy = *(qptr+i+1);
        CUDAREAL qz = *(qptr+i+2);
        VEC3 Q(qx,qy,qz);
        gpu.qVecs[i_q] = Q;
    }
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if(verbose)
        printf("Pre-kernel time=%f msec\n", time);

    gettimeofday(&t1, 0);
    // run the kernel
    MAT3 Binv = gpu.Bmat.inverse();
    orientMultiply<<<gpu.numBlocks, gpu.blockSize>>>
        (gpu.qVecs, gpu.rotMats, numQ, gpu.numRot,
         gpu.out, hcut, minPred, Binv);

    error_msg(cudaGetLastError(), gpu.mpi_rank);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    if(verbose)
        printf("kernel time=%f msec\n", time);

}


__global__
void orientMultiply(VEC3* qVecs, MAT3* rotMats, int numQ,
                    int numRot, bool* out, CUDAREAL hcut,
                    int minPred, MAT3 Bmat){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;

    for (int i_rot=tid; i_rot < numRot; i_rot += thread_stride){
        int count=0;
        for (int i_q=0; i_q < numQ; i_q ++ ){
            VEC3 Hkl = Bmat*(rotMats[i_rot]*qVecs[i_q]);

            CUDAREAL h = ceil(Hkl[0]-0.5);
            CUDAREAL k = ceil(Hkl[1]-0.5);
            CUDAREAL l = ceil(Hkl[2]-0.5);
            VEC3 Hi(h,k,l);
            VEC3 deltaH = Hkl-Hi;
            CUDAREAL hnorm = deltaH.norm();
            if (hnorm < hcut)
                count += 1;
        }
        if (count >= minPred)
            out[i_rot] = true;
        else
            out[i_rot] = false;
    }
}

