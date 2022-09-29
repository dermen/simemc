#ifndef CUDA_GEN_H
#define CUDA_GEN_H
#include "emc_ext.h"
#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }


static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static void error_msg(cudaError_t err, int rank){
    if (err != cudaSuccess){
        if (rank >=0)
            printf("RANK %d recvd CUDA error message: %s\n", rank, cudaGetErrorString(err));
        else
            printf("recvd CUDA error message: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

void broadcast_ipc_handle(cudaIpcMemHandle_t rotMats_memHand, CUDAREAL* rotMats, MPI_Comm COMM);


#endif
