#ifndef EMC_CUH
#define EMC_CUH

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static void error_msg(cudaError_t err, const char* msg, int rank){
    if (err != cudaSuccess){
        if (rank >=0)
            printf("RANK %d;%s recvd CUDA error message: %s\n", rank, msg, cudaGetErrorString(err));
        else
            printf("%s recvd CUDA error message: %s\n", msg, cudaGetErrorString(err));
        exit(err);
    }
}

#endif