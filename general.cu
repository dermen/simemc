
#include <mpi.h>
#include "emc_ext.h"
#include "general.cuh"

/*
  return free gpu memory in bytes for allocated device (gpu.device)
*/
size_t get_gpu_mem() {
   size_t free, total;
   cudaMemGetInfo( &free, &total );
   return free;
}


void copy_umats(MAT3* mats, np::ndarray& Umats, int numRot){
    MAT3 Umat; // orientation matrix
    CUDAREAL * u_ptr = reinterpret_cast<CUDAREAL*>(Umats.get_data());
    for (int i_rot=0; i_rot < numRot; i_rot ++){
        int i= i_rot*9;
        CUDAREAL uxx = *(u_ptr+i);
        CUDAREAL uxy = *(u_ptr+i+1);
        CUDAREAL uxz = *(u_ptr+i+2);
        CUDAREAL uyx = *(u_ptr+i+3);
        CUDAREAL uyy = *(u_ptr+i+4);
        CUDAREAL uyz = *(u_ptr+i+5);
        CUDAREAL uzx = *(u_ptr+i+6);
        CUDAREAL uzy = *(u_ptr+i+7);
        CUDAREAL uzz = *(u_ptr+i+8);
        Umat << uxx, uxy, uxz,
                uyx, uyy, uyz,
                uzx, uzy, uzz;
        mats[i_rot] = Umat.transpose();
    }
}


void broadcast_ipc_handle(cudaIpcMemHandle_t rotMats_memHand, CUDAREAL* rotMats, MPI_Comm COMM){
    int rank;
    MPI_Comm_rank(COMM, &rank);

    //  Broadcast the IPC handle
    int hand_size[1];
    if (rank==0)
        hand_size[0]= sizeof(rotMats_memHand);
    MPI_Bcast(&hand_size[0], 1, MPI_INT, 0, COMM);

    // create the char container for memHandler broadcast
    char memHand_C[hand_size[0]];
    if (rank==0)
        memcpy(&memHand_C, &rotMats_memHand, hand_size[0]);
    MPI_Bcast(&memHand_C, hand_size[0], MPI_BYTE, 0, COMM);
    if (rank >0)
        memcpy(&rotMats_memHand, &memHand_C, hand_size[0]);

    if (rank >0 )
        gpuErr(cudaIpcOpenMemHandle((void **) &rotMats, rotMats_memHand, cudaIpcMemLazyEnablePeerAccess));
}
