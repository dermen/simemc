
#include <boost/asio/ip/host_name.hpp>
#include <string>
#include "emc_ext.h"
#include "general.cuh"
#include <functional>
#include <stdint.h>
#include <limits.h>
#include <unordered_map>


#if SIZE_MAX == UCHAR_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "no statement"
#endif


/*
  return free gpu memory in bytes for allocated device (gpu.device)
*/
size_t get_gpu_mem() {
   size_t free, total;
   gpuErr(cudaMemGetInfo( &free, &total ));
   return free;
}


void copy_umats(MAT3*& mats, np::ndarray& Umats, int numRot){
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

/*
When muliple processes share a device, use this to map the
orientation matrix memory allocated by the root process
rotMats_memHand: cuda memory handle which instructs other processes
                how to read the memory allocated by the root process
rotMats: device pointer (MAT3, defined in emc_ext.h)
COMM: communicator for ranks sharing a single physical device (see mpi_utils.py get_host_dev_comm)
*/
void broadcast_ipc_handle(cudaIpcMemHandle_t& rotMats_memHand, MAT3*& rotMats, MPI_Comm& COMM){
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

/*
    get an MPI communicator for all ranks sharing a specific device on a specific host
*/

MPI_Comm get_host_dev_comm(int dev_id){
    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string host_name = boost::asio::ip::host_name();
    std::string dev_name = "gpudev" + std::to_string(dev_id);
    std::string host_key = host_name + "-" + dev_name;
    std::hash<std::string> hasher;
    size_t hash_key = hasher(host_key);
//    std::cout <<host_key << " " <<hash_key<<std::endl;

    std::vector<size_t> recvlen;
    if (world_rank==0)
        recvlen.resize(world_size);
    MPI_Gather(&hash_key, 1, MPI_SIZE_T, recvlen.data(), 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);

    std::vector <size_t> map_keys;
    std::vector<int> map_vals;
    std::unordered_map<size_t, int> host_map;
    if (world_rank==0){
        for(int i=0; i < recvlen.size(); i++){
            size_t key=recvlen[i];
            if (host_map.count(key)==0){
                host_map[key] = host_map.size();
            }
        }
        for(auto it = host_map.begin(); it != host_map.end(); ++it){
//            std::cout<< it->first<<" " <<it->second<< std::endl;
            map_keys.push_back(it->first);
            map_vals.push_back(it->second);
        }
    }

    int map_len;
    if(world_rank==0)
        map_len = host_map.size();
    MPI_Bcast(&map_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank > 0){
        map_keys.resize(map_len);
        map_vals.resize(map_len);
    }

    MPI_Bcast(map_keys.data(), map_len, MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(map_vals.data(), map_len, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank>0){
        for (int i=0; i < map_len; i++){
            host_map[map_keys[i]] = map_vals[i];
        }
    }

    MPI_Comm device_comm;
    int color = host_map[hash_key];
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &device_comm);
    int dev_rank, dev_size;
    MPI_Comm_rank(device_comm, &dev_rank);
    MPI_Comm_size(device_comm, &dev_size);
    printf("world rank=%d, device rank=%d, device_id = %d\n", world_rank, dev_rank, dev_id);
    return device_comm;
}


void get_mem_handle(cudaIpcMemHandle_t& rotMats_memHand, MAT3*& rotMats, np::ndarray& Umats, int numRot){
    gpuErr(cudaMalloc((void ** )&rotMats, sizeof(MAT3) * numRot));
    gpuErr(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &rotMats_memHand, (void *)rotMats));
    MAT3 * temp = new MAT3[numRot];
    copy_umats(temp, Umats, numRot);
    gpuErr(cudaMemcpy(rotMats, temp, sizeof(MAT3) * numRot, cudaMemcpyHostToDevice));
    delete temp;
}
