
#include <cub/cub.cuh>
#include <mpi.h>
#include "emc_ext.h"
#include "general.cuh"

__device__ __inline__ int get_densities_index(int i,int j,int k, int nx, int ny, int nz, int* sparse_lookup);

__global__ void normalize_density(CUDAREAL* densities, CUDAREAL* wts, int N);

__global__ void reparameterize_density_gradients_kernel(CUDAREAL* dens_grad, int N);

__global__ void convert_reparameterized_densities_kernel(CUDAREAL* dens, int N);

__global__ void update_unmasked_kernel(CUDAREAL* unmasked_vals, int * unmasked_idx, CUDAREAL* dens, int N);

__global__ void trilinear_interpolation_rotate_on_GPU(const CUDAREAL* __restrict__ densities,
                                        VEC3*vectors, CUDAREAL* out, MAT3 rotMat,
                                        int num_qvec,
                                        int nx, int ny, int nz,
                                        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                        int* sparse_lookup);


__global__ void trilinear_insertion_rotate_on_GPU(
        CUDAREAL * densities,
        CUDAREAL * wts,
        CUDAREAL* insertion_values,
        const bool* is_trusted,
        CUDAREAL tomo_wt,
        VEC3 *vectors,
        MAT3 rotMat, int num_qvec,
        int nx, int ny, int nz,
        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
        int * sparse_lookup, bool use_trusted_flags);


__global__ void EMC_equation_two(const CUDAREAL* __restrict__ densities,
                                 CUDAREAL* densities_gradient,
                                 const CUDAREAL* __restrict__ data,
                                 const CUDAREAL* __restrict__ background,
                                 const bool* is_trusted,
                                 CUDAREAL scale_factor,
                                 VEC3*vectors, CUDAREAL* out_rot,
                                 MAT3* rotMats, int * rot_inds,
                                 int numRot, int num_qvec,
                                 int nx, int ny, int nz,
                                 CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                 CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                 const bool compute_scale_derivative,
                                 const bool compute_density_derivative,
                                 const bool poisson, CUDAREAL sigma_r_sq,
                                 int* sparse_lookup);

__global__ void dens_deriv(const CUDAREAL * __restrict__ densities,
                           CUDAREAL * densities_gradient,
                           const CUDAREAL * __restrict__ data,
                           const CUDAREAL * __restrict__ background,
                           const CUDAREAL * P_dr_vals,
                           const bool * is_trusted,
                           const bool * is_peak_in_density,
                           CUDAREAL scale_factor,
                           VEC3 *vectors,
                           CUDAREAL * out_rot,
                           MAT3* rotMats, int* rot_inds, int numRot, int num_qvec,
                           int nx, int ny, int nz,
                           CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                           CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                           int* sparse_lookup);


void do_after_kernel(int rank){
    error_msg(cudaGetLastError(),  rank);
    cudaDeviceSynchronize();
}


void sym_ops_to_dev(lerpy& gpu, np::ndarray& rot_mats){
    int num_sym_op = rot_mats.shape(0) / 9;
    if (gpu.rotMatsSym==NULL){
        gpuErr(cudaMallocManaged((void **)&gpu.rotMatsSym, num_sym_op*sizeof(MAT3)));
        gpu.num_sym_op = num_sym_op;
    }
    else if (gpu.rotMatsSym != NULL && num_sym_op != gpu.num_sym_op){
        printf("Warning: re-allocating for different number of sym ops\n");
        gpu.num_sym_op = num_sym_op;
    }

    // copy the rot mats to the device
    copy_umats(gpu.rotMatsSym, rot_mats, num_sym_op);
}


void symmetrize_density(lerpy& gpu, np::ndarray& _q_cent){
    if (gpu.rotMatsSym==NULL){
        printf("set the symmetry operators first! (see lerpy set_sym_ops method)\n");
        exit(1);
    }

    std::vector<CUDAREAL> q_cent;
    CUDAREAL* q_cent_ptr = reinterpret_cast<CUDAREAL*>(_q_cent.get_data());
    for (int i=0; i < gpu.densDim; i++){
        double q_val = *(q_cent_ptr+i);
        q_cent.push_back(q_val);
    }

    set_threads_blocks(gpu, gpu.numQ);

    //// here we store the current values of gpu.data and gpu.qVecs, as we will be hijacking those arrays for symmetrization
    eigVec3_vec temp_qvec;
    temp_qvec.reserve(gpu.numDataPixels);
    std::vector<CUDAREAL> temp_data;
    //std::vector<bool> temp_mask;
    // check that nuymDataPixles is same as numQ;
    if (gpu.numQ != gpu.numDataPixels){
        printf("Warning numQ=%d != numDataPixels=%d\n", gpu.numQ, gpu.numDataPixels);
        exit(1);
    }
    for (int i=0; i< gpu.numDataPixels; i++){
        temp_data.push_back(gpu.data[i]);
        temp_qvec.push_back(gpu.qVecs[i]);
        //temp_mask.push_back(gpu.mask[i]);
        //gpu.mask[i] = true; // mask only applies to the detector images, so ignore for symmetrization
    }

    // copy the density to a host array
    std::vector<CUDAREAL> current_density;
    current_density.resize(gpu.numDens);
    CUDAREAL * curr = &current_density[0];
    from_dev_memcpy(gpu.densities, curr, gpu.numDens);


    // reset the density and wts to 0
    toggle_insert_mode(gpu);

    // The gpu instance was allocated for the size of the detector
    // however here we are going to use that same allocated array to insert the density into itself
    // Typically the density will have more "voxels" than "pixels" in the detector, hence we will need to
    // chunk the density
    int Nvox = gpu.densDim*gpu.densDim*gpu.densDim;
    int n_chunk = Nvox / gpu.numDataPixels + 1;

    std::vector<int> sparse_lookup;
    if (gpu.sparse_lookup != NULL){
        sparse_lookup.resize(Nvox);
        gpuErr(cudaMemcpy(sparse_lookup.data(), gpu.sparse_lookup, sizeof(int)*Nvox, cudaMemcpyDeviceToHost));
    }

    for (int i_chunk=0; i_chunk < n_chunk; i_chunk++ ){
        int start=i_chunk*gpu.numDataPixels;
        int stop = start + gpu.numDataPixels;
        if (stop > Nvox)
            stop = Nvox;
        int densDim_sq = gpu.densDim*gpu.densDim;
        int i_q=0;
        for (int i_dens=start; i_dens < stop; i_dens++){
            if (gpu.sparse_lookup != NULL) {
                int sparse_idx = sparse_lookup[i_dens];
                gpu.data[i_q] = (sparse_idx==-1) ? 0 : current_density[sparse_idx];
            }
            else{
                gpu.data[i_q] = current_density[i_dens];
            }
            int qi = i_dens / densDim_sq;
            int qj = (i_dens/gpu.densDim) % gpu.densDim;
            int qk = i_dens % gpu.densDim;
            CUDAREAL dens_qx=q_cent[qi];
            CUDAREAL dens_qy=q_cent[qj];
            CUDAREAL dens_qz=q_cent[qk];
            gpu.qVecs[i_q] = VEC3(dens_qx, dens_qy, dens_qz);
            i_q ++;
        }
        int chunk_num_q = stop - start;
        for (int i_sym=0; i_sym < gpu.num_sym_op; i_sym++){
            trilinear_insertion_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
                    (gpu.densities, gpu.wts, gpu.data, gpu.mask, 1, gpu.qVecs,
                     gpu.rotMatsSym[i_sym], chunk_num_q,
                     gpu.densDim, gpu.densDim, gpu.densDim,
                     gpu.corner[0], gpu.corner[1], gpu.corner[2],
                     gpu.delta[0], gpu.delta[1], gpu.delta[2],
                     gpu.sparse_lookup, false);
            do_after_kernel(gpu.mpi_rank);
        }
    }

    //// copy the original data and qvectors (corresponding to the detector pixels) back to device
    for (int i=0; i < gpu.numDataPixels; i++){
        gpu.data[i] = temp_data[i];
        gpu.qVecs[i] = temp_qvec[i];
        //gpu.mask[i] = temp_mask[i];
    }

    // normalize the new density

    int numBlocks = (gpu.numDens + 128 - 1) / 128;
    normalize_density<<<numBlocks, 128 >>> (gpu.densities, gpu.wts, gpu.numDens);
    do_after_kernel(gpu.mpi_rank);
}


void update_masked_density_gpu(lerpy& gpu, np::ndarray& new_vals){
    CUDAREAL* vals_ptr = reinterpret_cast<CUDAREAL*>(new_vals.get_data());
    int N = new_vals.shape(0);
    // assertion on N
    // assert that gpu.unmasked_inds is not None
    if (gpu.unmasked_vals ==NULL){
        gpuErr(cudaMallocManaged((void**)&gpu.unmasked_vals, sizeof(CUDAREAL)*N));
    }
    for (int i=0; i < N; i++)
        gpu.unmasked_vals[i] = *(vals_ptr+i);
    int numBlocks = (N + 128 - 1) / 128;
    update_unmasked_kernel <<< numBlocks, 128>>> (gpu.unmasked_vals, gpu.unmasked_inds, gpu.densities, N);
    do_after_kernel(gpu.mpi_rank);
}


__global__ void update_unmasked_kernel(CUDAREAL* unmasked_vals, int * unmasked_idx, CUDAREAL* dens, int N){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_stride = blockDim.x * gridDim.x;
    for (int i=tid;  i< N; i+= thread_stride){
        int dens_idx = unmasked_idx[i];
        CUDAREAL dens_val = unmasked_vals[i];
        dens[dens_idx] = dens_val;
    }
}


__global__ void normalize_density(CUDAREAL* densities, CUDAREAL* wts, int N){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_stride = blockDim.x * gridDim.x;
    for (int i=tid; i < N; i+= thread_stride){
        CUDAREAL wt = wts[i];
        if (wt > 0)
            densities[i] = densities[i] / wt;
        else
            densities[i] = 0; // note: the density should already by 0 if wt=0
    }
}

/*
this method reparameterizes the gradients that are
used in the L-BFGS densityUpdater class method (see emc_updaters.py)
Basically this updates the gradients in place via
            grad *= -x/np.sqrt(x**2+1)
*/
void reparameterize_density_gradients(lerpy& gpu){
    set_threads_blocks(gpu, gpu.numDens);
    reparameterize_density_gradients_kernel<<< gpu.numBlocks, gpu.blockSize>>>(gpu.densities_gradient, gpu.numDens);
    do_after_kernel(gpu.mpi_rank);
}

__global__ void reparameterize_density_gradients_kernel(CUDAREAL* dens_grad, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    for(int i=tid; i< N; i+= thread_stride){
        CUDAREAL x = dens_grad[i];
        CUDAREAL num = -x;
        CUDAREAL den = sqrt(x*x+1);
        dens_grad[i] = x*num / den;
    }
}

/* Convert the densities array in place on the GPU according to
            #    theta = np.sqrt(x**2+1) -1
   This is used in the emc_updaters.py DensityUpdater class method target
   when running L-BFGS
   L-BFGS uses reparmaterized coordinates to keep the density positive during refinement
   hence when we receive a new density from L-BFGS refiner, we need to
   convert it back to normal coordinates. This can be done un numpy
   but its probable faster on GPU
*/
void convert_reparameterized_densities(lerpy& gpu){
    set_threads_blocks(gpu, gpu.numDens);
    convert_reparameterized_densities_kernel<<<gpu.numBlocks, gpu.blockSize>>>(gpu.densities, gpu.numDens);
    do_after_kernel(gpu.mpi_rank);
}

__global__ void convert_reparameterized_densities_kernel(CUDAREAL* dens, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    for(int i=tid; i< N; i+= thread_stride){
        CUDAREAL x = dens[i];
        dens[i] = sqrt(x*x+1)-1;
    }
}

/*
is_peak_in_density is a boolean np.ndarray with size=number of voxels. Its True if the voxel is to be modeled
note: it should be contiguous array (np.ascontiguousarray)
this allocates gpu.sparse_lookup, which should have size=densDim**3 (number of voxels)
Its values should be either -1 (if voxel is masked) or else an integer from 0 to M
where M is the number of unmasked voxels, i.e. the index of the voxel in a sparse density vector
*/
void set_sparse_lookup(lerpy& gpu, np::ndarray& is_peak_in_density){

    // get the device host communicator
    // TODO: test IPC for sparse_lookup array
    //MPI_Comm device_comm;
    //device_comm = get_host_dev_comm(gpu.device);
    //int dev_rank;
    //MPI_Comm_rank(device_comm, &dev_rank);

    int N = gpu.densDim*gpu.densDim*gpu.densDim;
    if (gpu.sparse_lookup == NULL)
        gpuErr(cudaMalloc((void**)&gpu.sparse_lookup, sizeof(int)*N));
    int * sparse_lookup = new int[N];
    bool* is_peak_ptr = reinterpret_cast<bool*>(is_peak_in_density.get_data());

    int numUnMasked=0;
    for (int i=0; i < N; i++){
        bool is_peak = *(is_peak_ptr+i);
        if (is_peak){
            sparse_lookup[i] =numUnMasked;
            numUnMasked += 1;
        }
        else{
            sparse_lookup[i] = -1;
        }
    }
    gpu.numDens = numUnMasked;
    gpuErr(cudaMemcpy(gpu.sparse_lookup, sparse_lookup, sizeof(int) * N, cudaMemcpyHostToDevice));
    delete sparse_lookup;
}

void set_device(lerpy& gpu){
    gpuErr(cudaSetDevice(gpu.device));
}


void prepare_for_lerping(lerpy& gpu, np::ndarray& Umats,
                        np::ndarray& qvectors, bool use_IPC){

    gpu.numQ = qvectors.shape(0)/3;
    // TODO global verbose flag
    if (gpu.numDens == -1)
        gpu.numDens = gpu.densDim*gpu.densDim*gpu.densDim;

    //// TODO asserts on len of corner and delta (must be 3)
    MPI_Comm device_comm;
    int dev_rank=-1;

    if (use_IPC){
        device_comm = get_host_dev_comm(gpu.device);
        MPI_Comm_rank(device_comm, &dev_rank);
        if (dev_rank==0)
            gpu.numRot = Umats.shape(0)/9;
        MPI_Bcast(&gpu.numRot, 1, MPI_INT, 0, device_comm);
    }
    else
        gpu.numRot = Umats.shape(0)/9;

    cudaIpcMemHandle_t rotMats_memHand;
    if (dev_rank==0 && use_IPC)
        get_mem_handle(rotMats_memHand, gpu.rotMats, Umats, gpu.numRot);

    if (!use_IPC)
        gpuErr(cudaMallocManaged((void **)&gpu.rotMats, gpu.numRot*sizeof(MAT3)));

    // TODO assert on numDens > 0
    gpuErr(cudaMalloc((void ** )&gpu.densities, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMemset(gpu.densities, 0, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMalloc((void ** )&gpu.densities_gradient, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.out, gpu.maxNumQ*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.out_equation_two, gpu.maxNumRotInds*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.qVecs, gpu.maxNumQ*sizeof(VEC3)));
    gpuErr(cudaMallocManaged((void **)&gpu.rotInds, gpu.maxNumRotInds*sizeof(int)));
    gpuErr(cudaMallocManaged((void **)&gpu.Pdr, gpu.maxNumRotInds*sizeof(CUDAREAL)));
    gpuErr(cudaMallocManaged((void **)&gpu.data, gpu.numDataPixels*sizeof(CUDAREAL)));
    gpuErr(cudaMalloc((void **)&gpu.mask, gpu.numDataPixels*sizeof(bool)));
    gpuErr(cudaMalloc((void **)&gpu.background, gpu.numDataPixels*sizeof(CUDAREAL)));

    // broadcast and copy the memoryhandle to gpu.rotMats on other processes
    if (use_IPC){
        broadcast_ipc_handle(rotMats_memHand, gpu.rotMats, device_comm);
        gpu.free_rotMats = (dev_rank==0) ;
        gpu.close_rotMats_handle = (dev_rank >0) ;
    }
    else{
        copy_umats(gpu.rotMats, Umats, gpu.numRot);
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

    gpu.is_allocated = true;

}


void copy_image_data_to_device(lerpy& gpu, np::ndarray& shot_data, np::ndarray& shot_mask, np::ndarray& shot_background){
    unsigned int num_pix = shot_data.shape(0);
    CUDAREAL* data_ptr = reinterpret_cast<CUDAREAL*>(shot_data.get_data());
    bool* mask_ptr = reinterpret_cast<bool*>(shot_mask.get_data());
    CUDAREAL* background_ptr = reinterpret_cast<CUDAREAL*>(shot_background.get_data());
    for (int i=0; i < num_pix; i++) {
        gpu.data[i] = *(data_ptr + i);
        //gpu.mask[i] = *(mask_ptr + i);
        //gpu.background[i] = *(background_ptr + i);
    }
    to_dev_memcpy(gpu.background, background_ptr, num_pix);
    cudaMemcpy(gpu.mask, mask_ptr, sizeof(bool)*num_pix, cudaMemcpyHostToDevice);
}

void to_dev_memcpy(CUDAREAL* dev_ptr, CUDAREAL* host_ptr, int N){
    gpuErr(cudaMemcpy(dev_ptr, host_ptr, sizeof(CUDAREAL) * N, cudaMemcpyHostToDevice));
}

void from_dev_memcpy( CUDAREAL* dev_ptr,  CUDAREAL* host_ptr, int N){
    gpuErr(cudaMemcpy(host_ptr, dev_ptr, sizeof(CUDAREAL) * N, cudaMemcpyDeviceToHost));
}

void from_dev_memcpy_int( int* dev_ptr,  int* host_ptr, int N){
    gpuErr(cudaMemcpy(host_ptr, dev_ptr, sizeof(int) * N, cudaMemcpyDeviceToHost));
}

void densities_to_device(lerpy& gpu, np::ndarray& new_densities){
    unsigned int numDens = new_densities.shape(0);
    CUDAREAL* dens_ptr = reinterpret_cast<CUDAREAL*>(new_densities.get_data());
    to_dev_memcpy(gpu.densities, dens_ptr, gpu.numDens);
}

void reset_dens_deriv(lerpy& gpu){
    cudaMemset(gpu.densities_gradient, 0, sizeof(CUDAREAL)*gpu.numDens);
}

void malloc_relp_mask(lerpy& gpu){
    gpuErr(cudaMalloc((void ** )&gpu.is_peak_in_density, gpu.numDens*sizeof(bool)));
}

void malloc_unmasked_inds(lerpy& gpu){
//  TODO assert gpu.num_unmasked != -1
    gpuErr(cudaMalloc((void ** )&gpu.unmasked_inds, gpu.num_unmasked*sizeof(int)));
}

void relp_mask_to_device(lerpy& gpu, np::ndarray& relp_mask){
    unsigned int numDens = relp_mask.shape(0);
    bool* relp_mask_ptr = reinterpret_cast<bool*>(relp_mask.get_data());

    // TODO put the assert numDens==gpu.numDens
    if (gpu.is_peak_in_density==NULL){
        malloc_relp_mask(gpu);
    }

    if (gpu.unmasked_inds == NULL){
        std::vector<int> unmasked_inds;
        for(int i=0; i < gpu.numDens; i++){
            bool is_masked = relp_mask[i]; //*(relp_mask_ptr+i);
            if (is_masked)
                unmasked_inds.push_back(i);
        }
        int num_unmasked = unmasked_inds.size();
        gpuErr(cudaMalloc((void ** )&gpu.unmasked_inds, num_unmasked*sizeof(int)));
        gpuErr(cudaMemcpy(gpu.unmasked_inds, unmasked_inds.data(), sizeof(int)*num_unmasked, cudaMemcpyHostToDevice) );
        gpu.num_unmasked = num_unmasked;
    }
    bool * temp = new bool[gpu.numDens];
    for (int i=0; i< gpu.numDens; i++)
        temp[i] = *(relp_mask_ptr+i);
    gpuErr(cudaMemcpy(gpu.is_peak_in_density, temp, sizeof(bool) * gpu.numDens, cudaMemcpyHostToDevice));
    delete temp;
}

void toggle_insert_mode(lerpy& gpu){
    if (gpu.wts==NULL){
        gpuErr(cudaMalloc((void ** )&gpu.wts, gpu.numDens*sizeof(CUDAREAL)));
    }
    gpuErr(cudaMemset(gpu.wts, 0, gpu.numDens*sizeof(CUDAREAL)));
    gpuErr(cudaMemset(gpu.densities, 0, gpu.numDens*sizeof(CUDAREAL)));

}


void set_threads_blocks(lerpy& gpu, int N){
    // optional size of each device block, else default to 128
    char *threads = getenv("ORIENT_THREADS_PER_BLOCK");
    if (threads == NULL)
        gpu.blockSize = 128;
    else
        gpu.blockSize = atoi(threads);
    //gpu.blockSize=blockSize;
    gpu.numBlocks = (N + gpu.blockSize - 1) / gpu.blockSize;
}

void do_a_lerp(lerpy& gpu, std::vector<int>& rot_inds, bool verbose, int task) {
    double time;
    struct timeval t1, t2;//, t3 ,t4;

    gettimeofday(&t1, 0);

    set_threads_blocks(gpu, gpu.numQ);

    int numRotInds = rot_inds.size();
    for (int i=0; i< numRotInds; i++){
        gpu.rotInds[i] = rot_inds[i];
        if(task==1 || task==3 || task==4 || task==5){
            gpu.out_equation_two[i] = 0;
        }
    }
    if(task==4 || task==5 && gpu.alwaysResetDeriv){
        gpuErr(cudaMemset(gpu.densities_gradient, 0, gpu.numDens*sizeof(CUDAREAL)));
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
        MAT3 rotMat;
        int offset = gpu.rotInds[0];
        cudaMemcpy(&rotMat, gpu.rotMats+offset, sizeof(MAT3), cudaMemcpyDeviceToHost);
        trilinear_interpolation_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.qVecs, gpu.out,
                 rotMat, gpu.numQ,
                 gpu.densDim, gpu.densDim, gpu.densDim,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2],
                 gpu.sparse_lookup);
    }
    else if(task==1 || task==3 || task==4) {
        if (verbose)printf("Running equation 2! Shotscale=%f\n", gpu.shot_scale);
        bool use_poisson_stats=true;
        CUDAREAL sigma_r_sq = 0.5;
        // task 1: compute image logLikelihood
        // task 3: compute deriv of image logLikelihood w.r.t scale factor
        // task 4: ""  "" w.r.t. density
        if (gpu.sparse_lookup != NULL){
        }
        EMC_equation_two<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.densities_gradient,
                 gpu.data, gpu.background, gpu.mask,
                 gpu.shot_scale, gpu.qVecs, gpu.out_equation_two,
                 gpu.rotMats, gpu.rotInds, numRotInds, gpu.numQ,
                 gpu.densDim, gpu.densDim, gpu.densDim,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2],
                 task==3, task==4,
                 use_poisson_stats, sigma_r_sq,
                 gpu.sparse_lookup);

    }
    else if (task==5){
        for (int i=0; i< numRotInds; i++)
            gpu.Pdr[i] = gpu.Pdr_host[i];
        if (gpu.is_peak_in_density==NULL){
            if (gpu.sparse_lookup == NULL)
                printf("WARNING! NO RELP MASK ALLOCATED: use copy_relp_mask method or set_sparse_lookup\n");
            // TODO : do we need to do this if sparse_lookup != NULL ?
            malloc_relp_mask(gpu);
            bool* temp = new bool[gpu.numDens];
            for(int i=0; i < gpu.numDens;i++)
                temp[i] = true;
            gpuErr(cudaMemcpy(gpu.is_peak_in_density, temp, sizeof(bool) * gpu.numDens, cudaMemcpyHostToDevice));
            delete temp;
        }

        dens_deriv<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.densities_gradient,
                 gpu.data, gpu.background, gpu.Pdr, gpu.mask, gpu.is_peak_in_density,
                 gpu.shot_scale, gpu.qVecs, gpu.out_equation_two,
                 gpu.rotMats, gpu.rotInds, numRotInds, gpu.numQ,
                 gpu.densDim, gpu.densDim, gpu.densDim,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2],
                 gpu.sparse_lookup);
    }
    else if (task==2)  {
        MAT3 rotMat;
        int offset = gpu.rotInds[0];
        cudaMemcpy(&rotMat, gpu.rotMats+offset, sizeof(MAT3), cudaMemcpyDeviceToHost);
//      NOTE: here gpu.data are the insert values
        trilinear_insertion_rotate_on_GPU<<<gpu.numBlocks, gpu.blockSize>>>
                (gpu.densities, gpu.wts, gpu.data, gpu.mask, gpu.tomogram_wt, gpu.qVecs,
                 rotMat, gpu.numQ,
                 gpu.densDim, gpu.densDim, gpu.densDim,
                 gpu.corner[0], gpu.corner[1], gpu.corner[2],
                 gpu.delta[0], gpu.delta[1], gpu.delta[2],
                 gpu.sparse_lookup, true);
   
    }
    do_after_kernel(gpu.mpi_rank);
    if (verbose) {
        gettimeofday(&t2, 0);
        time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("kernel time=%f msec\n", time);
    }

    gettimeofday(&t1, 0);
    if (task==1 || task==3 || task==4 || task==5){
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
//  TODO free sym ops
    if (gpu.sparse_lookup != NULL)
        gpuErr(cudaFree(gpu.sparse_lookup));
    if(gpu.unmasked_vals != NULL)
        gpuErr(cudaFree(gpu.unmasked_vals));
    if(gpu.unmasked_inds != NULL)
        gpuErr(cudaFree(gpu.unmasked_inds));
    if (gpu.qVecs != NULL)
        gpuErr(cudaFree(gpu.qVecs));
    if (gpu.rotInds!= NULL)
        gpuErr(cudaFree(gpu.rotInds));
    if (gpu.Pdr!= NULL)
        gpuErr(cudaFree(gpu.Pdr));
    if (gpu.rotMats!= NULL && gpu.free_rotMats)
        gpuErr(cudaFree(gpu.rotMats));
    if (gpu.data!= NULL)
        gpuErr(cudaFree(gpu.data));
    if (gpu.mask!= NULL)
        gpuErr(cudaFree(gpu.mask));
    if (gpu.is_peak_in_density!= NULL)
        gpuErr(cudaFree(gpu.is_peak_in_density));
    if (gpu.background!= NULL)
        gpuErr(cudaFree(gpu.background));
    if (gpu.out!= NULL)
        gpuErr(cudaFree(gpu.out));
    if (gpu.out_equation_two!= NULL)
        gpuErr(cudaFree(gpu.out_equation_two));
    if (gpu.densities!= NULL)
        gpuErr(cudaFree(gpu.densities));
    if (gpu.densities_gradient!= NULL)
        gpuErr(cudaFree(gpu.densities_gradient));
    if (gpu.wts!= NULL)
        gpuErr(cudaFree(gpu.wts));
    if (gpu.close_rotMats_handle)
        gpuErr(cudaIpcCloseMemHandle(gpu.rotMats));
    gpu.is_allocated = false;
}

__device__ __inline__ int get_densities_index(int i,int j,int k, int nx, int ny, int nz, int* sparse_lookup)
{
    //int idx = i + j*nx + k*nx*ny;
    int idx = fma(nx, fma(k,ny,j), i);
    if (sparse_lookup != NULL)
        idx = sparse_lookup[idx];
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
                                        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                        int* sparse_lookup){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_stride = blockDim.x * gridDim.x;
    CUDAREAL i_f, j_f, k_f;
    CUDAREAL x0,x1,y0,y1,z0,z1;
    CUDAREAL qx,qy,qz;
    int i0, i1, j0, j1, k0, k1;
    CUDAREAL I0,I1,I2,I3,I4,I5,I6,I7;
    CUDAREAL a0,a1,a2,a3,a4,a5,a6,a7;
    CUDAREAL x0y0, x1y1, x0y1, x1y0;
    int idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7;
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

        idx0 = get_densities_index(i0, j0, k0, nx, ny, nz,sparse_lookup);
        idx1 = get_densities_index(i1, j0, k0, nx, ny, nz,sparse_lookup);
        idx2 = get_densities_index(i0, j1, k0, nx, ny, nz,sparse_lookup);
        idx3 = get_densities_index(i0, j0, k1, nx, ny, nz,sparse_lookup);
        idx4 = get_densities_index(i1, j0, k1, nx, ny, nz,sparse_lookup);
        idx5 = get_densities_index(i0, j1, k1, nx, ny, nz,sparse_lookup);
        idx6 = get_densities_index(i1, j1, k0, nx, ny, nz,sparse_lookup);
        idx7 = get_densities_index(i1, j1, k1, nx, ny, nz,sparse_lookup);

        I0 = (idx0 ==-1) ? 0: __ldg(&densities[idx0]);
        I1 = (idx1 ==-1) ? 0: __ldg(&densities[idx1]);
        I2 = (idx2 ==-1) ? 0: __ldg(&densities[idx2]);
        I3 = (idx3 ==-1) ? 0: __ldg(&densities[idx3]);
        I4 = (idx4 ==-1) ? 0: __ldg(&densities[idx4]);
        I5 = (idx5 ==-1) ? 0: __ldg(&densities[idx5]);
        I6 = (idx6 ==-1) ? 0: __ldg(&densities[idx6]);
        I7 = (idx7 ==-1) ? 0: __ldg(&densities[idx7]);

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
        const bool* is_trusted,
        CUDAREAL tomo_wt,
        VEC3 *vectors,
        MAT3 rotMat, int num_qvec,
        int nx, int ny, int nz,
        CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
        CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
        int* sparse_lookup, bool use_trusted_flags){

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
        if (use_trusted_flags && !is_trusted[i]){
            continue;
        }
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
        idx0 = get_densities_index(i0, j0, k0, nx, ny, nz,sparse_lookup);
        idx1 = get_densities_index(i1, j0, k0, nx, ny, nz,sparse_lookup);
        idx2 = get_densities_index(i0, j1, k0, nx, ny, nz,sparse_lookup);
        idx3 = get_densities_index(i0, j0, k1, nx, ny, nz,sparse_lookup);
        idx4 = get_densities_index(i1, j0, k1, nx, ny, nz,sparse_lookup);
        idx5 = get_densities_index(i0, j1, k1, nx, ny, nz,sparse_lookup);
        idx6 = get_densities_index(i1, j1, k0, nx, ny, nz,sparse_lookup);
        idx7 = get_densities_index(i1, j1, k1, nx, ny, nz,sparse_lookup);

        if (idx0>0){
            atomicAdd(&densities[idx0], val*a0);
            atomicAdd(&wts[idx0], a0);
        }
        if (idx1>0){
            atomicAdd(&densities[idx1], val*a1);
            atomicAdd(&wts[idx1], a1);
        }
        if (idx2>0){
            atomicAdd(&densities[idx2], val*a2);
            atomicAdd(&wts[idx2], a2);
        }
        if (idx3>0){
            atomicAdd(&densities[idx3], val*a3);
            atomicAdd(&wts[idx3], a3);
        }
        if (idx4>0){
            atomicAdd(&densities[idx4], val*a4);
            atomicAdd(&wts[idx4], a4);
        }
        if (idx5>0){
            atomicAdd(&densities[idx5], val*a5);
            atomicAdd(&wts[idx5], a5);
        }
        if (idx6>0){
            atomicAdd(&densities[idx6], val*a6);
            atomicAdd(&wts[idx6], a6);
        }
        if (idx7>0){
            atomicAdd(&densities[idx7], val*a7);
            atomicAdd(&wts[idx7], a7);
        }

    }
}

/*
 * Computes equation (2) in http://dx.doi.org/10.1107/S1600576716008165
 *
 */
__global__ void EMC_equation_two(const CUDAREAL * __restrict__ densities,
                                 CUDAREAL * densities_gradient,
                                 const CUDAREAL * __restrict__ data,
                                 const CUDAREAL * __restrict__ background,
                                 const bool * is_trusted,
                                 CUDAREAL scale_factor,
                                 VEC3 *vectors,
                                 CUDAREAL * out_rot,
                                 MAT3* rotMats, int* rot_inds, int numRot, int num_qvec,
                                 int nx, int ny, int nz,
                                 CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                                 CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                                 const bool compute_scale_derivative,
                                 const bool compute_density_derivative,
                                 const bool poisson, CUDAREAL sigma_r_sq,
                                 int* sparse_lookup){

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

    CUDAREAL u,v;  // for gaussian model, these are the model residual and the variance of the pixel;
    CUDAREAL model; // for the poisson model, this is scale_factor * tomogram
    CUDAREAL R_dr_thread, R_dr_block;
    typedef cub::BlockReduce<CUDAREAL, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    MAT3 R;
    CUDAREAL eps=1e-6;

    CUDAREAL Bkg_t; // value of background at pixel
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
    CUDAREAL W_rt_prime0, W_rt_prime1, W_rt_prime2, W_rt_prime3, W_rt_prime4, W_rt_prime5, W_rt_prime6, W_rt_prime7;
    CUDAREAL dens_grad_factor;

    for (i_rot =0; i_rot < numRot; i_rot++){
        R_dr_thread = 0;
        r = rot_inds[i_rot];
        R = rotMats[r];
        for (t=tid; t < num_qvec; t += thread_stride){
            if (! is_trusted[t]){
                continue;
            }
            K_t = __ldg(&data[t]);
            Bkg_t = __ldg(&background[t]);
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

            idx0 = get_densities_index(i0, j0, k0, nx, ny, nz,sparse_lookup);
            idx1 = get_densities_index(i1, j0, k0, nx, ny, nz,sparse_lookup);
            idx2 = get_densities_index(i0, j1, k0, nx, ny, nz,sparse_lookup);
            idx3 = get_densities_index(i0, j0, k1, nx, ny, nz,sparse_lookup);
            idx4 = get_densities_index(i1, j0, k1, nx, ny, nz,sparse_lookup);
            idx5 = get_densities_index(i0, j1, k1, nx, ny, nz,sparse_lookup);
            idx6 = get_densities_index(i1, j1, k0, nx, ny, nz,sparse_lookup);
            idx7 = get_densities_index(i1, j1, k1, nx, ny, nz,sparse_lookup);

            I0 = (idx0 ==-1) ? 0: __ldg(&densities[idx0]);
            I1 = (idx1 ==-1) ? 0: __ldg(&densities[idx1]);
            I2 = (idx2 ==-1) ? 0: __ldg(&densities[idx2]);
            I3 = (idx3 ==-1) ? 0: __ldg(&densities[idx3]);
            I4 = (idx4 ==-1) ? 0: __ldg(&densities[idx4]);
            I5 = (idx5 ==-1) ? 0: __ldg(&densities[idx5]);
            I6 = (idx6 ==-1) ? 0: __ldg(&densities[idx6]);
            I7 = (idx7 ==-1) ? 0: __ldg(&densities[idx7]);

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

            if (poisson){
                model = Bkg_t + scale_factor*W_rt;
                if (compute_scale_derivative) {
                    if (model > -eps)
                        R_dr_thread += K_t*W_rt/(model+eps) - W_rt;
                }
                else if (compute_density_derivative){
                    if (model > -eps) {
                        dens_grad_factor = K_t*scale_factor / (model+eps)-scale_factor;

                        W_rt_prime0 = dens_grad_factor*a0;
                        W_rt_prime1 = dens_grad_factor*a1;
                        W_rt_prime2 = dens_grad_factor*a2;
                        W_rt_prime3 = dens_grad_factor*a3;
                        W_rt_prime4 = dens_grad_factor*a4;
                        W_rt_prime5 = dens_grad_factor*a5;
                        W_rt_prime6 = dens_grad_factor*a6;
                        W_rt_prime7 = dens_grad_factor*a7;

                        if (idx0>0)
                            atomicAdd(&densities_gradient[idx0], W_rt_prime0);
                        if (idx1>0)
                            atomicAdd(&densities_gradient[idx1], W_rt_prime1);
                        if (idx2>0)
                            atomicAdd(&densities_gradient[idx2], W_rt_prime2);
                        if (idx3>0)
                            atomicAdd(&densities_gradient[idx3], W_rt_prime3);
                        if (idx4>0)
                            atomicAdd(&densities_gradient[idx4], W_rt_prime4);
                        if (idx5>0)
                            atomicAdd(&densities_gradient[idx5], W_rt_prime5);
                        if (idx6>0)
                            atomicAdd(&densities_gradient[idx6], W_rt_prime6);
                        if (idx7>0)
                            atomicAdd(&densities_gradient[idx7], W_rt_prime7);
                    }
                }
                else{ // compute logLikelihood
                    if (model > -eps)
                        R_dr_thread += K_t * log(model+eps) - model;
                }
            }
            else{
                // THIS MODEL IS NOT READY!
                u = K_t-W_rt;
                v = W_rt + sigma_r_sq;
                if (v >0)
                    R_dr_thread += -0.5* (log(v) + u*u/v);
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


/*
 * Computes derivative of logLikelihood w.r.t densities
 * Likelihood component = Sum_rot P_dr * Sum_pix [Data_pix * log(Model_pix) - Model_pix]
 * This method computes derivative of above w.r.t. the density component of the Model_pix
 * Model_pix = background_pix + scale_d * W_rot_pix  where W_rot_pix is a tomogram slice of the density
 * for given orientation (rot) and pixel (pix)
 */
__global__ void dens_deriv(const CUDAREAL * __restrict__ densities,
                           CUDAREAL * densities_gradient,
                           const CUDAREAL * __restrict__ data,
                           const CUDAREAL * __restrict__ background,
                           const CUDAREAL * P_dr_vals,
                           const bool * is_trusted,
                           const bool * is_peak_in_density,
                           CUDAREAL scale_factor,
                           VEC3 *vectors,
                           CUDAREAL * out_rot,
                           MAT3* rotMats, int* rot_inds, int numRot, int num_qvec,
                           int nx, int ny, int nz,
                           CUDAREAL cx, CUDAREAL cy, CUDAREAL cz,
                           CUDAREAL dx, CUDAREAL dy, CUDAREAL dz,
                           int* sparse_lookup){

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

    CUDAREAL model; // for the poisson model, this is scale_factor * tomogram
    typedef cub::BlockReduce<CUDAREAL, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    MAT3 R;
    CUDAREAL eps=1e-6;

    CUDAREAL Bkg_t; // value of background at pixel
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;
    CUDAREAL W_rt_prime0, W_rt_prime1, W_rt_prime2, W_rt_prime3, W_rt_prime4, W_rt_prime5, W_rt_prime6, W_rt_prime7;
    CUDAREAL dens_grad_factor;
    CUDAREAL P_dr;
    CUDAREAL R_dr_thread, R_dr_block;

    for (i_rot =0; i_rot < numRot; i_rot++){
        R_dr_thread = 0;
        P_dr = P_dr_vals[i_rot];
        r = rot_inds[i_rot];
        R = rotMats[r];
        for (t=tid; t < num_qvec; t += thread_stride){
            if (! is_trusted[t]){
                continue;
            }
            K_t = __ldg(&data[t]);
            Bkg_t = __ldg(&background[t]);
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

            idx0 = get_densities_index(i0, j0, k0, nx, ny, nz,sparse_lookup);
            idx1 = get_densities_index(i1, j0, k0, nx, ny, nz,sparse_lookup);
            idx2 = get_densities_index(i0, j1, k0, nx, ny, nz,sparse_lookup);
            idx3 = get_densities_index(i0, j0, k1, nx, ny, nz,sparse_lookup);
            idx4 = get_densities_index(i1, j0, k1, nx, ny, nz,sparse_lookup);
            idx5 = get_densities_index(i0, j1, k1, nx, ny, nz,sparse_lookup);
            idx6 = get_densities_index(i1, j1, k0, nx, ny, nz,sparse_lookup);
            idx7 = get_densities_index(i1, j1, k1, nx, ny, nz,sparse_lookup);

            I0 = (idx0 < 0) ? 0: __ldg(&densities[idx0]);
            I1 = (idx1 < 0) ? 0: __ldg(&densities[idx1]);
            I2 = (idx2 < 0) ? 0: __ldg(&densities[idx2]);
            I3 = (idx3 < 0) ? 0: __ldg(&densities[idx3]);
            I4 = (idx4 < 0) ? 0: __ldg(&densities[idx4]);
            I5 = (idx5 < 0) ? 0: __ldg(&densities[idx5]);
            I6 = (idx6 < 0) ? 0: __ldg(&densities[idx6]);
            I7 = (idx7 < 0) ? 0: __ldg(&densities[idx7]);

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

            model = Bkg_t + scale_factor*W_rt;
            if (model > -eps) {
                dens_grad_factor = P_dr* ( K_t*scale_factor / (model+eps)-scale_factor);

                W_rt_prime0 = dens_grad_factor*a0;
                W_rt_prime1 = dens_grad_factor*a1;
                W_rt_prime2 = dens_grad_factor*a2;
                W_rt_prime3 = dens_grad_factor*a3;
                W_rt_prime4 = dens_grad_factor*a4;
                W_rt_prime5 = dens_grad_factor*a5;
                W_rt_prime6 = dens_grad_factor*a6;
                W_rt_prime7 = dens_grad_factor*a7;

                if (sparse_lookup==NULL){
                    if (is_peak_in_density[idx0])
                        atomicAdd(&densities_gradient[idx0], W_rt_prime0);
                    if (is_peak_in_density[idx1])
                        atomicAdd(&densities_gradient[idx1], W_rt_prime1);
                    if (is_peak_in_density[idx2])
                        atomicAdd(&densities_gradient[idx2], W_rt_prime2);
                    if (is_peak_in_density[idx3])
                        atomicAdd(&densities_gradient[idx3], W_rt_prime3);
                    if (is_peak_in_density[idx4])
                        atomicAdd(&densities_gradient[idx4], W_rt_prime4);
                    if (is_peak_in_density[idx5])
                        atomicAdd(&densities_gradient[idx5], W_rt_prime5);
                    if (is_peak_in_density[idx6])
                        atomicAdd(&densities_gradient[idx6], W_rt_prime6);
                    if (is_peak_in_density[idx7])
                        atomicAdd(&densities_gradient[idx7], W_rt_prime7);
                }
                else{
                    if (idx0 >= 0)
                        atomicAdd(&densities_gradient[idx0], W_rt_prime0);
                    if (idx1 >= 0)
                        atomicAdd(&densities_gradient[idx1], W_rt_prime1);
                    if (idx2 >= 0)
                        atomicAdd(&densities_gradient[idx2], W_rt_prime2);
                    if (idx3 >= 0)
                        atomicAdd(&densities_gradient[idx3], W_rt_prime3);
                    if (idx4 >= 0)
                        atomicAdd(&densities_gradient[idx4], W_rt_prime4);
                    if (idx5 >= 0)
                        atomicAdd(&densities_gradient[idx5], W_rt_prime5);
                    if (idx6 >= 0)
                        atomicAdd(&densities_gradient[idx6], W_rt_prime6);
                    if (idx7 >= 0)
                        atomicAdd(&densities_gradient[idx7], W_rt_prime7);
                }

                R_dr_thread += K_t * log(model+eps) - model;
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
