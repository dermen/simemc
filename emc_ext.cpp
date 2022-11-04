
#include <Python.h>
// prevents an annoying compiler warning about auto_ptr deprecation:
#define BOOST_NO_AUTO_PTR
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/args.hpp>
#include <iostream>
#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include "emc_ext.h"
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
namespace bp=boost::python;
namespace np=boost::python::numpy;


// TODO: lerpy should be initialized with dens_dim and max_q args
class lerpyExt{
    public:
    virtual ~lerpyExt(){}
    lerpyExt(){}
    lerpy gpu;
    bool auto_convert_arrays = true;
    bool has_sym_ops = false;
    int size_of_cudareal = sizeof(CUDAREAL);


    inline void contig_check(np::ndarray& vals){
       if (!(vals.get_flags() & np::ndarray::C_CONTIGUOUS)){
           PyErr_SetString(PyExc_TypeError, "Array must be C-contig and of type CUDAREAL\n" );
           bp::throw_error_already_set();
       }
    }


    inline void type_check(np::ndarray& vals){
        contig_check(vals);

        np::dtype vals_t = vals.get_dtype();
        int vals_size = vals_t.get_itemsize();
        bool types_agree= (size_of_cudareal != vals_size);

        if (! types_agree){
            if (size_of_cudareal==4)
                PyErr_SetString(PyExc_TypeError, "Array must of type CUDAREAL=float (np.float32)\n" );
            else if (size_of_cudareal==8)
                PyErr_SetString(PyExc_TypeError, "Array must of type CUDAREAL=double, (np.float64)\n" );
            else{
                printf("sizeof(CUDAREAL) = %d\n", size_of_cudareal);
                PyErr_SetString(PyExc_TypeError, "Array must of type CUDAREAL\n" );
            }
            bp::throw_error_already_set();
        }
    }

    inline size_t _get_gpu_mem(){
        return get_gpu_mem();
    }

    inline int get_dev_id(){
        return gpu.device;
    }

    inline void set_dev_id(int val){
        gpu.device=val;
        set_device(gpu);
    }

    inline void copy_sym_info(np::ndarray& rot_mats){
        sym_ops_to_dev(gpu, rot_mats);
        has_sym_ops = true;
    }

    inline void symmetrize(np::ndarray& q_vecs){
        symmetrize_density(gpu, q_vecs);
    }

    inline void alloc(int device_id, np::ndarray& rotations, int maxNumQ,
                      bp::tuple corner, bp::tuple delta, np::ndarray& qvecs, int maxNumRotInds,
                      int numDataPix, bool use_IPC){
        int num_rot=rotations.shape(0)/9;
        gpu.device = device_id;
        gpu.numDataPixels = numDataPix;
        gpu.maxNumQ = maxNumQ;
        gpu.maxNumRotInds = maxNumRotInds;
        gpu.corner[0] = bp::extract<CUDAREAL>(corner[0]);
        gpu.corner[1] = bp::extract<CUDAREAL>(corner[1]);
        gpu.corner[2] = bp::extract<CUDAREAL>(corner[2]);

        gpu.delta[0] = bp::extract<CUDAREAL>(delta[0]);
        gpu.delta[1] = bp::extract<CUDAREAL>(delta[1]);
        gpu.delta[2] = bp::extract<CUDAREAL>(delta[2]);
        prepare_for_lerping( gpu, rotations, qvecs, use_IPC);
    }

    inline int get_max_num_rots(){
        return gpu.maxNumRotInds;
    }

    inline bool get_dev_is_allocated(){
        return gpu.is_allocated;
    }

    inline void copy_image_data( np::ndarray& pixels, np::ndarray& mask, np::ndarray& bg){
        // assert len pixels matches up
        if (pixels.shape(0) != gpu.numQ){
            PyErr_SetString(PyExc_TypeError, "Number of pixels passed does not agree with number of allocated pixels on device\n");
            bp::throw_error_already_set();
        }
        else if (mask.shape(0) != gpu.numQ){
            PyErr_SetString(PyExc_TypeError, "Number of mask flags passed does not agree with number of allocated pixels on device\n");
            bp::throw_error_already_set();
        }
        else if (bg.shape(0) != gpu.numQ){
            PyErr_SetString(PyExc_TypeError, "Number of background pixels passed does not agree with number of allocated pixels on device\n");
            bp::throw_error_already_set();
        }
        else{
            copy_image_data_to_device(gpu,pixels,mask,bg);
        }
    }

    inline void copy_relp_mask( np::ndarray& relp_mask){
        relp_mask_to_device(gpu, relp_mask);
    }

    inline void copy_densities( np::ndarray& new_dens, bool dens_is_reparam){
        // assert len pixels matches up
        if (new_dens.shape(0) != gpu.numDens){
            PyErr_SetString(PyExc_TypeError, "Number of densities passed does not agree with number of allocated densities on device\n");
            bp::throw_error_already_set();
        }
        else{
            densities_to_device(gpu,new_dens);
        }
        if (dens_is_reparam) {
            convert_reparameterized_densities(gpu);
        }
    }

    inline np::ndarray trilinear_interpolation(int rot_idx, bool verbose){
        if (rot_idx < 0 || rot_idx >= gpu.numRot) {
            PyErr_SetString(PyExc_TypeError,
                            "Rot index is out of bounds, check size of allocated rotMats\n");
            bp::throw_error_already_set();
        }
        std::vector<int> rot_inds;
        rot_inds.push_back(rot_idx);

        // 0 specifies only do interpolation
        do_a_lerp(gpu, rot_inds, verbose, 0);

        bp::tuple shape = bp::make_tuple(gpu.maxNumQ);
        bp::tuple stride = bp::make_tuple(sizeof(CUDAREAL));
        np::dtype dt = np::dtype::get_builtin<CUDAREAL>();
        np::ndarray output = np::from_data(&gpu.out[0], dt, shape, stride, bp::object());
        return output.copy();
    }
    
    inline void trilinear_insertion(int rot_idx, bool verbose, CUDAREAL tomo_wt){
        if (rot_idx < 0 || rot_idx >= gpu.numRot) {
            PyErr_SetString(PyExc_TypeError,
                            "Rot index is out of bounds, check size of allocated rotMats\n");
            bp::throw_error_already_set();
        }
        std::vector<int> rot_inds;
        rot_inds.push_back(rot_idx);
           
        // 2 specifies to do a trilinear insertion
        gpu.tomogram_wt = tomo_wt;
        do_a_lerp(gpu, rot_inds, verbose, 2);

    }

    inline void check_densities_are_set(){
        if (gpu.densities==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "densities has not been allocated\n");
            bp::throw_error_already_set();
        }
    }

    inline MPI_Comm mpi_comm_from_py_obj(bp::object py_comm){
        PyObject* py_obj = py_comm.ptr();
        MPI_Comm comm = *PyMPIComm_Get(py_obj);
        return comm;
    }

    bool inline _is_in_sparse_mode(){
        return gpu.sparse_lookup != NULL;
    }

    inline void mpi_set_starting_density(np::ndarray & dens_start, bp::object py_comm){
        MPI_Comm comm = mpi_comm_from_py_obj(py_comm);
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank==0){
            CUDAREAL* dens_ptr = reinterpret_cast<CUDAREAL*>(dens_start.get_data());
            to_dev_memcpy(gpu.densities, dens_ptr, gpu.numDens);
        }
        _bcast_in_chunks<CUDAREAL>(comm, gpu.densities, gpu.numDens, MPI_CUDAREAL);
    }

    inline void _update_masked_density(np::ndarray& new_vals){
    //  TODO assert is_peak_in_density is set, and densities is allocated
        check_densities_are_set();
        update_masked_density_gpu(gpu, new_vals);
    }

    inline void _bcast_density(bp::object py_comm){
        MPI_Comm comm = mpi_comm_from_py_obj(py_comm);
        _bcast_in_chunks<CUDAREAL>(comm, gpu.densities, gpu.numDens, MPI_CUDAREAL);
    }

    inline void _bcast_weights(bp::object py_comm){
        MPI_Comm comm = mpi_comm_from_py_obj(py_comm);
        _bcast_in_chunks<CUDAREAL>(comm, gpu.wts, gpu.numDens, MPI_CUDAREAL);
    }

    inline void _bcast_density_grad(bp::object py_comm){
        MPI_Comm comm = mpi_comm_from_py_obj(py_comm);
        _bcast_in_chunks<CUDAREAL>(comm, gpu.densities_gradient, gpu.numDens, MPI_CUDAREAL);
    }

    inline void _bcast_relp_mask(bp::object py_comm){
    //  TODO use IPC for is_peak_in_density
        if (gpu.is_peak_in_density == NULL)
            malloc_relp_mask(gpu);
        MPI_Comm comm = mpi_comm_from_py_obj(py_comm);
        MPI_Bcast(&gpu.num_unmasked, 1, MPI_INT, 0, comm);
        if (gpu.unmasked_inds==NULL)
            malloc_unmasked_inds(gpu);
        _bcast_in_chunks<bool>(comm, gpu.is_peak_in_density, gpu.numDens, MPI_C_BOOL);
        _bcast_in_chunks<int>(comm, gpu.unmasked_inds, gpu.num_unmasked, MPI_INT);
    }

    template <typename vec_t> inline void _bcast_in_chunks(MPI_Comm comm, vec_t *vec, int N, MPI_Datatype dt){
        int rank;
        MPI_Comm_rank(comm, &rank);
        int sz=16*1024*1024; // TODO: make variable chunk size
        int start=0;
        while (start < N){
            int count = sz;
            if (start + count >=N ){
                count = N-start;
            }
            MPI_Bcast(&vec[start], count, dt, 0, comm);
            start += count;
        }
    }

    inline void _set_sparse_lookup(np::ndarray& peak_mask){
        set_sparse_lookup(gpu, peak_mask);
    }

    inline void _reduce_densities(bp::object py_comm){
        check_densities_are_set();
        _reduce_in_chunks(py_comm, gpu.densities, gpu.numDens);
    }

    inline void _reduce_weights(bp::object py_comm){
        // TODO verify gpu.wts is not NULL
        _reduce_in_chunks(py_comm, gpu.wts, gpu.numDens);
    }

    inline void _reduce_density_derivs(bp::object py_comm){
        // TODO verify gpu.densities_gradient is not NULL
        _reduce_in_chunks(py_comm, gpu.densities_gradient, gpu.numDens);
    }

    inline void _allreduce_density_derivs(bp::object py_comm){
        // TODO verify gpu.densities_gradient is not NULL
        _reduce_in_chunks(py_comm, gpu.densities_gradient, gpu.numDens, true);
    }


    inline void _reduce_in_chunks(bp::object py_comm, CUDAREAL* vec, int N, bool all=false){

        PyObject* py_obj = py_comm.ptr();
        MPI_Comm *comm_p = PyMPIComm_Get(py_obj);
        int rank;
        int sz=16*1024*1024;
        MPI_Comm_rank(*comm_p, &rank);
        std::vector<CUDAREAL> temp;
        if (!all && rank==0)
            temp.resize(N);
        if (all)
            temp.resize(N);
        int start=0;
        while (start < N){
            int count = sz;
            if (start + count >=N ){
                count = N-start;
            }
            if(all)
                MPI_Allreduce(&vec[start], temp.data()+start, count, MPI_CUDAREAL, MPI_SUM, *comm_p);
            else
                MPI_Reduce(&vec[start], temp.data()+start, count, MPI_CUDAREAL, MPI_SUM, 0, *comm_p);
            start += count;
        }
        if (!all && rank==0){
            CUDAREAL* temp_ptr = &temp[0];
            to_dev_memcpy(vec, temp_ptr, N );
        }
        if (all){
            CUDAREAL* temp_ptr = &temp[0];
            to_dev_memcpy(vec, temp_ptr, N );
        }
    }

    inline  np::ndarray get_sparse_lookup(){
        if (gpu.sparse_lookup==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "sparse_lookup not been allocated\n");
            bp::throw_error_already_set();
        }
        int N = gpu.densDim*gpu.densDim*gpu.densDim;
        bp::tuple shape = bp::make_tuple(N);
        bp::tuple stride = bp::make_tuple(sizeof(int));
        np::dtype dt = np::dtype::get_builtin<int>();
        np::ndarray output = np::zeros(shape,dt);
        int* out_ptr = reinterpret_cast<int*>(output.get_data());
        from_dev_memcpy_int(gpu.sparse_lookup, out_ptr, N);
        return output;
    }

    inline  np::ndarray get_densities(){
        check_densities_are_set();
        if (gpu.densities==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "densities has not been allocated\n");
            bp::throw_error_already_set();
        }
        return dev_to_ndarray(gpu.densities, gpu.numDens);
    }

    inline np::ndarray get_reparameterized_densities_gradient(){
        if (gpu.densities_gradient==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "densities_gradient has not been allocated\n");
            bp::throw_error_already_set();
        }
        reparameterize_density_gradients(gpu);
        return dev_to_ndarray(gpu.densities_gradient, gpu.numDens);
    }

    inline  np::ndarray get_densities_gradient(){
        if (gpu.densities_gradient==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "densities_gradient has not been allocated\n");
            bp::throw_error_already_set();
        }
        return dev_to_ndarray(gpu.densities_gradient, gpu.numDens);
    }

    inline np::ndarray get_wts(){
        if (gpu.wts==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "wts has not been allocated\n");
            bp::throw_error_already_set();
        }
        return dev_to_ndarray(gpu.wts, gpu.numDens);
    }

    inline np::ndarray dev_to_ndarray(CUDAREAL* dev_ptr, int N){
        //CUDAREAL* temp = new CUDAREAL[N];
        //from_dev_memcpy(dev_ptr, temp, N);
        bp::tuple shape = bp::make_tuple(N);
        bp::tuple stride = bp::make_tuple(sizeof(CUDAREAL));
        np::dtype dt = np::dtype::get_builtin<CUDAREAL>();
        np::ndarray output = np::zeros(shape,dt);
        CUDAREAL* out_ptr = reinterpret_cast<CUDAREAL*>(output.get_data());
        //np::ndarray output = np::from_data(temp, dt, shape, stride, bp::object()).copy();
        from_dev_memcpy(dev_ptr, out_ptr, N);
        //delete temp;
        return output;
    }

    inline void do_equation_two(np::ndarray rot_idx, bool verbose, CUDAREAL shot_scale, const int deriv){
        int nrot = rot_idx.shape(0);
        std::vector<int> rot_inds;
        for (int i_rot=0; i_rot < nrot; i_rot++)
            rot_inds.push_back(  bp::extract<int>(rot_idx[i_rot])  );
        gpu.shot_scale = shot_scale;
        if (deriv==1 || deriv==2)
            if (deriv==1) // scale factor derivative (task 3)
                do_a_lerp(gpu, rot_inds, verbose, 3);
            else // density derivative (task 4)
                do_a_lerp(gpu, rot_inds, verbose, 4);
        else
            // run through EMC equation two (from the dragon fly paper) for the  specified rotation inds (task 1)
            do_a_lerp(gpu, rot_inds, verbose, 1);

    }

    inline void _reset_dens_deriv(){
        reset_dens_deriv(gpu);
    }

    inline void do_dens_deriv(np::ndarray rot_idx, np::ndarray Pdr_vals, bool verbose, CUDAREAL shot_scale, bool reset_derivs){
        bool temp = gpu.alwaysResetDeriv;
        gpu.alwaysResetDeriv = reset_derivs;

        int nrot = rot_idx.shape(0);
        std::vector<int> rot_inds;

        gpu.Pdr_host.clear();
        for (int i_rot=0; i_rot < nrot; i_rot++) {
            rot_inds.push_back(bp::extract<int>(rot_idx[i_rot]));
            gpu.Pdr_host.push_back( bp::extract<CUDAREAL>(Pdr_vals[i_rot]) );
        }
        gpu.shot_scale = shot_scale;
        do_a_lerp(gpu, rot_inds, verbose, 5);
        gpu.alwaysResetDeriv = temp;
    }

    inline void print_rotMat(int i_rot){
        MAT3 M = gpu.rotMats[i_rot];
        printf("Rotation matrix %d=\n%.7f %.7f %.7f\n%.7f %.7f %.7f\n%.7f %.7f %.7f\n",
               M(0,0), M(0,1), M(0,2),
               M(1,0), M(1,1), M(1,2),
               M(2,0), M(2,1), M(2,2));
    }
    inline bp::list get_out(){
        return gpu.outList;
    }
    inline void toggle_insert(){
        toggle_insert_mode(gpu);
    }

    inline int get_densDim(){
        return gpu.densDim;
    }

    inline void set_densDim(int densDim){
        gpu.densDim = densDim;
    }
    
    inline double get_maxQ(){
        return gpu.maxQ;
    }
    
    inline void set_maxQ(double maxQ){
        gpu.maxQ = maxQ;
    }

    inline double get_rank(){
        return gpu.mpi_rank;
    }

    inline void set_rank(int rank){
        gpu.mpi_rank = rank;
    }


    inline void free(){
        free_lerpy(gpu);
    }
};


class probaOr{
public:
    virtual ~probaOr(){}
    // constructor
    probaOr(){}
    gpuOrient gpu;
    int size_of_cudareal = sizeof(CUDAREAL);
    bool auto_convert_arrays = true;
    inline void set_B(bp::tuple vals){
        CUDAREAL bxx = bp::extract<CUDAREAL>(vals[0]);
        CUDAREAL bxy = bp::extract<CUDAREAL>(vals[1]);
        CUDAREAL bxz = bp::extract<CUDAREAL>(vals[2]);
        CUDAREAL byx = bp::extract<CUDAREAL>(vals[3]);
        CUDAREAL byy = bp::extract<CUDAREAL>(vals[4]);
        CUDAREAL byz = bp::extract<CUDAREAL>(vals[5]);
        CUDAREAL bzx = bp::extract<CUDAREAL>(vals[6]);
        CUDAREAL bzy = bp::extract<CUDAREAL>(vals[7]);
        CUDAREAL bzz = bp::extract<CUDAREAL>(vals[8]);
        gpu.Bmat << bxx, bxy, bxz,
                byx, byy, byz,
                bzx, bzy, bzz;
    }
    inline bp::tuple get_B(){
        bp::tuple B = bp::make_tuple(
                gpu.Bmat(0,0), gpu.Bmat(0,1), gpu.Bmat(0,2),
                gpu.Bmat(1,0), gpu.Bmat(1,1), gpu.Bmat(1,2),
                gpu.Bmat(2,0), gpu.Bmat(2,1), gpu.Bmat(2,2)
                );
        return B;
    }

    inline void alloc(int device_id, np::ndarray& rotations, int maxQvecs){
        int num_rot=rotations.shape(0)/9;
        setup_orientMatch( device_id, maxQvecs, gpu, rotations, true);
    }
    inline void free(){
        free_orientMatch(gpu);
    }
    inline np::ndarray oriPeaks(np::ndarray& qvecs,
                         float hcut, int minWithinHcut, bool verbose){
        orientPeaks(gpu, qvecs, hcut, minWithinHcut, verbose);

        bp::tuple shape = bp::make_tuple(gpu.numRot);
        bp::tuple stride = bp::make_tuple(sizeof(bool));
        np::dtype dt = np::dtype::get_builtin<bool>();
        np::ndarray output = np::from_data(&gpu.out[0], dt, shape, stride, bp::object());
        return output.copy();
    }

    inline bp::list listOrients(){
        return gpu.probable_rot_inds;
    }

    inline void print_rotMat(int i_rot){
        MAT3 M = gpu.rotMats[i_rot];
        printf("Rotation matrix %d=\n%.7f %.7f %.7f\n%.7f %.7f %.7f\n%.7f %.7f %.7f\n",
               M(0,0), M(0,1), M(0,2),
               M(1,0), M(1,1), M(1,2),
               M(2,0), M(2,1), M(2,2));
    }

    inline void alloc_IPC(int device_id, np::ndarray& rotations,
                        int maxQvecs, int numRot, bp::object py_comm){

      PyObject* py_obj = py_comm.ptr();
      MPI_Comm *comm_p = PyMPIComm_Get(py_obj);
      if (comm_p == NULL) bp::throw_error_already_set();
      setup_orientMatch_IPC(device_id, maxQvecs, gpu,
                     rotations, numRot, *comm_p);
    }

};


BOOST_PYTHON_MODULE(emc){
//  important initialization
    Py_Initialize();
    np::initialize();
    if (import_mpi4py() < 0) return;

    typedef bp::return_value_policy<bp::return_by_value> rbv;
    typedef bp::default_call_policies dcp;
    typedef bp::return_internal_reference<> rir;

    /**********************************************************************************************/
    /* Lerpy class (main kernels used in EMC, lots of linear interpolation, hence the name lerpy) */
    /**********************************************************************************************/
    bp::class_<lerpyExt>("lerpy", bp::no_init)
        .def(bp::init<>("returns a class instance"))
        .def ("_allocate_lerpy", &lerpyExt::alloc, "allocate the device")
        .def ("_copy_image_data", &lerpyExt::copy_image_data, "copy pixels to the GPU device")
        .def ("_copy_relp_mask", &lerpyExt::copy_relp_mask, "copy relp mask to the GPU device")
        .def ("_update_density", &lerpyExt::copy_densities, "copies new density to the GPU device")
        //.def("free_device", &lerpyExt::free, "free any allocated GPU memory")
        .def ("print_rotMat", &lerpyExt::print_rotMat, "show elements of allocated rotMat i_rot")
        .def ("get_out", &lerpyExt::get_out, "return the output array.")

        .def ("toggle_insert", &lerpyExt::toggle_insert, "Prepare for trilinear insertions.")

        .def("_trilinear_interpolation",
             &lerpyExt::trilinear_interpolation,
             (bp::arg("rot_idx"), bp::arg("verbose")=true),
             "interpolate the qvecs according to the supplied densities")
        
        .def("_trilinear_insertion",
             &lerpyExt::trilinear_insertion,
             (bp::arg("rot_idx"), bp::arg("verbose")=true, bp::arg("tomo_wt")=1),
             "insert the vals according into the densities")

        .def("_equation_two",
             &lerpyExt::do_equation_two,
             (bp::arg("rot_idx"), bp::arg("verbose")=true, bp::arg("shot_scale")=1,
                     bp::arg("deriv")=false),
             "compute equation to for the supplied rotation indices")
        .def("_dens_deriv",
            &lerpyExt::do_dens_deriv,
            (bp::arg("rot_idx"), bp::arg("Pdr"),
                 bp::arg("verbose")=true, bp::arg("shot_scale")=1),
            "derivative of log likeihood w.r.t. densities")
        .add_property("auto_convert_arrays",
                       make_getter(&lerpyExt::auto_convert_arrays,rbv()),
                       make_setter(&lerpyExt::auto_convert_arrays,dcp()),
                       "If arrays passed to `copy_image_data` or `update_density` aren't suitable, convert them to suitable arrays. A suitable array is C-contiguous and of type CUDAREAL")
        .add_property("size_of_cudareal",
                       make_getter(&lerpyExt::size_of_cudareal,rbv()),
                       "CUDAREAL is this many bytes")
        .add_property("has_sym_ops",
            make_getter(&lerpyExt::has_sym_ops,rbv()),
            "Whether the sym ops were set")
        .def("densities",
              &lerpyExt::get_densities,
              "get the densities")
        .def("sparse_lookup",
              &lerpyExt::get_sparse_lookup,
              "get the sparse lookup table (ndarray 1-D , size=number of voxels, -1 means masked voxel, other values are position of voxel in sparse density vector)")

        .def("reduce_densities",
              &lerpyExt::_reduce_densities,
              "reduce the densities using MPI")

        .def("reduce_weights",
              &lerpyExt::_reduce_weights,
              "reduce the densities weights using MPI")

        .def("densities_gradient",
              &lerpyExt::get_densities_gradient,
              "get the gradient of the logLikelikhood w.r.t. the densities (this just points to the data , one should run equation_two with deriv=2 prior to calling this method, otherwise densities will be meaningless")

        .def("reparameterized_densities_gradient",
              &lerpyExt::get_reparameterized_densities_gradient,
              "get the reparameterized version of gradient of the logLikelikhood w.r.t. the densities (this just points to the data , one should run equation_two with deriv=2 prior to calling this method, otherwise densities will be meaningless")
        .def("wts",
              &lerpyExt::get_wts,
              "get the density weights")
        
        .def("_copy_sym_info",
              &lerpyExt::copy_sym_info,
              "Copy symmetry operators to the GPU")
        .def("_symmetrize",
              &lerpyExt::symmetrize,
              "Symmetrize the density thats on the GPU (be sure to call _copy_sym_info first)")

        .def("free", &lerpyExt::free, "free the gpu")
        
        .add_property("dens_dim",
                       make_function(&lerpyExt::get_densDim,rbv()),
                       make_function(&lerpyExt::set_densDim,dcp()),
                       "the number of bins along the density edge (its always a cube); default=256")

        .add_property("is_in_sparse_mode",
                       make_function(&lerpyExt::_is_in_sparse_mode,rbv()),
                       "whether sparse density mode is being used")

        .add_property("max_q",
                       make_function(&lerpyExt::get_maxQ,rbv()),
                       make_function(&lerpyExt::set_maxQ,dcp()),
                       "the maximum q magnitude (defines density edge length from -maxQ to +maxQ)")

        .add_property("rank",
                       make_function(&lerpyExt::get_rank,rbv()),
                       make_function(&lerpyExt::set_rank,dcp()),
                       "set the mpi rank from python (its used in various printf statements)")

        .add_property("max_num_rots",
                       make_function(&lerpyExt::get_max_num_rots,rbv()),
                       "GPU was allocated for this many rotations")

        .add_property("dev_is_allocated",
                       make_function(&lerpyExt::get_dev_is_allocated,rbv()),
                       "return True if GPU arrays are allocated")

        .def("get_gpu_mem", &lerpyExt::_get_gpu_mem, "get free GPU memory in bytes (for dev_id that was used in allocate_lerpy)")

        .add_property("dev_id",
                       make_function(&lerpyExt::get_dev_id,rbv()),
                       make_function(&lerpyExt::set_dev_id,dcp()),
                       "the GPU device ID used for running the emc kernels")

        .def("_mpi_set_starting_density", &lerpyExt::mpi_set_starting_density, "set the starting density, broadcast to other ranks")

        .def("bcast_densities", &lerpyExt::_bcast_density, "broadcast density from root to other ranks")
        .def("bcast_wts", &lerpyExt::_bcast_weights, "broadcast weights from root to other ranks")
        .def("bcast_density_derivs", &lerpyExt::_bcast_density_grad, "broadcast dens gradients from root to other ranks")
        .def("bcast_relp_mask", &lerpyExt::_bcast_relp_mask, "broadcast relp mask from root to other ranks")
        .def("reset_density_derivs", &lerpyExt::_reset_dens_deriv, "reset the densities_gradient array to zeros")
        .def("reduce_density_derivs", &lerpyExt::_reduce_density_derivs, "reduce the densities_gradient (set on rank=0)")
        .def("allreduce_density_derivs", &lerpyExt::_allreduce_density_derivs, "Allreduce the densities_gradient (set on rank=0)")
        .def("update_masked_density", &lerpyExt::_update_masked_density, "receives np::ndarray V and updates masked density. Length of V is number of non masked voxels")
        .def ("set_sparse_lookup", &lerpyExt::_set_sparse_lookup, "takes a voxel mask, and creates a sparse vector framework for saving GPU memory (use carefully!)")
        ;

    /******************************/
    /* Orientation matching class */
    /******************************/
    bp::class_<probaOr>("probable_orients", bp::no_init)
        .def(bp::init<>("returns a class instance"))
        .def ("_allocate_orientations", &probaOr::alloc, "move the orientations to the device")
        .def ("_orient_peaks", &probaOr::oriPeaks, "compute probable orientations (main CUDA kernel)")
        .def("free_device", &probaOr::free, "free any allocated GPU memory")
        .def ("print_rotMat", &probaOr::print_rotMat, "show elements of allocated rotMat i_rot")
        .def ("get_probable_orients", &probaOr::listOrients, "returns a list of rotation matrix indices")
        .add_property("Bmatrix",
                       make_function(&probaOr::get_B,rbv()),
                       make_function(&probaOr::set_B,dcp()),
                       "the Bmatrix (dxtbx Crystal.get_B() format)")
        .add_property("size_of_cudareal",
            make_getter(&probaOr::size_of_cudareal,rbv()),
            "CUDAREAL is this many bytes")
        .add_property("auto_convert_arrays",
            make_getter(&probaOr::auto_convert_arrays,rbv()),
            make_setter(&probaOr::auto_convert_arrays,dcp()),
            "If arrays passed to `copy_image_data` or `update_density` aren't suitable, convert them to suitable arrays. A suitable array is C-contiguous and of type CUDAREAL")
        .def ("_allocate_orientations_IPC", &probaOr::alloc_IPC, "allocate the device using inter process comm")
        ;

}
