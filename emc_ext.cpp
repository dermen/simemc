
#include <Python.h>
// prevents an annoying compiler warning about auto_ptr deprecation:
#define BOOST_NO_AUTO_PTR
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/args.hpp>
#include <iostream>
#include "cuda_trilerp.h"
#define BOOST_LIB_NAME "boost_numpy"
#include <boost/config/auto_link.hpp>
namespace bp=boost::python;
namespace np=boost::python::numpy;


class lerpyExt{
    public:
    virtual ~lerpyExt(){}
    lerpyExt(){}
    lerpy gpu;
    bool auto_convert_arrays = true;
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
    inline void alloc(int device_id, np::ndarray rotations, np::ndarray densities, int maxNumQ,
                      bp::tuple corner, bp::tuple delta, np::ndarray qvecs, int maxNumRotInds,
                      int numDataPix){
        int num_rot=rotations.shape(0)/9;
        gpu.device = device_id;
        gpu.numDataPixels = numDataPix;
        gpu.maxNumQ = maxNumQ;
        gpu.maxNumRotInds = maxNumRotInds;
        gpu.corner[0] = bp::extract<double>(corner[0]);
        gpu.corner[1] = bp::extract<double>(corner[1]);
        gpu.corner[2] = bp::extract<double>(corner[2]);

        gpu.delta[0] = bp::extract<double>(delta[0]);
        gpu.delta[1] = bp::extract<double>(delta[1]);
        gpu.delta[2] = bp::extract<double>(delta[2]);
        prepare_for_lerping( gpu, rotations, densities, qvecs);
    }
    inline void copy_pixels( np::ndarray& pixels, np::ndarray& mask){
        // assert len pixels matches up
        if (pixels.shape(0) != gpu.numQ){
            PyErr_SetString(PyExc_TypeError, "Number of pixels passed does not agree with number of allocated pixels on device\n");
            bp::throw_error_already_set();
        }
        else if (mask.shape(0) != gpu.numQ){
            PyErr_SetString(PyExc_TypeError, "Number of mask flags passed does not agree with number of allocated pixels on device\n");
            bp::throw_error_already_set();
        }
        else{
            shot_data_and_mask_to_device(gpu,pixels,mask);
        }
    }

    inline void copy_densities( np::ndarray& new_dens){
        // assert len pixels matches up
        if (new_dens.shape(0) != gpu.numDens){
            PyErr_SetString(PyExc_TypeError, "Number of densities passed does not agree with number of allocated densities on device\n");
            bp::throw_error_already_set();
        }
        else{
            densities_to_device(gpu,new_dens);
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
    
    inline void trilinear_insertion(int rot_idx, np::ndarray vals, bool verbose, CUDAREAL tomo_wt){
        if (rot_idx < 0 || rot_idx >= gpu.numRot) {
            PyErr_SetString(PyExc_TypeError,
                            "Rot index is out of bounds, check size of allocated rotMats\n");
            bp::throw_error_already_set();
        }
        if (vals.shape(0) > gpu.numDataPixels) {
            PyErr_SetString(PyExc_TypeError,
                            "Number of passed values does not agree with number of allocated data pixels\n");
            bp::throw_error_already_set();
        }
        if (vals.shape(0) != gpu.numQ) {
            PyErr_SetString(PyExc_TypeError,
                            "For insertion the number of vals should be the same as the number of Qvecs on device\n");
            bp::throw_error_already_set();
        }

        std::vector<int> rot_inds;
        rot_inds.push_back(rot_idx);
           
        // copy the insertion values to the device 
        shot_data_to_device(gpu,vals);

        // 2 specifies to do a trilinear insertion
        gpu.tomogram_wt = tomo_wt;
        do_a_lerp(gpu, rot_inds, verbose, 2);

    }

    inline  np::ndarray get_densities(){
        if (gpu.densities==NULL){ 
            PyErr_SetString(PyExc_TypeError,
                            "densities has not been allocated\n");
            bp::throw_error_already_set();
        }
        bp::tuple shape = bp::make_tuple(gpu.numDens);
        bp::tuple stride = bp::make_tuple(sizeof(CUDAREAL));
        np::dtype dt = np::dtype::get_builtin<CUDAREAL>();
        np::ndarray output = np::from_data(&gpu.densities[0], dt, shape, stride, bp::object());
        return output.copy();
    }

    inline np::ndarray get_wts(){
        if (gpu.wts==NULL){
            PyErr_SetString(PyExc_TypeError,
                            "wts has not been allocated\n");
            bp::throw_error_already_set();
        }
        bp::tuple shape = bp::make_tuple(gpu.numDens);
        bp::tuple stride = bp::make_tuple(sizeof(CUDAREAL));
        np::dtype dt = np::dtype::get_builtin<CUDAREAL>();
        np::ndarray output = np::from_data(&gpu.wts[0], dt, shape, stride, bp::object());
        return output.copy();
    }
    
    inline void do_equation_two(np::ndarray rot_idx, bool verbose, CUDAREAL shot_scale, const bool deriv){
        int nrot = rot_idx.shape(0);
        std::vector<int> rot_inds;
        for (int i_rot=0; i_rot < nrot; i_rot++)
            rot_inds.push_back(  bp::extract<int>(rot_idx[i_rot])  );
        gpu.shot_scale = shot_scale;
        // 1 specifies to run through EMC equation two (from the dragon fly paper) for the  specified rotation inds
        if (deriv)
            do_a_lerp(gpu, rot_inds, verbose, 3);
        else
            do_a_lerp(gpu, rot_inds, verbose, 1);

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

    inline void alloc(int device_id, np::ndarray rotations, int maxQvecs){
        int num_rot=rotations.shape(0)/9;
        setup_orientMatch( device_id, maxQvecs, gpu, rotations, true);
    }
    inline void free(){
        free_orientMatch(gpu);
    }
    inline np::ndarray oriPeaks(np::ndarray qvecs,
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

};


BOOST_PYTHON_MODULE(emc){
    Py_Initialize();
    np::initialize();
    typedef bp::return_value_policy<bp::return_by_value> rbv;
    typedef bp::default_call_policies dcp;
    typedef bp::return_internal_reference<> rir;

    bp::class_<lerpyExt>("lerpy", bp::no_init)
        .def(bp::init<>("returns a class instance"))
        .def ("_allocate_lerpy", &lerpyExt::alloc, "allocate the device")
        .def ("_copy_image_data", &lerpyExt::copy_pixels, "copy pixels to the GPU device")
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
             (bp::arg("rot_idx"), bp::arg("vals"), bp::arg("verbose")=true, bp::arg("tomo_wt")=1),
             "insert the vals according into the densities")

        .def("_equation_two",
             &lerpyExt::do_equation_two,
             (bp::arg("rot_idx"), bp::arg("verbose")=true, bp::arg("shot_scale")=1,
                     bp::arg("deriv")=false),
             "compute equation to for the supplied rotation indices")
        .add_property("auto_convert_arrays",
                       make_getter(&lerpyExt::auto_convert_arrays,rbv()),
                       make_setter(&lerpyExt::auto_convert_arrays,dcp()),
                       "If arrays passed to `copy_image_data` or `update_density` aren't suitable, convert them to suitable arrays. A suitable array is C-contiguous and of type CUDAREAL")
        .add_property("size_of_cudareal",
                       make_getter(&lerpyExt::size_of_cudareal,rbv()),
                       "CUDAREAL is this many bytes")
        .def("densities",
              &lerpyExt::get_densities,
              "get the densities")
        .def("wts",
              &lerpyExt::get_wts,
              "get the density weights")

        .def("free", &lerpyExt::free, "free the gpu")
        ;

    /* Orientation matching class */
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
        ;

}
