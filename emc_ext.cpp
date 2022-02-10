
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
    inline void alloc(int device_id, np::ndarray rotations, np::ndarray densities, int maxNumQ,
                      bp::tuple corner, bp::tuple delta, np::ndarray qvecs, int maxNumRotInds,
                      int numDataPix){
        int num_rot=rotations.shape(0)/9;
        gpu.device = device_id;
        gpu.numDataPixels = numDataPix;
        gpu.maxNumQ = maxNumQ;
        gpu.maxNumRotInds = maxNumRotInds;
        printf("Determined number of rotations=%d\n", num_rot);
        gpu.corner[0] = bp::extract<double>(corner[0]);
        gpu.corner[1] = bp::extract<double>(corner[1]);
        gpu.corner[2] = bp::extract<double>(corner[2]);

        gpu.delta[0] = bp::extract<double>(delta[0]);
        gpu.delta[1] = bp::extract<double>(delta[1]);
        gpu.delta[2] = bp::extract<double>(delta[2]);
        prepare_for_lerping( gpu, rotations, densities, qvecs);
    }
    //inline void trilinear_interpolation(np::ndarray qvecs, bool verbose){
    inline int copy_pixels( np::ndarray pixels){
        // assert len pixels matches up
        if (pixels.shape(0) != gpu.numQ){
            printf("Number of pixels passed does not agree with number of allocated pixels on device\n");
            exit(1);
        }
        else{
            shot_data_to_device(gpu,pixels); 
            return 0;
        }
    }
    inline void trilinear_interpolation(np::ndarray rot_idx, bool verbose){
        int nrot = rot_idx.shape(0);
        std::vector<int> rot_inds;

        for (int i_rot=0; i_rot < nrot; i_rot++)
            rot_inds.push_back(  bp::extract<int>(rot_idx[i_rot])  );

        // 0 specifies only do interpolation
        do_a_lerp(gpu, rot_inds, verbose, 0);
    }
    
    inline void do_equation_two(np::ndarray rot_idx, bool verbose){
        int nrot = rot_idx.shape(0);
        std::vector<int> rot_inds;

        for (int i_rot=0; i_rot < nrot; i_rot++)
            rot_inds.push_back(  bp::extract<int>(rot_idx[i_rot])  );

        // 1 specifies to run through equation two for the  specified rotation inds
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
};


class probaOr{
public:
    virtual ~probaOr(){}
    // constructor
    probaOr(){}
    gpuOrient gpu;
    inline void alloc(int device_id, np::ndarray rotations, int maxQvecs){
        int num_rot=rotations.shape(0)/9;
        printf("Determined number of rotations=%d\n", num_rot);
        //printf("We will allocate space for %d orientations!\n", num_rot);
        setup_orientMatch( device_id, maxQvecs, gpu, rotations, true);
    }
    inline void free(){
        free_orientMatch(gpu);
    }
    inline void oriPeaks(np::ndarray qvecs,
                         float hcut, int minWithinHcut, bool verbose){
        orientPeaks(gpu, qvecs, hcut, minWithinHcut, verbose);
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
        .def ("allocate_lerpy", &lerpyExt::alloc, "allocate the device")
        .def ("copy_image_data", &lerpyExt::copy_pixels, "copy pixels to the GPU device")
        //.def("free_device", &lerpyExt::free, "free any allocated GPU memory")
        .def ("print_rotMat", &lerpyExt::print_rotMat, "show elements of allocated rotMat i_rot")
        .def ("get_out", &lerpyExt::get_out, "return the output array.")

        .def("trilinear_interpolation",
             &lerpyExt::trilinear_interpolation,
             (bp::arg("rot_idx"), bp::arg("verbose")=true),
             "interpolate the qvecs according to the supplied densities")

        .def("equation_two",
             &lerpyExt::do_equation_two,
             (bp::arg("rot_idx"), bp::arg("verbose")=true),
             "compute equation to for the supplied rotation indices")
        //.add_property("num_ori",
        //               make_getter(&lerpyExt::num_orient,rbv()),
        //               make_setter(&probaOr::num_orient,dcp()),
        //               "Number of orientations.")
        ;

    /* Orientation matching class */
    bp::class_<probaOr>("probable_orients", bp::no_init)
        .def(bp::init<>("returns a class instance"))
        .def ("allocate_orientations", &probaOr::alloc, "move the orientations to the device")
        .def ("orient_peaks", &probaOr::oriPeaks, "compute probable orientations (main CUDA kernel)")
        .def("free_device", &probaOr::free, "free any allocated GPU memory")
        .def ("print_rotMat", &probaOr::print_rotMat, "show elements of allocated rotMat i_rot")
        .def ("get_probable_orients", &probaOr::listOrients, "returns a list of rotation matrix indices")
        ;

}
