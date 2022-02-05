
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
                      bp::tuple corner, bp::tuple delta){
        int num_rot=rotations.shape(0)/9;
        gpu.device = device_id;
        gpu.maxNumQ = maxNumQ;
        printf("Determined number of rotations=%d\n", num_rot);
        prepare_for_lerping( gpu, rotations, densities, corner, delta);
    }
    inline void trilinear_interpolation(np::ndarray qvecs, bool verbose){
        do_a_lerp(gpu, qvecs, verbose);
    }
    //inline void free(){
    //    free_orientMatch(gpu);
    //}
    //inline void oriPeaks(np::ndarray qvecs,
    //float hcut, int minWithinHcut, bool verbose){
    //    orientPeaks(gpu, qvecs, hcut, minWithinHcut, verbose);
    //}

    //inline bp::list listOrients(){
    //    return gpu.probable_rot_inds;
    //}

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


BOOST_PYTHON_MODULE(trilerp){
    Py_Initialize();
    np::initialize();
    typedef bp::return_value_policy<bp::return_by_value> rbv;
    typedef bp::default_call_policies dcp;
    typedef bp::return_internal_reference<> rir;

    bp::class_<lerpyExt>("lerpy", bp::no_init)
        .def(bp::init<>("returns a class instance"))
        .def ("allocate_lerpy", &lerpyExt::alloc, "allocate the device")
        //.def("free_device", &lerpyExt::free, "free any allocated GPU memory")
        .def ("print_rotMat", &lerpyExt::print_rotMat, "show elements of allocated rotMat i_rot")
        .def ("get_out", &lerpyExt::get_out, "return the interpolated values")
        .def("trilinear_interpolation",
             &lerpyExt::trilinear_interpolation,
             (bp::arg("qvecs"), bp::arg("verbose")=true),
             "interpolate the qvecs according to the supplied densities")
        //.def ("get_probable_orients", &lerpyExt::listOrients, "returns a list of rotation matrix indices")
        //.add_property("num_ori",
        //               make_getter(&lerpyExt::num_orient,rbv()),
        //               make_setter(&probaOr::num_orient,dcp()),
        //               "Number of orientations.")
        ;
}