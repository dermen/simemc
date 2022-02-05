
export CPRE=/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalX/conda_base
export BOOST_LOCALE_HIDE_AUTO_PTR=1
nvcc -c cuda_trilerp.cu -I$CPRE/include -I$CPRE/include/python3.8 -I$MODZ/eigen  -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -l python3.8 -lboost_python38 -lboost_system -lboost_numpy38  --compiler-options=-lstdc++,-fPIC,-O3 -o cuda_trilerp.o --expt-relaxed-constexpr

nvcc -c trilerp_ext.cpp -I$CPRE/include -I$CPRE/include/python3.8  -I$MODZ/eigen -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8 -lboost_python38 -lboost_system  -lboost_numpy38 --compiler-options=-lstdc++,-fPIC,-O3 -o trilerp.o --expt-relaxed-constexpr

g++ -shared trilerp.o cuda_trilerp.o -L$CPRE/lib -L$CPRE/lib/python3.8/config-3.8-x86_64-linux-gnu/ -L$CUDA_HOME/lib64 -lpython3.8 -lboost_python38  -lboost_numpy38 -lcudart -o trilerp.so
