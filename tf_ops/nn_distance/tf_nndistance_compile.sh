#tf_packages_path=/usr/local/lib/python2.7/dist-packages
#tf_packages_path=/usr/local/lib/python3.5/dist-packages
tf_packages_path=/opt/conda/envs/pointnetae/lib/python3.6/site-packages
cuda_path=/usr/local/cuda

/usr/local/cuda/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ${tf_packages_path}/tensorflow/include -I ${cuda_path}/include -I ${tf_packages_path}/tensorflow/include/external/nsync/public -lcudart -L ${cuda_path}/lib64/ -L${tf_packages_path}/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

