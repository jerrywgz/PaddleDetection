include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

OPS='bottom_pool_op top_pool_op right_pool_op left_pool_op'
for op in ${OPS}
do
nvcc ${op}.cu -c -o ${op}.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O0 -g -DNVCC \
    -I ${include_dir}/third_party/install \
    -I ${include_dir}/third_party/install/gflags/include \
    -I ${include_dir}/third_party/install/glog/include \
    -I ${include_dir}/third_party/install/protobuf/include \
    -I ${include_dir}/third_party/install/xxhash/include \
    -I ${include_dir}/third_party/boost \
    -I ${include_dir}/third_party/eigen3 \
    -I ${include_dir}/third_party/threadpool/src/extern_threadpool \
    -I ${include_dir}/third_party/dlpack/include \
    -I /paddle/Paddle/build/third_party/install/mkldnn/include/ \
    -I ${include_dir}/third_party/ \
    -I ${include_dir}
done

g++ bottom_pool_op.cc bottom_pool_op.cu.o top_pool_op.cc top_pool_op.cu.o right_pool_op.cc right_pool_op.cu.o left_pool_op.cc left_pool_op.cu.o -o cornerpool_lib.so -shared -fPIC -DPADDLE_WITH_MKLDNN -std=c++11 -O0 -g \
  -I ${include_dir}/third_party/install/protobuf/include \
  -I ${include_dir}/third_party/install/glog/include \
  -I ${include_dir}/third_party/install/gflags/include \
  -I ${include_dir}/third_party/install/xxhash/include \
  -I ${include_dir}/third_party/install/zlib/include \
  -I ${include_dir}/third_party/boost \
  -I ${include_dir}/third_party/eigen3 \
  -I ${include_dir}/third_party/dlpack/include \
  -I /paddle/Paddle/build/third_party/install/mkldnn/include/ \
  -I ${include_dir}/third_party/ \
  -I ${include_dir} \
  -L ${lib_dir} \
  -L /usr/local/cuda/lib64 -lpaddle_framework -lcudart

rm *.cu.o
