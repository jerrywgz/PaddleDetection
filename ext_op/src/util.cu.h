/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/memory/memory.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

/*template <typename T>
void PrintGPUTensor(const platform::CUDADeviceContext &ctx, T* x, framework::DDim dims) {
    platform::CUDAPlace gpu;
    platform::CPUPlace cpu;
    int num = framework::product(dims);
    auto temp_cpu_ptr = memory::Alloc(cpu, num * sizeof(T));
    T* temp_cpu_data = reinterpret_cast<T*>(temp_cpu_ptr->ptr());
    memory::Copy(cpu, temp_cpu_data, gpu, x, sizeof(T) * num, ctx.stream());
    LOG(ERROR)<<"Print";
    for (int i = 0; i < num/5; ++i) {
      LOG(ERROR)<<"print data: "<<temp_cpu_data[i*5]<<" "<<temp_cpu_data[i*5+1]<<" "<<temp_cpu_data[i*5+2]<<" "<<temp_cpu_data[i*5+3]<<" "<<temp_cpu_data[i*5+4];;
    } 
}*/

template <typename T>
__global__ void FillConstant(T* x, int num, int fill_num) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    x[i] = static_cast<T>(fill_num);
  }
}

template <typename T>
__global__ void SliceOnAxis(const T* x, const int* x_dim, 
                   const int axis, const int start, const int end, 
                   T* output) {
  int NC_num = x_dim[0] * x_dim[1];
  int HW_num = x_dim[2] * x_dim[3];
  int length = axis == 2 ? x_dim[3] : x_dim[2];
  int sliced_len = end - start;
  int cur_HW_num = length * sliced_len;
  // slice input on H or W (axis is 2 or 3)
  CUDA_1D_KERNEL_LOOP(i, NC_num * cur_HW_num) {
    int NC_id = i / cur_HW_num;
    int HW_id = i % cur_HW_num;
    if (axis == 2){
      output[i] = x[NC_id * HW_num + start * x_dim[3] + HW_id];
    } else if (axis == 3) {
      int col = HW_id % sliced_len;
      int row = HW_id / sliced_len;
      output[i] = x[NC_id * HW_num + row * x_dim[3] + start + col];
    }
  } 
}


template <typename T>
__global__  void MaxOut(const int cur_ind, const int next_ind,
                        const int* x_dim, const int axis, 
                        const int start, const int end, T* output) {
  int NC_num = x_dim[0] * x_dim[1];
  int HW_num = x_dim[2] * x_dim[3];
  int length = axis == 2 ? x_dim[3] : x_dim[2]; 
  T cur = static_cast<T>(0.);
  T next = static_cast<T>(0.);
  T max_v = static_cast<T>(0.);
  int sliced_len = end - start;
  int cur_HW_num = length * sliced_len;
  // compare cur and next and assign max values to output
  CUDA_1D_KERNEL_LOOP(i, NC_num * cur_HW_num) {
    int NC_id = i / cur_HW_num;
    int HW_id = i % cur_HW_num;
    //T max_v = cur[i] > next[i] ? cur[i] : next[i];
    if (axis == 2){
      cur = output[NC_id * HW_num + cur_ind * x_dim[3] + HW_id];
      next = output[NC_id * HW_num + next_ind * x_dim[3] + HW_id];
      max_v = cur > next ? cur : next; 
      output[NC_id * HW_num + start * x_dim[3] + HW_id] = max_v;
    } else if (axis == 3) {
      int col = HW_id % sliced_len;
      int row = HW_id / sliced_len;
      cur = output[NC_id * HW_num + row * x_dim[3] + cur_ind + col];
      next = output[NC_id * HW_num + row * x_dim[3] + next_ind + col];
      max_v = cur > next ? cur : next;
      output[NC_id * HW_num + row * x_dim[3] + start + col] = max_v;
    }
  }
}

template <typename T>
__global__  void UpdateMaxInfo(const T* input, const int* dims, const int axis, const int index, T* max_val, int* max_ind) {
  int length = axis == 2 ? dims[3] : dims[2];
  int NC_num = dims[0] * dims[1];
  int HW_num = dims[2] * dims[3]; 
  T val = static_cast<T>(0.);
  CUDA_1D_KERNEL_LOOP(i, NC_num * length) {
    int NC_id = i / length;
    int length_id = i % length;
    if (axis == 2) {
      val = input[NC_id * HW_num + index * dims[3] + length_id];
    } else if (axis == 3) {
      val = input[NC_id * HW_num + length_id * dims[3] + index];
    }
    if (val > max_val[i]) {
      max_val[i] = val;
      max_ind[i] = index;
    }
  }
}

template <typename T>
__global__  void ScatterAddOnAxis(const T* input, const int start, const int* max_ind, const int* dims, const int axis, T* output) {
  int length = axis == 2 ? dims[3] : dims[2];
  int NC_num = dims[0] * dims[1];
  int HW_num = dims[2] * dims[3];
  CUDA_1D_KERNEL_LOOP(i, NC_num * length) { 
    int NC_id = i / length;
    int length_id = i % length;
    int id_ = max_ind[i];
    //T val_ = max_val[i];
    if (axis == 2) {
      output[NC_id * HW_num + id_ * dims[3] + length_id] += input[NC_id * HW_num + start * dims[3] + length_id];
    } else if (axis == 3) {
      output[NC_id * HW_num + length_id * dims[3] + id_] += input[NC_id * HW_num + length_id * dims[3] + start];
    }
  }
}

}  // namespace operators
}  // namespace paddle
