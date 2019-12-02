/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "util.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
class BottomPoolOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *x = ctx.Input<Tensor>("X");
    auto *output = ctx.Output<Tensor>("Output");
    auto *x_data = x->data<T>();
    auto x_dims = x->dims();
    int height = x_dims[2];
    auto& dev_ctx = ctx.cuda_device_context();

    framework::DDim temp_dims(x_dims);
    T *output_data = output->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    platform::CUDAPlace gpu_place;
    
    memory::Copy(gpu_place, output_data, gpu_place, x_data,
                sizeof(T) * x->numel(), dev_ctx.stream());

    int threads = kNumCUDAThreads;
    std::vector<int> x_dims_v = framework::vectorize<int>(x_dims);
    auto x_dims_gpu_ptr = memory::Alloc(gpu_place, x_dims_v.size() * sizeof(int));
    int *x_dims_gpu_data = reinterpret_cast<int*>(x_dims_gpu_ptr->ptr());
    memory::Copy(gpu_place, x_dims_gpu_data, platform::CPUPlace(), x_dims_v.data(), 
                 sizeof(int) * x_dims_v.size(), dev_ctx.stream());
    for (int ind = 1; ind < height; ind <<= 1) {
      temp_dims[2] = height - ind;
      int cur_num = framework::product(temp_dims);
      int bytes = cur_num * sizeof(T);
      auto cur_ptr = memory::Alloc(gpu_place, bytes);
      auto next_ptr = memory::Alloc(gpu_place, bytes);
      T* cur_data = reinterpret_cast<T*>(cur_ptr->ptr());
      T* next_data = reinterpret_cast<T*>(next_ptr->ptr());
      int blocks = NumBlocks(cur_num);

      SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(output_data, x_dims_gpu_data, 2, ind, height, cur_data);
      SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(output_data, x_dims_gpu_data, 2, 0, temp_dims[2], next_data);
      dev_ctx.Wait();

      MaxOut<T><<<blocks, threads, 0, dev_ctx.stream()>>>(cur_data, next_data, x_dims_gpu_data, 2, ind, height, output_data);
      dev_ctx.Wait();
    }
  }
};

template <typename T>
class BottomPoolGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto* in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto x_dims = x->dims();
    
    auto& dev_ctx = ctx.cuda_device_context();
    T* in_grad_data = in_grad->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    platform::CUDAPlace gpu_place;
    
    int threads = kNumCUDAThreads;
    int height = x_dims[2];
    int grad_num = in_grad->numel();
    int grad_block = NumBlocks(grad_num);
    FillConstant<T><<<grad_block, threads, 0, dev_ctx.stream()>>>(in_grad_data, x->numel(), 0);
    std::vector<int> x_dims_v = framework::vectorize<int>(x_dims);
    auto x_dims_gpu_ptr = memory::Alloc(gpu_place, x_dims_v.size() * sizeof(int));
    int *x_dims_gpu_data = reinterpret_cast<int*>(x_dims_gpu_ptr->ptr());
    memory::Copy(gpu_place, x_dims_gpu_data, platform::CPUPlace(), x_dims_v.data(), 
                 sizeof(int) * x_dims_v.size(), dev_ctx.stream());

    framework::DDim temp_dims(x_dims);
    temp_dims[2] = 1;

    int num = framework::product(x_dims) / height;
    int blocks = NumBlocks(num);

    // inital the max_value by the first row of input(x) 
    auto max_val_ptr = memory::Alloc(gpu_place, num * sizeof(T));
    T* max_val_data = reinterpret_cast<T*>(max_val_ptr->ptr());
    SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(x->data<T>(), x_dims_gpu_data, 2, 0, 1, max_val_data);
    //dev_ctx.Wait();
    auto cur_val_ptr = memory::Alloc(gpu_place, num * sizeof(T));
    T* cur_val_data = reinterpret_cast<T*>(cur_val_ptr->ptr());

    // inital the max_ind by 0
    auto max_ind_ptr = memory::Alloc(gpu_place, num * sizeof(int));
    int* max_ind_data = reinterpret_cast<int*>(max_ind_ptr->ptr());
    FillConstant<int><<<blocks, threads, 0, dev_ctx.stream()>>>(max_ind_data, num, 0);

    auto grad_ptr = memory::Alloc(gpu_place, num * sizeof(T));
    T* grad_data = reinterpret_cast<T*>(grad_ptr->ptr());

    // accumulate gradient on the location with maximum value
    SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_grad->data<T>(), x_dims_gpu_data, 2, 0, 1, grad_data);
    ScatterAddOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(grad_data, max_ind_data, x_dims_gpu_data, 2, in_grad_data);

    for (int ind = 1; ind < height; ++ind) {
      SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(x->data<T>(), x_dims_gpu_data, 2, ind, ind+1, cur_val_data);
      //dev_ctx.Wait();
      SliceOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_grad->data<T>(), x_dims_gpu_data, 2, ind, ind+1, grad_data);
      //dev_ctx.Wait();
      
      UpdateMaxInfo<T><<<blocks, threads, 0, dev_ctx.stream()>>>(cur_val_data, num, ind, max_val_data, max_ind_data);

      //dev_ctx.Wait();
      ScatterAddOnAxis<T><<<blocks, threads, 0, dev_ctx.stream()>>>(grad_data, max_ind_data, x_dims_gpu_data, 2, in_grad_data); 
      //dev_ctx.Wait();
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bottom_pool,
                        ops::BottomPoolOpCUDAKernel<float>,
                        ops::BottomPoolOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(bottom_pool_grad,
                        ops::BottomPoolGradOpCUDAKernel<float>,
                        ops::BottomPoolGradOpCUDAKernel<double>);
