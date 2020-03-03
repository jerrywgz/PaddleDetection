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
    int NC_num = x_dims[0] * x_dims[1];
    int height = x_dims[2];
    int width = x_dims[3];
    auto& dev_ctx = ctx.cuda_device_context();

    T *output_data = output->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());

    auto input_val_ptr = memory::Alloc(gpu_place, x->numel() * sizeof(T));
    T* input_val_data = reinterpret_cast<T*>(input_val_ptr->ptr());
    memory::Copy(gpu_place, input_val_data, gpu_place, x_data,
                sizeof(T) * x->numel(), dev_ctx.stream());

    memory::Copy(gpu_place, output_data, gpu_place, x_data,
                sizeof(T) * x->numel(), dev_ctx.stream());

    int threads = kNumCUDAThreads;
    for (int ind = 1; ind < height; ind <<= 1) {
      int cur_num = NC_num * width * (height - ind);
      int blocks = NumBlocks(cur_num);
      MaxOut<T><<<blocks, threads>>>(input_val_data, 0, NC_num, height, width, 2, ind, height, output_data);
      memory::Copy(gpu_place, input_val_data, gpu_place, output_data,
                 sizeof(T) * x->numel(), dev_ctx.stream());
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
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());
    
    int threads = kNumCUDAThreads;
    int NC_num = x_dims[0] * x_dims[1];
    int height = x_dims[2];
    int width = x_dims[3];
    int grad_num = in_grad->numel();
    int grad_block = NumBlocks(grad_num);
    cudaMemset(in_grad_data, 0, grad_num*sizeof(T));

    int num = grad_num / height;
    int blocks = NumBlocks(num);

    auto max_val_ptr = memory::Alloc(gpu_place, num * sizeof(T));
    T* max_val_data = reinterpret_cast<T*>(max_val_ptr->ptr());

    auto max_ind_ptr = memory::Alloc(gpu_place, num * sizeof(int));
    int* max_ind_data = reinterpret_cast<int*>(max_ind_ptr->ptr());

    auto max_map_ptr = memory::Alloc(gpu_place, grad_num * sizeof(int));
    int* max_map_data = reinterpret_cast<int*>(max_map_ptr->ptr());
    FillConstant<int><<<blocks, threads, 0, dev_ctx.stream()>>>(max_map_data, grad_num, 0);

    GetMaxInfo<T><<<blocks, threads, 0, dev_ctx.stream()>>>(x->data<T>(), NC_num, height, width, 2, false, max_val_data, max_ind_data, max_map_data);

    ScatterAdd<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_grad->data<T>(), max_map_data, grad_num, in_grad_data);
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
