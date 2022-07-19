/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_

#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(ThunkInfo thunk_info, cublas_lt::MatmulPlan plan,
                      int64_t algorithm_idx,
                      const BufferAllocation::Slice& a_buffer,
                      const BufferAllocation::Slice& b_buffer,
                      const BufferAllocation::Slice& c_buffer,
                      const BufferAllocation::Slice& d_buffer);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  cublas_lt::MatmulPlan plan_;
  int64_t algorithm_idx_;
  const BufferAllocation::Slice a_buffer_;
  const BufferAllocation::Slice b_buffer_;
  const BufferAllocation::Slice c_buffer_;
  const BufferAllocation::Slice d_buffer_;
  std::optional<se::cuda::BlasLt::MatmulAlgorithm> algorithm_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_