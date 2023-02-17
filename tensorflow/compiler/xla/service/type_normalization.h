/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_NORMALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_NORMALIZATION_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/type_support.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which adds F32 <-> F8 conversions for HLO instructions that do not
// support F8 input/output or mixed precision, according to the passed-in
// backend-specific F8 support rules.
class TypeNormalization : public HloModulePass {
 public:
  explicit TypeNormalization(PrimitiveType source_type, PrimitiveType target_type, const TypeSupport* type_support)
      : source_type_(source_type), target_type_(target_type), type_support_(type_support) {}

  ~TypeNormalization() override = default;
  absl::string_view name() const override { return "type-normalization"; }

  // Run F8 normalization on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  PrimitiveType sourceType() const { return source_type_; }
  PrimitiveType targetType() const { return target_type_; }

 private:
  const PrimitiveType source_type_;
  const PrimitiveType target_type_;
  const TypeSupport* type_support_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_NORMALIZATION_H_
