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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_SUPPORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_SUPPORT_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {

class TypeSupport {
 public:
  TypeSupport(PrimitiveType source_type, PrimitiveType target_type)
    : source_type_(source_type), target_type_(target_type) {}
  virtual ~TypeSupport() = default;

  // Returns whether the backend supports `source_type` operand for the HLO
  // instruction at the given index.
  virtual bool SupportsOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const;

  // Returns whether the backend supports `source_type` output for the HLO
  // instruction.
  virtual bool SupportsOutput(const HloInstruction& hlo) const;

  // Returns whether the backend support mixed precision: the operands, output,
  // and parameters/output of the called computations can have different
  // precisions (`source_type` and `target_type`).
  virtual bool SupportsMixedPrecisions(const HloInstruction& hlo) const;

  // Returns whether the given HLO preserves its `source_type` operand
  // precision at the given index, so even if the output is `target_type`,
  // elements in the output that depend on the `source_type` operand will still
  // have `source_type` effective precision even if they have `target_type`
  // format. Similarly, this also means if the output is `source_type` then
  // increasing the operand precision from `source_type` to `target_type` will
  // not change the output. This typically includes HLOs that pass elements
  // from the operand to the output without arithmetic operations.
  static bool EffectiveOperandPrecisionIsOutputPrecision(
      const HloInstruction& hlo, int64_t operand_index);

  // Returns if the backend only uses `source_type` precision for the operand
  // at the specified index, even if the operand is `target_type`.
  virtual bool EffectiveOperandPrecisionIsSourceType(const HloInstruction& hlo, int64_t operand_index) const;

  PrimitiveType sourceType() const { return source_type_; }
  PrimitiveType targetType() const { return target_type_; }

protected:
  const PrimitiveType source_type_;
  const PrimitiveType target_type_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT8_SUPPORT_H_
