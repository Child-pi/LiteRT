/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <vector>
#include "tflite/delegates/my_op_test/my_op_test_delegate.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(const TensorData& lhs, const TensorData& rhs,
                     bool adj_x, bool adj_y) {
    lhs_ = AddInput(lhs);
    rhs_ = AddInput(rhs);
    output_ = AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_), GetShape(rhs_)});
  }

  void SetLHS(const std::vector<float>& data) { PopulateTensor(lhs_, data); }
  void SetRHS(const std::vector<float>& data) { PopulateTensor(rhs_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int lhs_;
  int rhs_;
  int output_;
};

TEST(MyOpTestDelegate, BatchMatMulTest) {
  // Reference op expects RHS to be [cols, accum_depth] when adj_y is true.
  // Standard matmul [1, 2] x [2, 3] -> [1, 3]
  // With adj_y = true, RHS should be [3, 2].
  BatchMatMulOpModel m({TensorType_FLOAT32, {1, 2}},
                       {TensorType_FLOAT32, {3, 2}}, false, true);
  m.SetLHS({1.0, 2.0});
  m.SetRHS({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // [1, 2] * [1, 3, 5]^T = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {5.0, 11.0, 17.0}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
