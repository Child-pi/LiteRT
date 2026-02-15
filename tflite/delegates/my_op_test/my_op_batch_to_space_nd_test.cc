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

class BatchToSpaceNdOpModel : public SingleOpModel {
 public:
  BatchToSpaceNdOpModel(const TensorData& input, const std::vector<int>& block_shape,
                        const std::vector<int>& crops) {
    input_ = AddInput(input);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {static_cast<int>(block_shape.size())});
    crops_ = AddConstInput(TensorType_INT32, crops, {static_cast<int>(crops.size()) / 2, 2});
    output_ = AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND, BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) { PopulateTensor(input_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int block_shape_;
  int crops_;
  int output_;
};

TEST(MyOpTestDelegate, BatchToSpaceNdTest) {
  // Input: [4, 1, 1, 1], Block: [2, 2], Crops: [[0, 0], [0, 0]]
  // Output: [1, 2, 2, 1]
  BatchToSpaceNdOpModel m({TensorType_FLOAT32, {4, 1, 1, 1}}, {2, 2}, {0, 0, 0, 0});
  m.SetInput({1.0, 2.0, 3.0, 4.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {1.0, 2.0, 3.0, 4.0}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
