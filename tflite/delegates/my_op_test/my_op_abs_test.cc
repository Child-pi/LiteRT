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

class AbsOpModel : public SingleOpModel {
 public:
  AbsOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ABS, BuiltinOptions_AbsOptions,
                 CreateAbsOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) { PopulateTensor(input_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(MyOpTestDelegate, AbsTest) {
  AbsOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
               {TensorType_FLOAT32, {1, 2, 2, 1}});
  m.SetInput({-2.0, 0.2, -0.7, 0.8});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {2.0, 0.2, 0.7, 0.8}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
