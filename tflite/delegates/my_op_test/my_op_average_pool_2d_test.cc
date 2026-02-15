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

class AveragePoolOpModel : public SingleOpModel {
 public:
  AveragePoolOpModel(const TensorData& input, int filter_height, int filter_width,
                     int stride_height, int stride_width, Padding padding) {
    input_ = AddInput(input);
    output_ = AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(BuiltinOperator_AVERAGE_POOL_2D, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, stride_width, stride_height,
                                   filter_width, filter_height, ActivationFunctionType_NONE)
                     .Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) { PopulateTensor(input_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(MyOpTestDelegate, AveragePoolTest) {
  AveragePoolOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, 2, 2, 1, 1, Padding_VALID);
  m.SetInput({1.0, 2.0, 3.0, 4.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {2.5}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
