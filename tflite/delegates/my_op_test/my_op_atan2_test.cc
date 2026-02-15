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

class Atan2OpModel : public SingleOpModel {
 public:
  Atan2OpModel(const TensorData& input_y, const TensorData& input_x,
               const TensorData& output) {
    input_y_ = AddInput(input_y);
    input_x_ = AddInput(input_x);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ATAN2, BuiltinOptions_ATan2Options,
                 CreateATan2Options(builder_).Union());
    BuildInterpreter({GetShape(input_y_), GetShape(input_x_)});
  }

  void SetInputY(const std::vector<float>& data) { PopulateTensor(input_y_, data); }
  void SetInputX(const std::vector<float>& data) { PopulateTensor(input_x_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_y_;
  int input_x_;
  int output_;
};

TEST(MyOpTestDelegate, Atan2Test) {
  Atan2OpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                 {TensorType_FLOAT32, {1, 2, 2, 1}},
                 {TensorType_FLOAT32, {}});
  m.SetInputY({1.0, 0.0, -1.0, 0.0});
  m.SetInputX({0.0, 1.0, 0.0, -1.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatNear(1e-5),
              {1.570796, 0.0, -1.570796, 3.141592}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
