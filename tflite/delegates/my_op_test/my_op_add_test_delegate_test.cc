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
#include "tflite/delegates/my_op_test/my_op_add_test_delegate.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace {

class MyOpAddTestModel : public SingleOpModel {
 public:
  MyOpAddTestModel(const TensorData& input1, const TensorData& input2,
                   const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(const std::vector<float>& data) { PopulateTensor(input1_, data); }
  void SetInput2(const std::vector<float>& data) { PopulateTensor(input2_, data); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input1_;
  int input2_;
  int output_;
};

TEST(MyOpAddTestDelegate, AddTest) {
  MyOpAddTestModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                     {TensorType_FLOAT32, {1, 2, 2, 1}},
                     {TensorType_FLOAT32, {}});
  m.SetInput1({-2.0, 0.2, 0.7, 0.8});
  m.SetInput2({0.1, 0.2, 0.3, 0.5});

  MyOpAddTestDelegateOptions options = TfLiteMyOpAddTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpAddTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {-1.9, 0.4, 1.0, 1.3}));

  TfLiteMyOpAddTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
