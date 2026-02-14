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

class AddNOpModel : public SingleOpModel {
 public:
  AddNOpModel(const std::vector<TensorData>& inputs, const TensorData& output) {
    for (const auto& input : inputs) {
      inputs_.push_back(AddInput(input));
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD_N, BuiltinOptions_AddNOptions,
                 CreateAddNOptions(builder_).Union());
    std::vector<std::vector<int>> input_shapes;
    for (int input : inputs_) {
      input_shapes.push_back(GetShape(input));
    }
    BuildInterpreter(input_shapes);
  }

  void SetInput(int index, const std::vector<float>& data) {
    PopulateTensor(inputs_[index], data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  std::vector<int> inputs_;
  int output_;
};

TEST(MyOpTestDelegate, AddNTest) {
  AddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                 {TensorType_FLOAT32, {1, 2, 2, 1}},
                 {TensorType_FLOAT32, {1, 2, 2, 1}}},
                {TensorType_FLOAT32, {1, 2, 2, 1}});
  m.SetInput(0, {1.0, 2.0, 3.0, 4.0});
  m.SetInput(1, {10.0, 20.0, 30.0, 40.0});
  m.SetInput(2, {100.0, 200.0, 300.0, 400.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::Pointwise(testing::FloatEq(), {111.0, 222.0, 333.0, 444.0}));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
