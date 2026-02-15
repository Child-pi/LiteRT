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

class ArgMinMaxOpModel : public SingleOpModel {
 public:
  ArgMinMaxOpModel(BuiltinOperator op, const TensorData& input, int axis_value,
                   TensorType output_type) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, {axis_value}, {1});
    output_ = AddOutput({output_type, {}});

    if (op == BuiltinOperator_ARG_MAX) {
      SetBuiltinOp(op, BuiltinOptions_ArgMaxOptions,
                   CreateArgMaxOptions(builder_, output_type).Union());
    } else {
      SetBuiltinOp(op, BuiltinOptions_ArgMinOptions,
                   CreateArgMinOptions(builder_, output_type).Union());
    }
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) { PopulateTensor(input_, data); }

  template <typename T>
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int axis_;
  int output_;
};

TEST(MyOpTestDelegate, ArgMaxTest) {
  ArgMinMaxOpModel m(BuiltinOperator_ARG_MAX, {TensorType_FLOAT32, {1, 3}}, 1, TensorType_INT32);
  m.SetInput({1.0, 3.0, 2.0});

  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  m.SetDelegate(delegate);
  m.ApplyDelegate();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), testing::ElementsAre(1));

  TfLiteMyOpTestDelegateDelete(delegate);
}

}  // namespace
}  // namespace tflite
