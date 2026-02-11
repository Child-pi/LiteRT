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
#include "tflite/delegates/my_op_test/my_op_test_delegate.h"

#include <memory>
#include <vector>
#include <cstdlib>

#include <gtest/gtest.h>
#include "tflite/builtin_ops.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/core/kernels/builtin_op_kernels.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"

namespace tflite {
namespace my_op_test {
namespace {

TEST(MyOpTestDelegateTest, BasicAdd) {
  std::unique_ptr<Interpreter> interpreter = std::make_unique<Interpreter>();
  interpreter->AddTensors(3);
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  TfLiteQuantizationParams quant;
  for (int i = 0; i < 3; ++i) {
    interpreter->SetTensorParametersReadWrite(i, kTfLiteFloat32, "", {1}, quant);
  }

  TfLiteAddParams* add_params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;

  TfLiteRegistration reg = *ops::builtin::Register_ADD();
  reg.builtin_code = kTfLiteBuiltinAdd;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, add_params, &reg);

  // Apply the delegate.
  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  auto delegate_ptr = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      delegate, TfLiteMyOpTestDelegateDelete);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate_ptr)), kTfLiteOk);

  // Check that node was delegated.
  ASSERT_EQ(interpreter->execution_plan().size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(interpreter->execution_plan()[0]);
  EXPECT_STREQ("MyOpTestDelegate", node_and_reg->second.custom_name);

  // Run inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  interpreter->typed_tensor<float>(0)[0] = 2.0f;
  interpreter->typed_tensor<float>(1)[0] = 3.0f;
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  EXPECT_EQ(interpreter->typed_tensor<float>(2)[0], 5.0f);
}

TEST(MyOpTestDelegateTest, BasicMul) {
  std::unique_ptr<Interpreter> interpreter = std::make_unique<Interpreter>();
  interpreter->AddTensors(3);
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  TfLiteQuantizationParams quant;
  for (int i = 0; i < 3; ++i) {
    interpreter->SetTensorParametersReadWrite(i, kTfLiteFloat32, "", {1}, quant);
  }

  TfLiteMulParams* mul_params = reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  mul_params->activation = kTfLiteActNone;

  TfLiteRegistration reg = *ops::builtin::Register_MUL();
  reg.builtin_code = kTfLiteBuiltinMul;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, mul_params, &reg);

  // Apply the delegate.
  MyOpTestDelegateOptions options = TfLiteMyOpTestDelegateOptionsDefault();
  auto delegate = TfLiteMyOpTestDelegateCreate(&options);
  auto delegate_ptr = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      delegate, TfLiteMyOpTestDelegateDelete);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate_ptr)), kTfLiteOk);

  // Check that node was delegated.
  ASSERT_EQ(interpreter->execution_plan().size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(interpreter->execution_plan()[0]);
  EXPECT_STREQ("MyOpTestDelegate", node_and_reg->second.custom_name);

  // Run inference.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  interpreter->typed_tensor<float>(0)[0] = 4.0f;
  interpreter->typed_tensor<float>(1)[0] = 5.0f;
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  EXPECT_EQ(interpreter->typed_tensor<float>(2)[0], 20.0f);
}

}  // namespace
}  // namespace my_op_test
}  // namespace tflite
