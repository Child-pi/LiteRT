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

#include "tflite/delegates/my_op_test/my_op_add_test_delegate.h"

#include <memory>
#include <utility>
#include <vector>

#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_delegate.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace my_op_test {

class MyOpAddTestDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // For this simple delegate, we only replace one node at a time.
    int node_index = params->nodes_to_replace->data[0];
    TfLiteNode* node;
    TfLiteRegistration* registration_ptr;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration_ptr));
    const TfLiteRegistration* registration = registration_ptr;
    builtin_code_ = registration->builtin_code;

    // Store fused activation if any
    if (builtin_code_ == kTfLiteBuiltinAdd) {
      auto* add_params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);
      activation_ = add_params ? add_params->activation : kTfLiteActNone;
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    const TfLiteTensor* input1 = GetInput(context, node, 0);
    const TfLiteTensor* input2 = GetInput(context, node, 1);
    TfLiteTensor* output = GetOutput(context, node, 0);

    if (output->type == kTfLiteFloat32) {
      tflite::ArithmeticParams params;
      float output_activation_min, output_activation_max;
      CalculateActivationRange(activation_, &output_activation_min, &output_activation_max);
      SetActivationParams(output_activation_min, output_activation_max, &params);

      if (builtin_code_ == kTfLiteBuiltinAdd) {
        reference_ops::Add(params, GetTensorShape(input1), GetTensorData<float>(input1),
                           GetTensorShape(input2), GetTensorData<float>(input2),
                           GetTensorShape(output), GetTensorData<float>(output));
      }
    } else {
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

 private:
  int builtin_code_;
  TfLiteFusedActivation activation_ = kTfLiteActNone;
};

class MyOpAddTestDelegate : public SimpleDelegateInterface {
 public:
  explicit MyOpAddTestDelegate(const MyOpAddTestDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Support ADD op.
    if (registration->builtin_code != kTfLiteBuiltinAdd) {
      return false;
    }

    // Only support float32 for now.
    const TfLiteTensor* input1 = GetInput(context, node, 0);
    if (input1->type != kTfLiteFloat32) return false;

    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "MyOpAddTestDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<MyOpAddTestDelegateKernel>();
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

 private:
  const MyOpAddTestDelegateOptions options_;
};

}  // namespace my_op_test
}  // namespace tflite

MyOpAddTestDelegateOptions TfLiteMyOpAddTestDelegateOptionsDefault() {
  MyOpAddTestDelegateOptions options;
  return options;
}

TfLiteDelegate* TfLiteMyOpAddTestDelegateCreate(const MyOpAddTestDelegateOptions* options) {
  std::unique_ptr<tflite::my_op_test::MyOpAddTestDelegate> delegate(
      new tflite::my_op_test::MyOpAddTestDelegate(
          options ? *options : TfLiteMyOpAddTestDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(delegate));
}

void TfLiteMyOpAddTestDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
