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
#include <utility>
#include <vector>

#include "tflite/builtin_ops.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/c_api_types.h"
#include "tflite/core/c/common.h"
#include "tflite/delegates/utils/simple_delegate.h"
#include "tflite/kernels/internal/reference/add.h"
#include "tflite/kernels/internal/reference/mul.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace my_op_test {

struct NodeData {
  int builtin_code;
  TfLiteFusedActivation activation;
  std::vector<int> inputs;
  std::vector<int> outputs;
};

class MyOpTestDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      int node_index = params->nodes_to_replace->data[i];
      TfLiteNode* node;
      TfLiteRegistration* registration;
      TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
          context, node_index, &node, &registration));

      NodeData node_data;
      node_data.builtin_code = registration->builtin_code;
      for (int j = 0; j < node->inputs->size; ++j) {
        node_data.inputs.push_back(node->inputs->data[j]);
      }
      for (int j = 0; j < node->outputs->size; ++j) {
        node_data.outputs.push_back(node->outputs->data[j]);
      }

      if (node_data.builtin_code == kTfLiteBuiltinAdd) {
        auto* add_params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);
        node_data.activation = add_params ? add_params->activation : kTfLiteActNone;
      } else if (node_data.builtin_code == kTfLiteBuiltinMul) {
        auto* mul_params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
        node_data.activation = mul_params ? mul_params->activation : kTfLiteActNone;
      }
      nodes_.push_back(node_data);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    for (const auto& node_data : nodes_) {
      const TfLiteTensor* input1 = &context->tensors[node_data.inputs[0]];
      const TfLiteTensor* input2 = &context->tensors[node_data.inputs[1]];
      TfLiteTensor* output = &context->tensors[node_data.outputs[0]];

      if (output->type == kTfLiteFloat32) {
        tflite::ArithmeticParams params;
        float output_activation_min, output_activation_max;
        CalculateActivationRange(node_data.activation, &output_activation_min, &output_activation_max);
        SetActivationParams(output_activation_min, output_activation_max, &params);

        if (node_data.builtin_code == kTfLiteBuiltinAdd) {
          reference_ops::Add(params, GetTensorShape(input1), GetTensorData<float>(input1),
                             GetTensorShape(input2), GetTensorData<float>(input2),
                             GetTensorShape(output), GetTensorData<float>(output));
        } else if (node_data.builtin_code == kTfLiteBuiltinMul) {
          reference_ops::Mul(params, GetTensorShape(input1), GetTensorData<float>(input1),
                             GetTensorShape(input2), GetTensorData<float>(input2),
                             GetTensorShape(output), GetTensorData<float>(output));
        }
      } else {
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

 private:
  std::vector<NodeData> nodes_;
};

class MyOpTestDelegate : public SimpleDelegateInterface {
 public:
  explicit MyOpTestDelegate(const MyOpTestDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (registration->builtin_code != kTfLiteBuiltinAdd &&
        registration->builtin_code != kTfLiteBuiltinMul) {
      return false;
    }

    const TfLiteTensor* input1 = GetInput(context, node, 0);
    if (input1->type != kTfLiteFloat32) return false;

    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "MyOpTestDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<MyOpTestDelegateKernel>();
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

 private:
  const MyOpTestDelegateOptions options_;
};

}  // namespace my_op_test
}  // namespace tflite

MyOpTestDelegateOptions TfLiteMyOpTestDelegateOptionsDefault() {
  MyOpTestDelegateOptions options;
  return options;
}

TfLiteDelegate* TfLiteMyOpTestDelegateCreate(const MyOpTestDelegateOptions* options) {
  std::unique_ptr<tflite::my_op_test::MyOpTestDelegate> delegate(
      new tflite::my_op_test::MyOpTestDelegate(
          options ? *options : TfLiteMyOpTestDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(delegate));
}

void TfLiteMyOpTestDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
