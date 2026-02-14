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
#include <cmath>

#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_delegate.h"
#include "tflite/kernels/internal/reference/add_n.h"
#include "tflite/kernels/internal/reference/arg_min_max.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace my_op_test {

class MyOpTestDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    int node_index = params->nodes_to_replace->data[0];
    TfLiteNode* node;
    TfLiteRegistration* registration_ptr;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration_ptr));
    const TfLiteRegistration* registration = registration_ptr;
    builtin_code_ = registration->builtin_code;

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    TfLiteTensor* output = GetOutput(context, node, 0);

    if (builtin_code_ == kTfLiteBuiltinAbs) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      if (input->type == kTfLiteFloat32) {
        const float* in_data = GetTensorData<float>(input);
        float* out_data = GetTensorData<float>(output);
        int num_elements = NumElements(input);
        for (int i = 0; i < num_elements; ++i) {
          out_data[i] = std::abs(in_data[i]);
        }
      } else {
        return kTfLiteError;
      }
    } else if (builtin_code_ == kTfLiteBuiltinAddN) {
      int num_inputs = NumInputs(node);
      const TfLiteTensor* input0 = GetInput(context, node, 0);
      if (input0->type == kTfLiteFloat32) {
        std::vector<const float*> input_data(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
          input_data[i] = GetTensorData<float>(GetInput(context, node, i));
        }
        reference_ops::AddN<float>(GetTensorShape(input0), num_inputs,
                                   input_data.data(), GetTensorData<float>(output));
      } else {
        return kTfLiteError;
      }
    } else if (builtin_code_ == kTfLiteBuiltinArgMax || builtin_code_ == kTfLiteBuiltinArgMin) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      const TfLiteTensor* axis = GetInput(context, node, 1);
      bool is_arg_max = (builtin_code_ == kTfLiteBuiltinArgMax);

      if (input->type == kTfLiteFloat32) {
        if (output->type == kTfLiteInt32) {
          if (axis->type == kTfLiteInt32) {
            reference_ops::ArgMinMax<float, int32_t, int32_t>(
                GetTensorShape(input), GetTensorData<float>(input),
                GetTensorData<int32_t>(axis), GetTensorShape(output),
                GetTensorData<int32_t>(output), is_arg_max);
          } else if (axis->type == kTfLiteInt64) {
            reference_ops::ArgMinMax<float, int32_t, int64_t>(
                GetTensorShape(input), GetTensorData<float>(input),
                GetTensorData<int64_t>(axis), GetTensorShape(output),
                GetTensorData<int32_t>(output), is_arg_max);
          }
        } else if (output->type == kTfLiteInt64) {
          if (axis->type == kTfLiteInt32) {
            reference_ops::ArgMinMax<float, int64_t, int32_t>(
                GetTensorShape(input), GetTensorData<float>(input),
                GetTensorData<int32_t>(axis), GetTensorShape(output),
                GetTensorData<int64_t>(output), is_arg_max);
          } else if (axis->type == kTfLiteInt64) {
            reference_ops::ArgMinMax<float, int64_t, int64_t>(
                GetTensorShape(input), GetTensorData<float>(input),
                GetTensorData<int64_t>(axis), GetTensorShape(output),
                GetTensorData<int64_t>(output), is_arg_max);
          }
        } else {
          return kTfLiteError;
        }
      } else {
        return kTfLiteError;
      }
    } else {
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

 private:
  int builtin_code_;
};

class MyOpTestDelegate : public SimpleDelegateInterface {
 public:
  explicit MyOpTestDelegate(const MyOpTestDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (registration->builtin_code != kTfLiteBuiltinAbs &&
        registration->builtin_code != kTfLiteBuiltinAddN &&
        registration->builtin_code != kTfLiteBuiltinArgMax &&
        registration->builtin_code != kTfLiteBuiltinArgMin) {
      return false;
    }

    const TfLiteTensor* input0 = GetInput(context, node, 0);
    if (input0->type != kTfLiteFloat32) return false;

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
