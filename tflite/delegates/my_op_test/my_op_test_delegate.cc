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
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/delegates/utils/simple_delegate.h"
#include "tflite/kernels/internal/reference/add_n.h"
#include "tflite/kernels/internal/reference/arg_min_max.h"
#include "tflite/kernels/internal/reference/batch_matmul.h"
#include "tflite/kernels/internal/reference/batch_to_space_nd.h"
#include "tflite/kernels/internal/reference/pooling.h"
#include "tflite/kernels/padding.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace my_op_test {

inline RuntimeShape SwapRowColumnDims(const RuntimeShape& shape) {
  RuntimeShape swapped_shape(shape);
  const int32_t dims = shape.DimensionsCount();
  if (dims >= 2) {
    swapped_shape.SetDim(dims - 2, shape.Dims(dims - 1));
    swapped_shape.SetDim(dims - 1, shape.Dims(dims - 2));
  }
  return swapped_shape;
}

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

    if (builtin_code_ == kTfLiteBuiltinAveragePool2d) {
      pool_params_ = *reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
    } else if (builtin_code_ == kTfLiteBuiltinBatchMatmul) {
      batch_matmul_params_ = *reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    TfLiteTensor* output = GetOutput(context, node, 0);
    if (builtin_code_ == kTfLiteBuiltinAbs || builtin_code_ == kTfLiteBuiltinAtan2 ||
        builtin_code_ == kTfLiteBuiltinAddN) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
    } else if (builtin_code_ == kTfLiteBuiltinArgMax || builtin_code_ == kTfLiteBuiltinArgMin) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      const TfLiteTensor* axis = GetInput(context, node, 1);
      int axis_value = axis->type == kTfLiteInt64 ? *GetTensorData<int64_t>(axis) : *GetTensorData<int32_t>(axis);
      if (axis_value < 0) axis_value += NumDimensions(input);
      TfLiteIntArray* output_dims = TfLiteIntArrayCreate(NumDimensions(input) - 1);
      for (int i = 0, j = 0; i < NumDimensions(input); ++i) {
        if (i != axis_value) output_dims->data[j++] = input->dims->data[i];
      }
      return context->ResizeTensor(context, output, output_dims);
    } else if (builtin_code_ == kTfLiteBuiltinAveragePool2d) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
      ComputePaddingHeightWidth(
          pool_params_.stride_height, pool_params_.stride_width, 1, 1, input->dims->data[1],
          input->dims->data[2], pool_params_.filter_height, pool_params_.filter_width,
          pool_params_.padding, &output_dims->data[1], &output_dims->data[2]);
      return context->ResizeTensor(context, output, output_dims);
    } else if (builtin_code_ == kTfLiteBuiltinBatchMatmul) {
      const TfLiteTensor* lhs = GetInput(context, node, 0);
      const TfLiteTensor* rhs = GetInput(context, node, 1);
      int lhs_rank = NumDimensions(lhs);
      int rhs_rank = NumDimensions(rhs);
      int output_rank = std::max(lhs_rank, rhs_rank);
      TfLiteIntArray* output_dims = TfLiteIntArrayCreate(output_rank);
      for (int i = 0; i < output_rank - 2; ++i) {
        int lhs_dim = i < lhs_rank - 2 ? lhs->dims->data[i] : 1;
        int rhs_dim = i < rhs_rank - 2 ? rhs->dims->data[i] : 1;
        output_dims->data[i] = std::max(lhs_dim, rhs_dim);
      }
      output_dims->data[output_rank - 2] = batch_matmul_params_.adj_x ? lhs->dims->data[lhs_rank - 1] : lhs->dims->data[lhs_rank - 2];
      output_dims->data[output_rank - 1] = batch_matmul_params_.adj_y ? rhs->dims->data[rhs_rank - 2] : rhs->dims->data[rhs_rank - 1];
      return context->ResizeTensor(context, output, output_dims);
    } else if (builtin_code_ == kTfLiteBuiltinBatchToSpaceNd) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      const TfLiteTensor* block_shape = GetInput(context, node, 1);
      const TfLiteTensor* crops = GetInput(context, node, 2);
      const int32_t* block_shape_data = GetTensorData<int32_t>(block_shape);
      const int32_t* crops_data = GetTensorData<int32_t>(crops);
      int spatial_dims_num = input->dims->size - 2;
      TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
      int output_batch_size = input->dims->data[0];
      for (int dim = 0; dim < spatial_dims_num; ++dim) {
        output_batch_size /= block_shape_data[dim];
        output_dims->data[dim + 1] = input->dims->data[dim + 1] * block_shape_data[dim] -
                                     crops_data[dim * 2] - crops_data[dim * 2 + 1];
      }
      output_dims->data[0] = output_batch_size;
      return context->ResizeTensor(context, output, output_dims);
    }
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
    } else if (builtin_code_ == kTfLiteBuiltinAtan2) {
      const TfLiteTensor* input_y = GetInput(context, node, 0);
      const TfLiteTensor* input_x = GetInput(context, node, 1);
      if (input_y->type == kTfLiteFloat32) {
        const float* data_y = GetTensorData<float>(input_y);
        const float* data_x = GetTensorData<float>(input_x);
        float* data_out = GetTensorData<float>(output);
        int num_elements = NumElements(input_y);
        for (int i = 0; i < num_elements; ++i) {
          data_out[i] = std::atan2(data_y[i], data_x[i]);
        }
      } else {
        return kTfLiteError;
      }
    } else if (builtin_code_ == kTfLiteBuiltinAveragePool2d) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      PoolParams op_params;
      op_params.stride_height = pool_params_.stride_height;
      op_params.stride_width = pool_params_.stride_width;
      op_params.filter_height = pool_params_.filter_height;
      op_params.filter_width = pool_params_.filter_width;
      op_params.float_activation_min = -std::numeric_limits<float>::infinity();
      op_params.float_activation_max = std::numeric_limits<float>::infinity();
      CalculateActivationRange(pool_params_.activation, &op_params.float_activation_min,
                               &op_params.float_activation_max);

      TfLitePaddingValues padding_values = ComputePaddingHeightWidth(
          pool_params_.stride_height, pool_params_.stride_width, 1, 1, input->dims->data[1],
          input->dims->data[2], pool_params_.filter_height, pool_params_.filter_width,
          pool_params_.padding, &output->dims->data[1], &output->dims->data[2]);
      op_params.padding_values.height = padding_values.height;
      op_params.padding_values.width = padding_values.width;

      if (input->type == kTfLiteFloat32) {
        reference_ops::AveragePool(op_params, GetTensorShape(input), GetTensorData<float>(input),
                                   GetTensorShape(output), GetTensorData<float>(output));
      } else {
        return kTfLiteError;
      }
    } else if (builtin_code_ == kTfLiteBuiltinBatchMatmul) {
      const TfLiteTensor* lhs = GetInput(context, node, 0);
      const TfLiteTensor* rhs = GetInput(context, node, 1);
      if (lhs->type == kTfLiteFloat32 && !batch_matmul_params_.adj_x && batch_matmul_params_.adj_y) {
        RuntimeShape rhs_shape = GetTensorShape(rhs);
        RuntimeShape lhs_shape = SwapRowColumnDims(GetTensorShape(lhs));
        reference_ops::BatchMatMul(rhs_shape, GetTensorData<float>(rhs),
                                   lhs_shape, GetTensorData<float>(lhs),
                                   GetTensorShape(output), GetTensorData<float>(output));
      } else {
        // For simplicity in this test delegate, we only support this specific case.
        // In a real delegate, you'd handle all cases.
        return kTfLiteError;
      }
    } else if (builtin_code_ == kTfLiteBuiltinBatchToSpaceNd) {
      const TfLiteTensor* input = GetInput(context, node, 0);
      const TfLiteTensor* block_shape = GetInput(context, node, 1);
      const TfLiteTensor* crops = GetInput(context, node, 2);
      if (input->type == kTfLiteFloat32) {
        reference_ops::BatchToSpaceND(GetTensorShape(input), GetTensorData<float>(input),
                                      GetTensorShape(block_shape), GetTensorData<int32_t>(block_shape),
                                      GetTensorShape(crops), GetTensorData<int32_t>(crops),
                                      GetTensorShape(output), GetTensorData<float>(output));
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
  TfLitePoolParams pool_params_;
  TfLiteBatchMatMulParams batch_matmul_params_;
};

class MyOpTestDelegate : public SimpleDelegateInterface {
 public:
  explicit MyOpTestDelegate(const MyOpTestDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (registration->builtin_code != kTfLiteBuiltinAbs &&
        registration->builtin_code != kTfLiteBuiltinAtan2 &&
        registration->builtin_code != kTfLiteBuiltinAveragePool2d &&
        registration->builtin_code != kTfLiteBuiltinBatchMatmul &&
        registration->builtin_code != kTfLiteBuiltinBatchToSpaceNd &&
        registration->builtin_code != kTfLiteBuiltinAddN &&
        registration->builtin_code != kTfLiteBuiltinArgMax &&
        registration->builtin_code != kTfLiteBuiltinArgMin) {
      return false;
    }

    if (registration->builtin_code == kTfLiteBuiltinBatchMatmul) {
        auto* params = reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
        if (params->adj_x || !params->adj_y) return false;
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
