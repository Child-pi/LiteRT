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

#ifndef TENSORFLOW_LITE_DELEGATES_MY_OP_TEST_MY_OP_MUL_TEST_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_MY_OP_TEST_MY_OP_MUL_TEST_DELEGATE_H_

#include <memory>

#include "tflite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Add options here if needed.
} MyOpMulTestDelegateOptions;

// Returns a structure with the default delegate options.
MyOpMulTestDelegateOptions TfLiteMyOpMulTestDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteMyOpMulTestDelegateDelete` when delegate is no longer used by TFLite.
TfLiteDelegate* TfLiteMyOpMulTestDelegateCreate(const MyOpMulTestDelegateOptions* options);

// Destroys a delegate created with `TfLiteMyOpMulTestDelegateCreate` call.
void TfLiteMyOpMulTestDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_MY_OP_TEST_MY_OP_MUL_TEST_DELEGATE_H_
