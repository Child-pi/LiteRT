load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "my_op_add_test_delegate",
    srcs = [
        "my_op_add_test_delegate.cc",
    ],
    hdrs = [
        "my_op_add_test_delegate.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/delegates/utils:simple_delegate",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
    ],
)

cc_library(
    name = "my_op_mul_test_delegate",
    srcs = [
        "my_op_mul_test_delegate.cc",
    ],
    hdrs = [
        "my_op_mul_test_delegate.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/delegates/utils:simple_delegate",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
    ],
)
