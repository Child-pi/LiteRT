load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "my_op_add_test_delegate",
    srcs = ["my_op_add_test_delegate.cc"],
    hdrs = ["my_op_add_test_delegate.h"],
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
    srcs = ["my_op_mul_test_delegate.cc"],
    hdrs = ["my_op_mul_test_delegate.h"],
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
    name = "my_op_test_delegate",
    srcs = ["my_op_test_delegate.cc"],
    hdrs = ["my_op_test_delegate.h"],
    copts = tflite_copts(),
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:builtin_op_data",
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

cc_test(
    name = "my_op_add_test_delegate_test",
    srcs = ["my_op_add_test_delegate_test.cc"],
    deps = [
        ":my_op_add_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_mul_test_delegate_test",
    srcs = ["my_op_mul_test_delegate_test.cc"],
    deps = [
        ":my_op_mul_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_abs_test",
    srcs = ["my_op_abs_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_add_n_test",
    srcs = ["my_op_add_n_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_arg_max_test",
    srcs = ["my_op_arg_max_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_arg_min_test",
    srcs = ["my_op_arg_min_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_atan2_test",
    srcs = ["my_op_atan2_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_average_pool_2d_test",
    srcs = ["my_op_average_pool_2d_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_batch_matmul_test",
    srcs = ["my_op_batch_matmul_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "my_op_batch_to_space_nd_test",
    srcs = ["my_op_batch_to_space_nd_test.cc"],
    deps = [
        ":my_op_test_delegate",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
