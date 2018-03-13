load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "sticky_lengths.so",
    srcs = ["sticky_lengths.h", "sticky_lengths.cc"],
    gpu_srcs = ["sticky_lengths.h", "sticky_lengths.cu.cc"],
)

# bazel build --config opt //tensorflow/core/user_ops/sticky_lengths:sticky_lengths.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"; rm sticky_lengths.so -f; cp bazel-bin/tensorflow/core/user_ops/sticky_lengths/sticky_lengths.so .
