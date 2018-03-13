#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "sticky_lengths.h"


REGISTER_OP("StickyLengths")
    // Accepts either int or float as input values. Return floats
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto input_shape = c->input(0);

        c->set_output(0, c->MakeShape({c->Dim(input_shape, 0), c->MakeDim(STICKY_LENGTHS_NUM_LIMBS)}));
        return Status::OK();
    });


template <typename Device, typename T>
class StickyLengthsOp : public OpKernel {
 public:
  explicit StickyLengthsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    // We want the input tensor to be num_samples, joints, x/y
    OP_REQUIRES(context, input_tensor.dims() == 3, errors::InvalidArgument("input tensor must be rank 3"));
    OP_REQUIRES(context, input_tensor.dim_size(1) == STICKY_LENGTHS_NUM_JOINTS, errors::InvalidArgument("Unsupported number of joints"));
    OP_REQUIRES(context, input_tensor.dim_size(2) == 2, errors::InvalidArgument("Unsupported number of coordinates"));

    const int num_samples = input_tensor.dim_size(0);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {num_samples, STICKY_LENGTHS_NUM_LIMBS}, &output_tensor));

    StickyLengthsFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(),
        num_samples,
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

#ifdef GOOGLE_CUDA

#define DECLARE_GPU_SPEC(T)                                                                 \
  template <>                                                                               \
  void StickyLengthsFunctor<GPUDevice, T>::operator()(                                           \
      const GPUDevice& d, int32 num_samples, const T* in, T* out);         \
extern template struct StickyLengthsFunctor<GPUDevice, T>;
DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);

#define REGISTER_GPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("StickyLengths")                      \
      .Device(DEVICE_GPU)                                       \
      .TypeConstraint<T>("T"), StickyLengthsOp<GPUDevice, T>);
REGISTER_GPU(int32);
REGISTER_GPU(float);

#endif
