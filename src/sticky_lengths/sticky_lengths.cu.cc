#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "sticky_lengths.h"

using GPUDevice = Eigen::GpuDevice;

template <typename T>
__global__ void StickyLengthsCudaKernel(CudaLaunchConfig cfg, const T* in, T* out) {
    for(int i : CudaGridRangeX(cfg.virtual_thread_count)) {

        int sample_size = STICKY_LENGTHS_NUM_JOINTS * 2;

        int sample_start = i * sample_size;
        int out_start = i * STICKY_LENGTHS_NUM_LIMBS;

        const T neck_x = in[sample_start + STICKY_LENGTHS_NECK * 2];
        const T neck_y = in[sample_start + STICKY_LENGTHS_NECK * 2 + 1];

        const T hand_r_x = in[sample_start + STICKY_LENGTHS_HAND_R * 2];
        const T hand_r_y = in[sample_start + STICKY_LENGTHS_HAND_R * 2 + 1];

        const T hand_l_x = in[sample_start + STICKY_LENGTHS_HAND_L * 2];
        const T hand_l_y = in[sample_start + STICKY_LENGTHS_HAND_L * 2 + 1];


        const T foot_r_x = in[sample_start + STICKY_LENGTHS_FOOT_R * 2];
        const T foot_r_y = in[sample_start + STICKY_LENGTHS_FOOT_R * 2 + 1];


        const T foot_l_x = in[sample_start + STICKY_LENGTHS_FOOT_L * 2];
        const T foot_l_y = in[sample_start + STICKY_LENGTHS_FOOT_L * 2 + 1];


        out[out_start + STICKY_LENGTHS_LIMB_ARM_L] = sqrtf(powf((neck_x - hand_l_x), 2) + powf((neck_y - hand_l_y), 2));
        out[out_start + STICKY_LENGTHS_LIMB_ARM_R] = sqrtf(powf((neck_x - hand_r_x), 2) + powf((neck_y - hand_r_y), 2));
        out[out_start + STICKY_LENGTHS_LIMB_LEG_L] = sqrtf(powf((neck_x - foot_l_x), 2) + powf((neck_y - foot_l_y), 2));
        out[out_start + STICKY_LENGTHS_LIMB_LEG_R] = sqrtf(powf((neck_x - foot_r_x), 2) + powf((neck_y - foot_r_y), 2));
    }
}

template <typename T>
void StickyLengthsFunctor<GPUDevice, T>::operator()(const GPUDevice& d, int32 num_samples, const T* in, T* out) {
    // In our case, each kernel will compute the length of the limbs of a sample. We could also have each kernel compute
    // a single limb, which would perform better on a smaller number of samples. h is always NUM_JOINTS and w is always 2
    // for X and Y

    CudaLaunchConfig cfg = GetCudaLaunchConfig(num_samples, d, StickyLengthsCudaKernel<T>, 0, 0);
    StickyLengthsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, in, out);
}

template struct StickyLengthsFunctor<GPUDevice, float>;
template struct StickyLengthsFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
