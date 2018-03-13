#ifndef KERNEL_STICKY_LENGTHS_H_
#define KERNEL_STICKY_LENGTHS_H_

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define STICKY_LENGTHS_HEAD 0
#define STICKY_LENGTHS_NECK 1
#define STICKY_LENGTHS_HAND_L 2
#define STICKY_LENGTHS_HAND_R 3
#define STICKY_LENGTHS_FOOT_L 4
#define STICKY_LENGTHS_FOOT_R 5

#define STICKY_LENGTHS_NUM_JOINTS 6

#define STICKY_LENGTHS_LIMB_ARM_L 0
#define STICKY_LENGTHS_LIMB_ARM_R 1
#define STICKY_LENGTHS_LIMB_LEG_L 2
#define STICKY_LENGTHS_LIMB_LEG_R 3

#define STICKY_LENGTHS_NUM_LIMBS 4

template <typename Device, typename T>
struct StickyLengthsFunctor {
    void operator()(const Device& d, int32 c, const T* in, T* out);
};

#ifdef GOOGLE_CUDA
template <typename T>
struct StickyLengthsFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, int32 c, const T* in, T* out);
};
#endif

#endif  // KERNEL_STICKY_LENGTHS_H_
