#ifndef __GPU_ABS_H__
#define __GPU_ABS_H__

namespace CudaMath {
template <typename T>
__device__ T abs(T x) {
    if (x < 0)
        return -x;
    return x;
}
} // namespace CudaMath
#endif
