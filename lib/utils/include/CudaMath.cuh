#ifndef __GPU_MATH_H__
#define __GPU_MATH_H__

namespace utils {
template <typename T>
__device__ T abs(T x) {
    if (x < 0)
        return -x;
    return x;
}
} // namespace utils
#endif
