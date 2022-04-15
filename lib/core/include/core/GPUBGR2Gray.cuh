#pragma once

#include <opencv2/core/cuda.hpp>
namespace GPU {
template <typename T_in, typename T_out>
__global__ void BGR2Gray(const cv::cuda::PtrStep<T_in> src,
                         cv::cuda::PtrStep<T_out> dst, size_t cols,
                         size_t rows) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        auto cell = src(y, x);
        dst(y, x) = (T_out)0.114f * cell.x + 0.587f * cell.y + 0.299f * cell.z;
    }
}
} // namespace GPU
