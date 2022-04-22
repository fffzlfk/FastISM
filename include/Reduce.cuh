#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <opencv2/core/cuda.hpp>

constexpr size_t BLOCK_SIZE = 16;

template <typename T_in, typename T_out>
__global__ void reduce(const cv::cuda::PtrStep<T_in> src, T_out *dst,
                       size_t cols) {
    __shared__ T_out s_data[BLOCK_SIZE * BLOCK_SIZE];

    auto tid = threadIdx.x;
    auto i = threadIdx.x + blockDim.x * blockIdx.x;

    auto y = i / cols;
    auto x = i % cols;

    s_data[tid] = src(y, x);
    __syncthreads();

    for (auto s = blockDim.x >> 1; s > 0; s >>= 1) {
        // 避免了`bank conflicts`
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dst, s_data[0]);
    }
}

using ulonglong = unsigned long long;

template <typename T_in, typename T_out>
T_out Reduce(const cv::cuda::GpuMat &src) {
    auto cols = src.cols;
    auto rows = src.rows;
    auto size = cols * rows;

    dim3 reduceThreadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 reduceNumBlocks((size + BLOCK_SIZE * BLOCK_SIZE - 1) /
                         (BLOCK_SIZE * BLOCK_SIZE));

    T_out h_dst = 0;
    T_out *d_dst;
    CHECK(cudaMalloc((void **)&d_dst, sizeof(T_out)));
    CHECK(
        cudaMemcpy(d_dst, &h_dst, sizeof(T_out), cudaMemcpyHostToDevice));
    reduce<T_in, T_out>
        <<<reduceNumBlocks, reduceThreadsPerBlock>>>(src, d_dst, cols);
    CHECK(cudaDeviceSynchronize());

    CHECK(
        cudaMemcpy(&h_dst, d_dst, sizeof(T_out), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_dst));

    return h_dst;
}

#endif
