#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <opencv2/core/cuda.hpp>

constexpr size_t BLOCK_SIZE = 16;

// `warp`的大小为32，`warp`中是`SIMT`策略，
// 也就是单指令多线程，所以无需考虑同步问题，
// 使用`volatile`防止寄存器缓存（编译器优化）
// 我们可以直接展开
template <typename T>
__device__ void warpReduce(volatile T *s_data, int tid) {
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

template <typename T_in, typename T_out>
__global__ void reduce(const cv::cuda::PtrStep<T_in> src, T_out *dst,
                       size_t cols) {
    __shared__ T_out s_data[BLOCK_SIZE * BLOCK_SIZE];

    auto tid = threadIdx.x;
    auto i = threadIdx.x + blockDim.x * 2 * blockIdx.x;
    auto y = i / cols;
    auto x = i % cols;

    auto ni = i + blockDim.x;
    auto ny = ni / cols;
    auto nx = ni % cols;

    // 在赋值的时候执行第一层reduce，避免了一半线程浪费
    s_data[tid] = src(y, x) + src(ny, nx);
    __syncthreads();

    for (auto s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(s_data, tid);
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
    // 每个`block`中的线程不变，但总线程数少了一半
    dim3 reduceNumBlocks((size + BLOCK_SIZE * BLOCK_SIZE * 2 - 1) /
                         (BLOCK_SIZE * BLOCK_SIZE * 2));

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
