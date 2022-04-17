#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <opencv2/core/cuda.hpp>

constexpr unsigned FULL_MASK = 0xffffffff;

template<typename T>
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
    extern __shared__ T_out s_data[];

    auto tid = threadIdx.x;
    auto i = threadIdx.x + blockDim.x * blockIdx.x;

    auto y = i / cols;
    auto x = i % cols;

    s_data[tid] = src(y, x);
    __syncthreads();

    for (size_t s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<T_out>(s_data, tid);
    }

    if (tid == 0) {
        atomicAdd(dst, s_data[0]);
    }
}

using ulonglong = unsigned long long;

template <typename T_in, typename T_out>
T_out Reduce(const cv::cuda::GpuMat &src, const size_t BLOCK_SIZE) {
    auto cols = src.cols;
    auto rows = src.rows;
    auto size = cols * rows;

    dim3 reduceThreadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 reduceNumBlocks((size + BLOCK_SIZE * BLOCK_SIZE - 1) /
                         (BLOCK_SIZE * BLOCK_SIZE));

    T_out h_dst = 0;
    T_out *d_dst;
    CHECK(cudaMalloc((void **)&d_dst, sizeof(T_out)));
    CHECK(cudaMemcpy(d_dst, &h_dst, sizeof(T_out), cudaMemcpyHostToDevice));
    reduce<T_in, T_out>
        <<<reduceNumBlocks, reduceThreadsPerBlock,
           BLOCK_SIZE * BLOCK_SIZE * sizeof(T_out)>>>(src, d_dst, cols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h_dst, d_dst, sizeof(T_out), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_dst));

    return h_dst;
}

#endif
