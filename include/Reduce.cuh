#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <cooperative_groups.h>
#include <opencv2/core/cuda.hpp>

namespace cg = cooperative_groups;

constexpr unsigned FULL_MASK = 0xffffffff;

template <typename T_in, typename T_out>
__global__ void reduce(const cv::cuda::PtrStep<T_in> src, T_out *dst,
                       size_t cols) {
    extern __shared__ T_out s_data[];

    auto tid = threadIdx.x;
    auto i = threadIdx.x + blockDim.x * 2 * blockIdx.x;

    auto y = i / cols;
    auto x = i % cols;

    auto ni = i + blockDim.x;
    auto ny = ni / cols;
    auto nx = ni % cols;

    s_data[tid] = src(y, x) + src(ny, nx);
    __syncthreads();

    for (size_t s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    T_out temp = s_data[tid];
    cg::thread_block_tile<32> g =
        cg::tiled_partition<32>(cg::this_thread_block());
    for (size_t s = g.size() >> 1; s >= 1; s >>= 1) {
        temp += g.shfl_down(temp, s);
    }

    if (tid == 0) {
        atomicAdd(dst, temp);
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
                         (BLOCK_SIZE * BLOCK_SIZE) / 2);

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
