#pragma once

#include "GPUBGR2Gray.cuh"
#include "Reduce.cuh"
#include "utils/CpuTimer.h"
#include "utils/CudaCheck.h"
#include "utils/CudaMath.cuh"
#include "utils/GpuTimer.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tuple>

namespace smd {

using namespace cv;

static constexpr size_t BLOCK_SIZE = 16;

template <typename T_in, typename T_out>
__global__ void SMDKernel(const cuda::PtrStep<T_in> src,
                          cuda::PtrStep<T_out> dst, size_t cols, size_t rows) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    T_out dx, dy;
    if (x < cols - 1 && y < rows - 1) {
        dx = src(y, x) - src(y, x + 1);
        dy = src(y, x) - src(y + 1, x);
        dst(y, x) = ::utils::abs<T_out>(dx * dy);
    }
}

std::tuple<float, float> gpuSMD(const Mat &h_oriImg) {
    const size_t rows = h_oriImg.rows;
    const size_t cols = h_oriImg.cols;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 numBlocks((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    ::utils::GpuTimer timer;
    timer.Start();
    cuda::GpuMat d_oriImg;
    d_oriImg.upload(h_oriImg);
    cuda::GpuMat d_grayImg(d_oriImg.size(), CV_8U);
    cuda::GpuMat d_dstImg(d_oriImg.size(), CV_32S);

    GPU::BGR2Gray<uchar3, uchar>
        <<<numBlocks, threadsPerBlock>>>(d_oriImg, d_grayImg, cols, rows);
    CHECK(cudaDeviceSynchronize());

    SMDKernel<uchar, int>
        <<<numBlocks, threadsPerBlock>>>(d_grayImg, d_dstImg, cols, rows);
    CHECK(cudaDeviceSynchronize());
    auto sum = Reduce<uint, ulonglong>(d_dstImg);
    timer.Stop();

    auto time = timer.Elapsed();
    auto res = static_cast<float>(sum) / (rows * cols);

    printf("Gpu elapsed time: %f\n", time);
    printf("Gpu res = %f\n", res);

    return std::make_tuple(time, res);
}

std::tuple<float, float> cpuSMD(const Mat &src) {
    Mat grayImage(src.size(), CV_8U, Scalar(0));
    ulonglong sum = 0;

    ::utils::CpuTimer timer;
    timer.Start();

    const auto cols = src.cols;
    const auto rows = src.rows;

    CPU::BGR2Gray<uchar3, uchar>(src, grayImage, cols, rows);

    for (auto x = 0; x < cols - 1; x++) {
        for (auto y = 0; y < rows - 1; y++) {
            long dx, dy;
            dx = grayImage.at<uchar>(y, x) - grayImage.at<uchar>(y, x + 1);
            dy = grayImage.at<uchar>(y, x) - grayImage.at<uchar>(y + 1, x);
            sum += abs(dx * dy);
        }
    }
    timer.Stop();

    auto time = timer.Elapsed();
    auto res = static_cast<float>(sum) / (rows * cols);

    printf("Cpu elapsed time: %f\n", time);
    printf("Cpu res = %f\n", res);

    return std::make_tuple(time, res);
}
} // namespace smd
