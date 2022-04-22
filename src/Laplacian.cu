#include "CPUBGR2Gray.h"
#include "CpuTimer.h"
#include "CudaCheck.h"
#include "CudaMath.h"
#include "GPUBGR2Gray.cuh"
#include "GpuTimer.cuh"
#include "Reduce.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

template <typename T_in, typename T_out>
__global__ void LaplacianKernel(const cuda::PtrStep<T_in> src,
                                cuda::PtrStep<T_out> dst, size_t cols,
                                size_t rows) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    T_out lx, ly;
    if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
        lx = CudaMath::abs<T_out>(src(y, x + 1) + src(y, x - 1) - 2 * src(y, x));
        ly = CudaMath::abs<T_out>(src(y + 1, x) + src(y - 1, x) - 2 * src(y, x));
        dst(y, x) = lx + ly;
    }
}

void gpuLaplacian(const Mat &h_oriImg) {
    const size_t rows = h_oriImg.rows;
    const size_t cols = h_oriImg.cols;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 numBlocks((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    GpuTimer timer;
    timer.Start();
    cuda::GpuMat d_oriImg;
    d_oriImg.upload(h_oriImg);
    cuda::GpuMat d_grayImg(d_oriImg.size(), CV_8U);
    cuda::GpuMat d_dstImg(d_oriImg.size(), CV_32S);

    GPU::BGR2Gray<uchar3, uchar>
        <<<numBlocks, threadsPerBlock>>>(d_oriImg, d_grayImg, cols, rows);
    CHECK(cudaDeviceSynchronize());

    LaplacianKernel<uchar, int>
        <<<numBlocks, threadsPerBlock>>>(d_grayImg, d_dstImg, cols, rows);
    CHECK(cudaDeviceSynchronize());
    auto sum = Reduce<uint, ulonglong>(d_dstImg);
    timer.Stop();

    printf("Gpu elapsed time: %f\n", timer.Elapsed());
    printf("Gpu res = %f\n", (float)sum / (rows * cols));
}

void cpuLaplacian(const Mat &src) {
    Mat grayImage(src.size(), CV_8U, Scalar(0));
    ulonglong sum = 0;

    CpuTimer timer;
    timer.Start();

    const auto cols = src.cols;
    const auto rows = src.rows;

    CPU::BGR2Gray<uchar3, uchar>(src, grayImage, cols, rows);

    for (auto x = 1; x < cols - 1; x++) {
        for (auto y = 1; y < rows - 1; y++) {
            int lx, ly;
            lx = grayImage.at<uchar>(y, x + 1) + grayImage.at<uchar>(y, x - 1) -
                 2 * grayImage.at<uchar>(y, x);
            ly = grayImage.at<uchar>(y + 1, x) + grayImage.at<uchar>(y - 1, x) -
                 2 * grayImage.at<uchar>(y, x);
            sum += abs(lx) + abs(ly);
        }
    }
    timer.Stop();

    printf("Cpu elapsed time: %f\n", timer.Elapsed());
    printf("Cpu res = %f\n", (float)sum / (rows * cols));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf(
            "%s: Invalid number of command line arguments. Exiting program\n",
            argv[0]);
        printf("Usage: %s [image]", argv[0]);
    }

    Mat h_oriImg = imread(argv[1], IMREAD_COLOR);

    gpuLaplacian(h_oriImg);

    cpuLaplacian(h_oriImg);

    return 0;
}
