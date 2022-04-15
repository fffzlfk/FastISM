#pragma once

#include "CudaCheck.h"

namespace utils {
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    void Start() { CHECK(cudaEventRecord(start, 0)); }

    void Stop() { CHECK(cudaEventRecord(stop, 0)); }

    float Elapsed() {
        float elapsed;
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};
} // namespace utils
