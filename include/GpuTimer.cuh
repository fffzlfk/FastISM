#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include "CudaCheck.h"

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

#endif /* __GPU_TIMER_H__ */
