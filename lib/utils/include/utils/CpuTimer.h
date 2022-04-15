#pragma once

#include <sys/time.h>

namespace utils {
class CpuTimer {
  private:
    long startTime, endTime;

  public:
    CpuTimer() : startTime(0), endTime(0) {}

    ~CpuTimer() {}

    void Start() {
        struct timeval time;
        if (gettimeofday(&time, 0))
            return;
        startTime = 1000000 * time.tv_sec + time.tv_usec;
    }

    void Stop() {
        struct timeval time;
        if (gettimeofday(&time, 0))
            return;
        endTime = 1000000 * time.tv_sec + time.tv_usec;
    }

    float Elapsed() {
        float elapsed = (endTime - startTime) / 1000.0;
        return elapsed;
    }
};
} // namespace utils