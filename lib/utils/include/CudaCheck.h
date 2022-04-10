#ifndef __CUDA_CHECK_H__
#define __CUDA_CHECK_H__

#include <cstdio>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("ERROR: %s:%d", __FILE__, __LINE__);                        \
            printf("code: %d, reason: %s\n", error,                            \
                   cudaGetErrorString(error));                                 \
            exit(1);                                                           \
        }                                                                      \
    }
#endif
