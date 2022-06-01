#pragma once

#include "core/Laplacian.cuh"
#include "core/SMD.cuh"
#include "core/Tenengrad.cuh"
#include "method_types.h"
#include <tuple>
#include <vector>
#include <iostream>

namespace ui {
auto compute(Method method, const std::vector<std::string> &files,
             const size_t index) {
    auto filepath = files[index];
    auto image = cv::imread(filepath, cv::IMREAD_COLOR);
    switch (method) {
    case Method::CPULaplacian:
        return laplacian::cpuLaplacian(image);
    case Method::GPULaplacian:
        return laplacian::gpuLaplacian(image);
    case Method::GPUTenengrad:
        return tenengrad::gpuTenengrad(image);
    case Method::CPUTenengrad:
        return tenengrad::cpuTenengrad(image);
    case Method::CPUSMD:
        return smd::cpuSMD(image);
    case Method::GPUSMD:
        return smd::gpuSMD(image);
    default:
        return std::make_tuple(.0f, .0f);
    }
}
} // namespace ui