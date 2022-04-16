#pragma once

#include "core/Laplacian.cuh"
#include "core/Tenengrad.cuh"
#include "core/SMD.cuh"
#include "method_types.h"
#include <tuple>
#include <vector>

namespace ui {
auto compute(Method method, const std::vector<std::string> &files,
             const size_t index) {
    auto filepath = files[index];
    auto image = cv::imread(filepath);
    switch (method) {
    case Method::Laplacian:
        return laplacian::gpuLaplacian(image);
    case Method::Tenengrad:
        return tenengrad::gpuTenengrad(image);
    case Method::SMD:
        return smd::gpuSMD(image);
    default:
        return std::make_tuple(.0f, .0f);
    }
}
} // namespace ui