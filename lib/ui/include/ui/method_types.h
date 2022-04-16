#pragma once

#include <string>
#include <unordered_map>
namespace ui {
enum class Method {
    CPUTenengrad,
    GPUTenengrad,
    CPULaplacian,
    GPULaplacian,
    CPUSMD,
    GPUSMD,
};

const std::unordered_map<Method, std::string> MethodMap = {
    {Method::CPUTenengrad, "CPU Tenengrad"},
    {Method::GPUTenengrad, "GPU Tenengrad"},
    {Method::CPULaplacian, "CPU Laplacian"},
    {Method::GPULaplacian, "GPU Laplacian"},
    {Method::CPUSMD, "CPU SMD"},
    {Method::GPUSMD, "GPU SMD"},
};
} // namespace ui