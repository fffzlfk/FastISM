#pragma once

#include <string>
#include <unordered_map>
namespace ui {
enum class Method {
    Tenengrad,
    Laplacian,
    SMD,
};

const std::unordered_map<Method, std::string> MethodMap = {
    {Method::Tenengrad, "Tenengrad"},
    {Method::Laplacian, "Laplacian"},
    {Method::SMD, "SMD"},
};
} // namespace ui