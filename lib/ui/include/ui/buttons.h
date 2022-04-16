#pragma once

#include "dialog.h"
#include "method_types.h"
#include <elements.hpp>

namespace ui {
using namespace cycfi::elements;

auto make_button(view &_view, Method method,
                 const std::vector<std::string> &files) {
    auto on_click = [&_view, &files, method](bool) {
        auto dialog = make_dialog(_view, method, files);
        std::cout << MethodMap.at(method) << std::endl;
        _view.add(dialog);
        _view.refresh();
    };
    auto _button = button(MethodMap.at(method));
    _button.on_click = on_click;
    return _button;
}

auto make_buttons(view &_view, const std::vector<std::string> &files) {
    auto cpu_tenengrad_button = make_button(_view, Method::CPUTenengrad, files);
    auto gpu_tenengrad_button = make_button(_view, Method::GPUTenengrad, files);
    auto cpu_laplacian_button = make_button(_view, Method::CPULaplacian, files);
    auto gpu_laplacian_button = make_button(_view, Method::GPULaplacian, files);
    auto cpu_smd_button = make_button(_view, Method::CPUSMD, files);
    auto gpu_smd_button = make_button(_view, Method::GPUSMD, files);

    return htile(left_margin(10, cpu_tenengrad_button),
                 left_margin(10, gpu_tenengrad_button),
                 left_margin(10, cpu_laplacian_button),
                 left_margin(10, gpu_laplacian_button),
                 left_margin(10, cpu_smd_button),
                 left_margin(10, gpu_smd_button));
}
} // namespace ui