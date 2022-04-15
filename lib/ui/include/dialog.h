#pragma once

#include "compute.h"
#include "method_types.h"
#include <elements.hpp>

namespace ui {
using namespace cycfi::elements;

auto dialog_content(Method method, const std::vector<std::string> &files) {
    vtile_composite comp;

    for (size_t i = 1; i < files.size(); i++) {
        auto [time, res] = compute(method, files, i);
        comp.push_back(
            share(label(std::to_string(res) + "     " + std::to_string(time))));
    }

    return simple_heading(margin({10, 10, 10, 10}, comp), "The Thraxian Legacy",
                          1.1);
}

auto make_dialog(view &_view, Method method,
                 const std::vector<std::string> &files) {
    auto &&on_ok = [&_view]() {
        // close the dialog
    };
    auto dialog = dialog1(_view, dialog_content(method, files), on_ok);
    return dialog;
}
} // namespace ui