#pragma once

#include <elements.hpp>
#include "method_types.h"
#include "dialog.h"

namespace ui {
using namespace cycfi::elements;
auto make_buttons(view &_view, const std::vector<std::string> &files) {
    auto laplacian_button = button("Laplacian");
    laplacian_button.on_click = [&](bool) {
        auto dialog = make_dialog(_view, Method::Laplacian, files);
        _view.add(dialog);
        _view.refresh();
    };
    auto tenengrad_button = button("Tenengrad");
    tenengrad_button.on_click = [&](bool) {
        auto dialog = make_dialog(_view, Method::Tenengrad, files);
        _view.add(dialog);
        _view.refresh();
    };
    auto smd_button = button("SMD");
    smd_button.on_click = [&](bool) {
        auto dialog = make_dialog(_view, Method::SMD, files);
        _view.add(dialog);
        _view.refresh();
    };
    return htile(left_margin(10, laplacian_button),
                 left_margin(10, tenengrad_button),
                 left_margin(10, smd_button));
}
} // namespace ui