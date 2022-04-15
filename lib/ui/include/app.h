#pragma once

#include "dialog.h"
#include "file_chooser.h"
#include "method_types.h"
#include <cstddef>
#include <elements.hpp>

namespace ui {
using namespace cycfi::elements;

auto constexpr bkd_color = rgba(35, 35, 35, 255);
auto background = box(bkd_color);

std::vector<std::string> files;

auto draw_cell(const size_t index) {
    auto text = files[index];
    return align_left(label(text));
}

int run(int argc, char *argv[]) {
    app _app(argc, argv, "demo", "com.xdu.demo");
    window _win(_app.name());
    _win.on_close = [&_app]() { _app.stop(); };
    view view_(_win);

    files.emplace_back("-----");
    size_t list_size = 1;

    std::vector<element_ptr> ptr_list;
    ptr_list.resize(list_size, nullptr);

    auto &&make_cell = [&](size_t index) {
        if (ptr_list[index].get() == nullptr)
            ptr_list[index] = share(draw_cell(index));
        return ptr_list[index];
    };

    auto cp = basic_vertical_cell_composer(list_size, make_cell);
    auto content = vdynamic_list(cp);
    auto linked = link(content);

    auto add_button = button("Add");
    add_button.on_click = [&](bool) {
        if (auto filepaths = file_chooser(); filepaths.has_value()) {
            files.insert(files.end(), filepaths.value().begin(),
                         filepaths.value().end());
            list_size += filepaths.value().size();
            ptr_list.resize(list_size);
            content.resize(list_size);
            view_.refresh();
        }
    };

    auto open_button = button("Laplacian");
    open_button.on_click = [&](bool) {
        auto dialog = make_dialog(view_, Method::Laplacian, files);
        view_.add(dialog);
        view_.refresh();
    };

    constexpr view_limits files_limits = {{400, 200},
                                          {full_extent, full_extent}};

    view_.content(
        margin(
            {10, 10, 10, 10},
            vtile(htile(left_margin(10, limit(files_limits,
                                              vscroller(hold(share(linked))))),
                        left_margin(10, add_button)),
                  margin({10, 10, 10, 10}, open_button))),
        background);
    _app.run();
    return 0;
}
} // namespace ui