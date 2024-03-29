#pragma once

#include "buttons.h"
#include "file_chooser.h"
#include "method_types.h"
#include <cstddef>
#include <elements.hpp>

namespace ui {
using namespace cycfi::elements;

auto constexpr bkd_color = rgba(35, 35, 35, 255);
auto background = box(bkd_color);

std::vector<std::string> files;

// 绘制文件列表项目
auto draw_cell(const size_t index) {
    auto text = files[index];
    return align_left(label(text));
}

int run(int argc, char *argv[]) {
    app _app(argc, argv, "demo", "com.xdu.demo");
    window _win(_app.name());
    _win.on_close = [&_app]() { _app.stop(); };
    view _view(_win);

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
            _view.refresh();
        }
    };

    auto clear_button = button("Clear");
    clear_button.on_click = [&](bool) {
        list_size = 1;
        files.resize(1);
        ptr_list.resize(list_size);
        content.resize(list_size);
        _view.refresh();
    };

    constexpr view_limits files_limits = {{400, 200},
                                          {full_extent, full_extent}};

    _view.content(
        scale(1.5,
              margin(
                  {10, 10, 10, 10},
                  vtile(htile(left_margin(
                                  10, limit(files_limits,
                                            vscroller(hold(share(linked))))),
                              left_margin(
                                  10,
                                  valign(0.5,
                                         vtile(top_margin(10, add_button),
                                               top_margin(10, clear_button))))),
                        margin({10, 10, 10, 10}, make_buttons(_view, files))))),
        background);
    _app.run();
    return 0;
}
} // namespace ui