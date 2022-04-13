#include <elements.hpp>
#include <nfd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void file_chooser() {
    NFD_Init();

    nfdchar_t *outPath;
    nfdfilteritem_t filterItem[] = {{"Image", "png,jpg,bmp"}};
    nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, NULL);
    if (result == NFD_OKAY) {
        puts("Success!");
        puts(outPath);
        NFD_FreePath(outPath);
    } else if (result == NFD_CANCEL) {
        puts("User pressed cancel.");
    } else {
        printf("Error: %s\n", NFD_GetError());
    }

    NFD_Quit();
}

using namespace cycfi::elements;

auto constexpr bkd_color = rgba(35, 35, 37, 255);
auto background = box(bkd_color);

auto make_button() {
    auto add_button = button("Add");
    add_button.on_click = [](auto) { file_chooser(); };
    return add_button;
}

int main(int argc, char *argv[]) {
    app _app(argc, argv, "demo", "com.xdu.demo");
    window _win(_app.name());
    _win.on_close = [&_app]() { _app.stop(); };

    view view_(_win);
    view_.content(make_button(), background);

    _app.run();
    return 0;
}
