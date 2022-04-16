#pragma once

#include <nfd.hpp>
#include <optional>
#include <string>
#include <vector>

namespace ui {
std::optional<std::vector<std::string>> file_chooser() {
    NFD::Guard nfdGuard;

    // auto-freeing memory
    NFD::UniquePathSet outPaths;

    // prepare filters for the dialog
    nfdfilteritem_t filterItem[1] = {{"Image", "png,jpg,jepg,bmp"}};

    // show the dialog
    nfdresult_t result = NFD::OpenDialogMultiple(outPaths, filterItem, 1);
    if (result == NFD_OKAY) {
        nfdpathsetsize_t num_paths;
        NFD::PathSet::Count(outPaths, num_paths);
        std::vector<std::string> file_names;
        for (size_t i = 0; i < num_paths; i++) {
            NFD::UniquePathSetPath path;
            NFD::PathSet::GetPath(outPaths, i, path);
            file_names.emplace_back(path.get());
        }
        std::cout << "Success!" << std::endl;

        return file_names;
    } else if (result == NFD_CANCEL) {
        std::cout << "User pressed cancel." << std::endl;
    } else {
        std::cout << "Error: " << NFD::GetError() << std::endl;
    }
    return {};
}
} // namespace ui