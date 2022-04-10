add_library(ui INTERFACE)
find_package(imgui CONFIG REQUIRED)
target_link_libraries(ui INTERFACE imgui::imgui)
