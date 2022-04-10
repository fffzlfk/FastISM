cmake_minimum_required(VERSION 3.16)

project(demo CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(lib)

add_executable(main src/main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "50;72")

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
target_include_directories(main PRIVATE lib/core/include lib/ui/include)
target_link_libraries(main glfw3 imgui::imgui glad::glad core)
