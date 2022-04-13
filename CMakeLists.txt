cmake_minimum_required(VERSION 3.16)

project(demo CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION ON)
add_compile_options(-Wall -Wextra)

add_subdirectory(lib)
add_subdirectory(elements)
add_subdirectory(nativefiledialog-extended)

add_executable(main src/main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "50;72")
target_link_libraries(main PRIVATE elements core nfd)
