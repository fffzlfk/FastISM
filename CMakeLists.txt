cmake_minimum_required(VERSION 3.16)

project(demo CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

include_directories(include)

add_executable(Tenengrad src/Tenengrad.cu)
# add_executable(TenengradReduce src/TenengradReduce.cu)
# add_executable(Laplacian src/Laplacian.cu)
# add_executable(SMDReduce src/SMDReduce.cu)
# add_executable(Old src/Old.cu)
# add_executable(Reduce src/Reduce.cu)
add_executable(SMD src/SMD.cu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Tenengrad ${OpenCV_LIBS})
# target_link_libraries(TenengradReduce ${OpenCV_LIBS})
# target_link_libraries(Laplacian ${OpenCV_LIBS})
# target_link_libraries(SMDReduce ${OpenCV_LIBS})
# target_link_libraries(Old ${OpenCV_LIBS})
target_link_libraries(SMD ${OpenCV_LIBS})
