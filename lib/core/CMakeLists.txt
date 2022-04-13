add_library(core INTERFACE)
find_package(OpenCV CONFIG REQUIRED)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(core INTERFACE ${OpenCV_LIBS} utils)