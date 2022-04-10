add_library(core INTERFACE)
find_package(OpenCV CONFIG REQUIRED)
target_include_directories(core INTERFACE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/lib/utils/include)
target_link_libraries(core INTERFACE ${OpenCV_LIBS})