add_library(ui INTERFACE)
target_include_directories(ui INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ui INTERFACE elements core nfd)