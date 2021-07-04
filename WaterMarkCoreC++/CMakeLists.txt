cmake_minimum_required(VERSION 3.19)
project(WaterMarkCoreC__)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 20)

add_executable(WaterMarkCoreC__ DCT-Watermark_Core.cpp)
target_link_libraries(WaterMarkCoreC__ "${TORCH_LIBRARIES}")
set_property(TARGET WaterMarkCoreC__ PROPERTY CXX_STANDARD 20)