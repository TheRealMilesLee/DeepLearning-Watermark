cmake_minimum_required(VERSION 3.19)
project(WaterMarkCoreC__)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 20)

add_executable(WaterMarkCore_TorchC DCT-Watermark_Core.cpp DCT_Watermark.cpp DCT_Watermark.h)
target_link_libraries(WaterMarkCore_TorchC "${TORCH_LIBRARIES}")
set_property(TARGET WaterMarkCore_TorchC PROPERTY CXX_STANDARD 20)