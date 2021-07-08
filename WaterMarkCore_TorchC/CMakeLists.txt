cmake_minimum_required(VERSION 3.19)
project(WaterMarkCoreC__)
set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(WaterMarkCore_TorchC DCT-Watermark_Core.cpp DCT_Watermark.cpp DCT_Watermark.h)



target_link_libraries(WaterMarkCore_TorchC "${OpenCV_LIBS} 
${TORCH_LIBRARIES}")

