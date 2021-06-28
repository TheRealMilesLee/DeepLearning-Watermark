cmake_minimum_required(VERSION 3.19)
project(PreprocessC___In_CLion)

find_package(OpenCV)
include_directories($(OpenCV_INCLUDE_DIRS))
set(CMAKE_CXX_STANDARD 11)

add_executable(PreprocessC___In_CLion Preprocess.cpp Preprocess.h PreprocessC++.cpp)

target_link_libraries(PreprocessC___In_CLion ${OpenCV_LIBS})