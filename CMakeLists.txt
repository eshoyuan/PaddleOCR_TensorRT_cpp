cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# add_compile_options(-fno-elide-constructors) 

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR "/usr/yyx/tensorrt/TensorRT/")
# TODO: Specify the path to cuda root dir
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# TODO: Specify the path to opencv root dir
set(OpenCV_DIR "/home/opencv-3.4.15")

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(demo src/main.cpp)
target_link_libraries(demo tensorrt_cpp_api)
