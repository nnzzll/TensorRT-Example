cmake_minimum_required(VERSION 3.0.0)
project(VTKTEST VERSION 0.1.0)

include(CTest)
enable_testing()

# VTK
set(VTK_DIR "/home/jianjia/VTK-9.0.3/build")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# TensorRT
include_directories("./include")
set(TENSORRT_LIB_PATH "/home/jianjia/TensorRT-7.2.3.4/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# CUDA
find_package(CUDA 11.1 REQUIRED)
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})


add_executable(VTKTEST main.cpp)
target_link_libraries(VTKTEST ${VTK_LIBRARIES} ${CUDA_LIBRARIES} ${LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
