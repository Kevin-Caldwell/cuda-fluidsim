cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_fluidsim LANGUAGES CXX CUDA)

# include(CTest)

file(GLOB srcs "${PROJECT_SOURCE_DIR}/src/*")

add_executable(cuda_fluidsim ${srcs} main.cu)
add_executable(ppm_handler ${srcs} ppm_main.cpp)

set_property(TARGET cuda_fluidsim 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  "include/Common"
  "${PROJECT_SOURCE_DIR}/include/"
  )

if(APPLE)
  set_property( TARGET cuda_fluidsim
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
