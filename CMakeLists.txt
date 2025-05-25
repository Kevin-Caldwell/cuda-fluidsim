cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(cuda_fluidsim LANGUAGES CUDA CXX)


include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB srcs "${PROJECT_SOURCE_DIR}/src/*")

add_executable(cuda_fluidsim ${srcs} main.cu)
add_executable(ppm_handler ${srcs} ppm_main.cpp)

add_compile_options(-allow-unsupported-compiler)

set_property(TARGET cuda_fluidsim
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  "${PROJECT_SOURCE_DIR}/include/Common"
  "${PROJECT_SOURCE_DIR}/include/"
)

if(APPLE)
  set_property( TARGET cuda_fluidsim
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
