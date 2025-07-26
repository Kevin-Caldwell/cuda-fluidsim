cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(cuda_fluidsim LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# add_subdirectory(src)
# add_subdirectory(tests)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB srcs "${PROJECT_SOURCE_DIR}/src/*")

add_library(fsim STATIC ${srcs})
add_executable(cuda_fluidsim main.cu)
add_executable(ppm_handler ppm_main.cpp)

target_link_libraries(cuda_fluidsim fsim)
target_link_libraries(ppm_handler fsim)

# add_compile_options(-allow-unsupported-compiler)

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

# Google Tests
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB test_src "${PROJECT_SOURCE_DIR}/tests/*")

add_executable(fsim_test ${test_src})

target_link_libraries(fsim_test PRIVATE
    gtest
    gtest_main
    fsim
)

include(GoogleTest)
gtest_discover_tests(fsim_test)

set_property(TARGET fsim_test
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
