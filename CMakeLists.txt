cmake_minimum_required(VERSION 3.8)
project(recon CXX)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/YAKL)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/YAKL)
add_subdirectory(${YAKL_HOME}  ${YAKL_BIN})
include(${YAKL_HOME}/yakl_utils.cmake)

add_executable(recon recon.cpp)
yakl_process_target(recon)

