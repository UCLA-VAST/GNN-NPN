cmake_minimum_required(VERSION 3.18)
project(gnn_npn_wrapper LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- pybind11 -----------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- mockturtle (header-only INTERFACE target, defines `mockturtle`) ----------
set(MOCKTURTLE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MOCKTURTLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MOCKTURTLE_BUILD_EXPERIMENTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/mockturtle)

# --- the pybind11 wrapper -----------------------------------------------------
add_subdirectory(wrapper)
