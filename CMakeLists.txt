# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(Radia LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------- Options ----------
option(RADIA_WITH_MPI  "Enable MPI support"  ON)
option(RADIA_WITH_CUDA "Enable CUDA support" ON)

# ---------- Dependencies ----------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# FFTW3 single-precision (fftwf) from conda
find_library(FFTW3F_LIB NAMES fftw3f fftwf REQUIRED)
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h REQUIRED)

if(RADIA_WITH_MPI)
    if(WIN32)
        # Intel MPI from conda on Windows
        set(MPI_CXX_ADDITIONAL_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/Library/include")
        set(MPI_CXX_LIB_NAMES "impi" "impicxx")
        set(MPI_impi_LIBRARY "$ENV{CONDA_PREFIX}/Library/lib/impi.lib")
        set(MPI_impicxx_LIBRARY "$ENV{CONDA_PREFIX}/Library/lib/impicxx.lib")
        set(MPI_CXX_HEADER_DIR "$ENV{CONDA_PREFIX}/Library/include")
    endif()
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(RADIA_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(cpp)