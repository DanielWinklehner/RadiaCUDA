# cpp/CMakeLists.txt

# ---------- Collect sources ----------
set(RAD_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(RAD_LIB_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src/lib)
set(RAD_EXT_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src/ext)

file(GLOB RAD_CORE_SRC   ${RAD_CORE_DIR}/*.cpp)
file(GLOB RAD_LIB_SRC    ${RAD_LIB_DIR}/*.cpp)
set(RAD_AUXPARSE_SRC      ${RAD_EXT_DIR}/auxparse/auxparse.cpp)
set(RAD_GENMATH_SRC
    ${RAD_EXT_DIR}/genmath/gmtrans.cpp
    ${RAD_EXT_DIR}/genmath/gmfft.cpp
    ${RAD_EXT_DIR}/genmath/gmmeth.cpp
    ${RAD_EXT_DIR}/genmath/gminterp.cpp
    ${RAD_EXT_DIR}/genmath/gmfit.cpp
)
set(RAD_TRIANGLE_SRC      ${RAD_EXT_DIR}/triangle/triangle.c)

# ---------- Static library: radia_core ----------
add_library(radia_core STATIC
    ${RAD_CORE_SRC}
    ${RAD_LIB_SRC}
    ${RAD_AUXPARSE_SRC}
    ${RAD_GENMATH_SRC}
    ${RAD_TRIANGLE_SRC}
)

target_include_directories(radia_core PUBLIC
    ${RAD_CORE_DIR}
    ${RAD_LIB_DIR}
    ${RAD_EXT_DIR}/auxparse
    ${RAD_EXT_DIR}/genmath
    ${RAD_EXT_DIR}/triangle
    ${FFTW3_INCLUDE_DIR}
)

target_compile_definitions(radia_core PUBLIC
    FFTW_ENABLE_FLOAT
    NO_TIMER
    ANSI_DECLARATORS
    TRILIBRARY
    _GM_WITHOUT_BASE
    ALPHA__LIB__
)

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(radia_core PUBLIC _WINDOWS)
    target_compile_definitions(radia_core PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
else()
    target_compile_definitions(radia_core PUBLIC LINUX)
endif()

# MSVC-specific flags
if(MSVC)
    target_compile_options(radia_core PRIVATE
        /W3 /wd4996 /wd4244 /wd4267 /wd4305
    )
    # Suppress warnings in triangle.c more aggressively
    set_source_files_properties(${RAD_TRIANGLE_SRC} PROPERTIES
        COMPILE_OPTIONS "/W0"
    )
else()
    target_compile_options(radia_core PRIVATE
        -O3 -Wall
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-misleading-indentation
        -Wno-parentheses
        -Wno-comment
        -Wno-unknown-pragmas
        -Wno-stringop-truncation
        -Wno-char-subscripts
    )
endif()

# Link FFTW
target_link_libraries(radia_core PUBLIC ${FFTW3F_LIB})

# MPI
if(RADIA_WITH_MPI)
    target_compile_definitions(radia_core PUBLIC _WITH_MPI)
    target_link_libraries(radia_core PUBLIC MPI::MPI_CXX)
endif()

# Math library (Unix only)
if(UNIX)
    target_link_libraries(radia_core PUBLIC m pthread dl)
endif()

# ---------- Python extension: radia ----------
Python_add_library(radia MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clients/python/radpy.cpp
)

target_link_libraries(radia PRIVATE radia_core)

target_include_directories(radia PRIVATE
    ${RAD_LIB_DIR}
    ${RAD_EXT_DIR}/auxparse
    ${RAD_CORE_DIR}
)

# Install the extension into the wheel
install(TARGETS radia DESTINATION .)