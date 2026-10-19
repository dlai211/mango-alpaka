cmake_minimum_required(VERSION 3.24)

project(mango-alpaka LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CPython 3.8+ does not search PATH for the dependencies of an extension
# module, so a dynamically linked cudart64_12.dll makes `import` fail with
# "DLL load failed" even though CUDA is on PATH. Rather than link the runtime
# statically (which breaks Alpaka's CMake, it expects the *_static CUDA
# targets), copy the DLL next to the module after the build.
find_package(CUDAToolkit REQUIRED)

option(MANGO_ENABLE_CUDA "Build the CUDA backend" ON)
option(MANGO_ENABLE_OMP2 "Build the OpenMP2-blocks CPU backend" ON)
option(MANGO_ENABLE_SERIAL "Build the serial CPU backend" ON)

# ---------------------------------------------------------------------------
# Alpaka
# ---------------------------------------------------------------------------
# 1.3.0 is required, not cosmetic: 1.0.0 fails to compile its CUDA buffer
# header under nvcc 12.4 ("Idx is not a template"), because the use of
# alpaka::Idx<T> there is unqualified. 1.3.0 qualifies it.
set(alpaka_ACC_GPU_CUDA_ENABLE ${MANGO_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ${MANGO_ENABLE_SERIAL} CACHE BOOL "" FORCE)
set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ${MANGO_ENABLE_OMP2} CACHE BOOL "" FORCE)
set(alpaka_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/alpaka-1.3.0 alpaka_build)

# ---------------------------------------------------------------------------
# pybind11: prefer an explicit -Dpybind11_DIR, else ask the interpreter
# ---------------------------------------------------------------------------
if(NOT DEFINED MANGO_PYTHON)
    set(MANGO_PYTHON "python" CACHE STRING "Interpreter used to locate pybind11")
endif()

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${MANGO_PYTHON}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _mango_pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_mango_pybind11_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_mango_pybind11_dir}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Install it (pip install pybind11) or pass "
            "-Dpybind11_DIR=<python -m pybind11 --cmakedir>.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
# One target with two sources is deliberate. CMake compiles bindings.cpp with
# MSVC and launcher.cu with nvcc, then links both with the C++ driver. Keeping
# pybind11 out of the nvcc translation unit avoids the classic nvcc/pybind11
# flag conflicts, and keeping Alpaka out of bindings.cpp means Brown can
# include the interface header without a GPU toolchain.
pybind11_add_module(mango_alpaka_ext MODULE
    src/bindings.cpp
    src/launcher.cu
)

target_include_directories(mango_alpaka_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mango_alpaka_ext PRIVATE alpaka::alpaka)

if(MANGO_ENABLE_OMP2)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mango_alpaka_ext PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(mango_alpaka_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# Put cudart beside the .pyd so `import` works without touching the caller's
# PATH (see the note at the top of this file). CUDA::cudart is the import
# library, not the DLL, so locate the runtime DLL by name.
file(GLOB _mango_cudart_dlls "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
if(NOT _mango_cudart_dlls)
    message(WARNING "cudart64_*.dll not found in ${CUDAToolkit_BIN_DIR}; "
                    "the module may fail to import unless CUDA is on PATH")
endif()
foreach(_mango_dll IN LISTS _mango_cudart_dlls)
    add_custom_command(
        TARGET mango_alpaka_ext POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_mango_dll}"
            "$<TARGET_FILE_DIR:mango_alpaka_ext>"
        COMMENT "Copying ${_mango_dll} next to mango_alpaka_ext"
    )
endforeach()

message(STATUS "mango-alpaka: CUDA=${MANGO_ENABLE_CUDA} OMP2=${MANGO_ENABLE_OMP2} SERIAL=${MANGO_ENABLE_SERIAL}")
