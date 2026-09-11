# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 David Liptak
#
# Python extension for HypercubeCNN. Invoked by scikit-build-core from the
# repo-root pyproject.toml (cmake.source-dir = "python").
#
# Core + train-helpers C++ sources are compiled into _core (PIC). Do not
# reconfigure CLion cmake-build-* directories from this tree.

cmake_minimum_required(VERSION 3.21)
project(HypercubeCNNPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Portable vs local arch. cibuildwheel sets x86-64-v2 / none; local default native.
set(HYPERCUBE_ARCH "native" CACHE STRING
    "Target architecture for -march: native | x86-64-v2 | none")

# Match C++ library default (docs/CPP_SDK.md). Wheels document this as ON.
option(HCNN_FAST_TANH
    "Padé tanh in HCNNConv (same as HypercubeCNNCore default)." ON)

set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/HCNN.cpp
    ${CORE_DIR}/HCNNConv.cpp
    ${CORE_DIR}/HCNNPool.cpp
    ${CORE_DIR}/HCNNNetwork.cpp
    ${CORE_DIR}/HCNNReadout.cpp
    ${CORE_DIR}/HCNNTrainHelpers.cpp
    ${CORE_DIR}/HCNNSpatialAug.cpp
    ${CORE_DIR}/HCNNSpatialEmbed.cpp
)

pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})

target_include_directories(_core PRIVATE ${CORE_DIR})
target_link_libraries(_core PRIVATE Threads::Threads)

target_compile_definitions(_core PRIVATE
    $<$<BOOL:${HCNN_FAST_TANH}>:HCNN_FAST_TANH>
)

if(MSVC)
    target_compile_options(_core PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(_core PRIVATE
        -O3 -Wall -Wextra -Wno-unknown-pragmas
        # Relaxed float flags aligned with C++ HCNN_FAST_MATH spirit (not full
        # -ffast-math / associative-math).
        -fno-math-errno -ffinite-math-only -fno-signed-zeros -fno-trapping-math
    )
    if(HYPERCUBE_ARCH STREQUAL "native")
        target_compile_options(_core PRIVATE -march=native -mtune=native)
    elseif(HYPERCUBE_ARCH STREQUAL "x86-64-v2")
        target_compile_options(_core PRIVATE -march=x86-64-v2 -mtune=generic)
    endif()
    # "none" = no -march (portable wheels, ARM, MSVC-style portable builds)
endif()

# MinGW: static-link libgcc/libstdc++. Prefer dynamic winpthread and ship
# libwinpthread-1.dll next to the extension (ThreadPool / std::thread).
# Full `-static` and `-Bstatic pthread` both fail on CLion MinGW 15.2
# (undefined __intrinsic_setjmpex / __ms_vsnprintf).
if(MINGW)
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    find_file(WINPTHREAD_DLL libwinpthread-1.dll PATHS ENV PATH NO_DEFAULT_PATH)
    if(WINPTHREAD_DLL)
        install(FILES ${WINPTHREAD_DLL} DESTINATION hypercube_cnn)
    endif()
endif()

install(TARGETS _core DESTINATION hypercube_cnn)
