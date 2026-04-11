cmake_minimum_required(VERSION 3.21)
project(HypercubeCNN VERSION 0.1.0 LANGUAGES CXX)

# ——— Build type fallback ———
# Single-config generators (Ninja, Make) default to an empty build type, which
# silently disables every $<CONFIG:Release> generator expression below.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# ——— SDK options ———
option(HCNN_NATIVE_ARCH
    "Tune codegen for the build host (-march=native -mtune=native). \
Disable when building a binary that must run on a different CPU." ON)
option(HCNN_FAST_MATH
    "Enable -ffast-math (relaxes IEEE compliance for speed). \
Disable when downstream code depends on strict NaN/Inf semantics." ON)
option(HCNN_WERROR
    "Treat library warnings as errors." OFF)
option(HCNN_BUILD_EXAMPLES
    "Build the in-tree examples and smoke test (auto: ON when standalone)."
    ${PROJECT_IS_TOP_LEVEL})

# ——— Threading ———
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ——— Shared compile flags (single source of truth) ———
# Library and exes both link this INTERFACE target so flags are defined once.
add_library(hcnn_compile_flags INTERFACE)
target_compile_features(hcnn_compile_flags INTERFACE cxx_std_23)

if(MSVC)
    target_compile_options(hcnn_compile_flags INTERFACE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
        /W4
        $<$<BOOL:${HCNN_WERROR}>:/WX>
    )
else()
    target_compile_options(hcnn_compile_flags INTERFACE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
        $<$<BOOL:${HCNN_WERROR}>:-Werror>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${HCNN_NATIVE_ARCH}>>:-march=native>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${HCNN_NATIVE_ARCH}>>:-mtune=native>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${HCNN_FAST_MATH}>>:-ffast-math>
    )
endif()

# ——— Static library: core pipeline ———
# HCNN.{h,cpp} is the canonical SDK front door.  The underlying layer
# headers are re-exported transitively for power users who need direct
# access to HCNNNetwork / HCNNConv / HCNNPool / HCNNReadout.
set(HCNN_PUBLIC_HEADERS
    HCNN.h
    HCNNNetwork.h
    HCNNConv.h
    HCNNPool.h
    HCNNReadout.h
    ThreadPool.h
)

add_library(HypercubeCNNCore STATIC
    HCNNConv.cpp
    HCNNPool.cpp
    HCNNNetwork.cpp
    HCNNReadout.cpp
    HCNN.cpp
)
add_library(HypercubeCNN::HypercubeCNNCore ALIAS HypercubeCNNCore)

target_compile_features(HypercubeCNNCore PUBLIC cxx_std_23)
set_target_properties(HypercubeCNNCore PROPERTIES
    CXX_EXTENSIONS OFF
    PUBLIC_HEADER "${HCNN_PUBLIC_HEADERS}"
)

target_include_directories(HypercubeCNNCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/HypercubeCNN>
)

target_link_libraries(HypercubeCNNCore
    PUBLIC  Threads::Threads
    # Build-only: compile flags are an internal concern.  Wrapping in
    # $<BUILD_INTERFACE:...> prevents the install export from referencing
    # the helper target so downstream consumers never inherit our flags.
    PRIVATE $<BUILD_INTERFACE:hcnn_compile_flags>
)

# ——— Executables: only when building standalone (not via FetchContent) ———
if(HCNN_BUILD_EXAMPLES)
    enable_testing()

    add_executable(HypercubeCNN  main.cpp)
    add_executable(CoreSmokeTest tests/CoreSmokeTest.cpp)

    # MNISTTrain pulls HCNNDataset directly as a private source -- the
    # dataloader is no longer part of the SDK surface, just an in-tree
    # example utility for this target.
    add_executable(MNISTTrain
        examples/mnist_train.cpp
        dataloader/HCNNDataset.cpp
    )
    target_include_directories(MNISTTrain PRIVATE
        ${PROJECT_SOURCE_DIR}/dataloader
    )

    # RegressionTimeseries: time-series next-step prediction from a
    # synthetic reservoir-like state vector.  Validates HCNN as a
    # scalable regression readout for the workload that drives the
    # HypercubeRC integration (Phase B).  Self-contained -- no
    # external data, no HypercubeRC dependency.
    add_executable(RegressionTimeseries examples/regression_timeseries.cpp)

    set(HCNN_EXE_TARGETS HypercubeCNN MNISTTrain CoreSmokeTest
                         RegressionTimeseries)

    foreach(tgt IN LISTS HCNN_EXE_TARGETS)
        target_link_libraries(${tgt}
            PRIVATE HypercubeCNNCore hcnn_compile_flags
        )
        if(MINGW)
            # Self-contained exe: don't depend on libgcc/libstdc++ DLLs.
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()

    add_test(NAME smoke COMMAND CoreSmokeTest)
endif()

# ——— Install rules ———
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS HypercubeCNNCore
    EXPORT HypercubeCNNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HypercubeCNN
)

install(EXPORT HypercubeCNNTargets
    FILE HypercubeCNNTargets.cmake
    NAMESPACE HypercubeCNN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeCNN
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/HypercubeCNNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeCNN
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeCNN
)
