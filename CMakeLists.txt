cmake_minimum_required(VERSION 4.1)
project(HypercubeCNN VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# ——— Static library: core pipeline ———
add_library(HypercubeCNNCore STATIC
    ${PROJECT_SOURCE_DIR}/placeholder.cpp
)

# Public headers — consumers need these via target_link_libraries
set(PUBLIC_HEADERS
    # Add public headers here as they are created
)

# Include directories: PROJECT_SOURCE_DIR for in-tree builds,
# install prefix include/HypercubeCNN for installed consumers
target_include_directories(HypercubeCNNCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/HypercubeCNN>
)

# ——— Optimization and warning flags (library) ———
target_compile_options(HypercubeCNNCore PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
)

if(MINGW)
    target_link_options(HypercubeCNNCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Executables: only when building standalone (not via FetchContent) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(HypercubeCNN main.cpp)

    set(DIAGNOSTIC_TARGETS HypercubeCNN)
    set(EXAMPLE_TARGETS)
    set(TEST_TARGETS)
    set(EXE_TARGETS ${DIAGNOSTIC_TARGETS} ${EXAMPLE_TARGETS} ${TEST_TARGETS})

    foreach(tgt IN LISTS EXE_TARGETS)
        target_link_libraries(${tgt} PRIVATE HypercubeCNNCore)
    endforeach()

    # ——— OpenMP (diagnostics only) ———
    # Core library does NOT use OpenMP.  Diagnostic executables use it for
    # parallel sweeps and benchmarking.
    if(MINGW)
        foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
            target_compile_options(${tgt} PRIVATE -fopenmp)
            target_link_options(${tgt} PRIVATE -fopenmp)
        endforeach()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
                target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
            endforeach()
        endif()
    elseif(MSVC)
        foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
            target_compile_options(${tgt} PRIVATE /openmp)
        endforeach()
    endif()

    foreach(tgt IN LISTS EXE_TARGETS)
        target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
                $<$<CONFIG:Debug>:-O0 -g>
                -Wall -Wextra -Wno-unknown-pragmas
        )

        if(MINGW)
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()
endif()

# ——— Install rules ———
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the static library
install(TARGETS HypercubeCNNCore
    EXPORT HypercubeCNNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install public headers into include/HypercubeCNN/
install(FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HypercubeCNN
)

# Generate and install CMake package config
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
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeCNN
)
