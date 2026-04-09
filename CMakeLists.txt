cmake_minimum_required(VERSION 3.21)
project(HypercubeCNN VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ——— Threading ———
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ——— Static library: core pipeline ———
# HCNN.{h,cpp} is the canonical SDK front door.  The underlying layer
# headers are re-exported transitively for power users who need direct
# access to HCNNNetwork / HCNNConv / HCNNPool / HCNNReadout.
add_library(HypercubeCNNCore STATIC
    HCNNConv.cpp
    HCNNPool.cpp
    HCNNNetwork.cpp
    HCNNReadout.cpp
    HCNN.cpp
)

set(PUBLIC_HEADERS
    HCNN.h
    HCNNNetwork.h
    HCNNConv.h
    HCNNPool.h
    HCNNReadout.h
    ThreadPool.h
)

target_include_directories(HypercubeCNNCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/HypercubeCNN>
)

target_link_libraries(HypercubeCNNCore PUBLIC Threads::Threads)

# ——— Optimization and warning flags (library) ———
if(MSVC)
    target_compile_options(HypercubeCNNCore PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
        /W4
    )
else()
    target_compile_options(HypercubeCNNCore PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
    )
endif()

if(MINGW)
    target_link_options(HypercubeCNNCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Executables: only when building standalone (not via FetchContent) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(HypercubeCNN main.cpp)
    # MNISTTrain pulls HCNNDataset directly as a private source -- the
    # dataloader is no longer part of the SDK surface, just an in-tree
    # example utility for this target.
    add_executable(MNISTTrain examples/mnist_train.cpp dataloader/HCNNDataset.cpp)
    add_executable(CoreSmokeTest tests/CoreSmokeTest.cpp)
    set(EXE_TARGETS HypercubeCNN MNISTTrain CoreSmokeTest)

    foreach(tgt IN LISTS EXE_TARGETS)
        target_link_libraries(${tgt} PRIVATE HypercubeCNNCore)
    endforeach()

    foreach(tgt IN LISTS EXE_TARGETS)
        if(MSVC)
            target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:/O2>
                $<$<CONFIG:Debug>:/Od /Zi>
                /W4
            )
        else()
            target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
                $<$<CONFIG:Debug>:-O0 -g>
                -Wall -Wextra -Wno-unknown-pragmas
            )
        endif()

        if(MINGW)
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()
endif()

# ——— Install rules ———
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS HypercubeCNNCore
    EXPORT HypercubeCNNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HypercubeCNN
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
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeCNNConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeCNN
)
