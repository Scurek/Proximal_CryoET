cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME CUDA_PROXIMAL_SART)

ADD_DEFINITIONS(-DUSE_CUDA)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES src/main.cu src/proximal_iteration.h src/linearized_admm.h src/admm.h src/primal_dual.h src/data_term_proximal_operator.h src/global_definitions.h src/kernel_functions.h src/tv_proximal_operator.h src/huber_proximal_operator.h src/NLM.h src/MRCParser.h)

add_executable(${PROJECT_NAME} ${SOURCES})

# Accept CUDA_ARCHITECTURES from CMake (e.g., -DCUDA_ARCHITECTURES=80 from Dockerfile)
# Default to "native" if not specified
set(CUDA_ARCH_VALUE "native" CACHE STRING "CUDA Compute Capability")
if(DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCH_VALUE "${CUDA_ARCHITECTURES}")
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_VALUE})

# This might work for older CMake versions
# if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
#     set_property(TARGET ${PROJECT_NAME}
#         PROPERTY CUDA_ARCHITECTURES native
#     )
# else()
#     execute_process(
#         COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#         OUTPUT_VARIABLE CUDA_ARCH_RAW
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#         ERROR_QUIET
#     )

#     # Take first line if multiple GPUs
#     string(REPLACE "\n" ";" CUDA_ARCH_LIST "${CUDA_ARCH_RAW}")
#     list(GET CUDA_ARCH_LIST 0 CUDA_ARCH)

#     string(REPLACE "." "" CUDA_ARCH "${CUDA_ARCH}")

#     if(CUDA_ARCH)
#         set_property(TARGET ${PROJECT_NAME}
#             PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH}
#         )
#     endif()
# endif()

find_package(CUDAToolkit REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC src)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)

    message(STATUS "X11_FOUND = ${X11_FOUND}")
    message(STATUS "X11_INCLUDE_DIR = ${X11_INCLUDE_DIR}")
    message(STATUS "X11_LIBRARIES = ${X11_LIBRARIES}")

    target_include_directories(${PROJECT_NAME} PUBLIC ${X11_INCLUDE_DIR})

    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
endif()
