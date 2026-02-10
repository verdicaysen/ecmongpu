cmake_minimum_required(VERSION 3.21)

# HIP setup - must be before project() so enable_language(HIP) works
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

project(co-ecm C CXX HIP)

set(VERSION "v0.8-hip")
set(CMAKE_BUILD_TYPE Release)

set(BUILD_BENCHMARKS 1)

find_package(hip REQUIRED)

# Options and Settings

# Log level: VERBOSE, DEBUG, INFO, WARNING, ERROR, FATAL
set(LOG_LEVEL INFO)

# Multi precision limbs, use 32 or 64 bit datatype
set(LIMB_BITS 32)

# Bits of basic multi precision datatype. Has to be 32 bit more than the modulus to be factored
if(DEFINED ENV{BITWIDTH})
	set(BITWIDTH $ENV{BITWIDTH})
else()
	set(BITWIDTH 192)
endif()
message("Building for ${BITWIDTH}-bit moduli")

# Default threads per block.
set(BLOCK_SIZE 128)

# Curves per single batch.
if(DEFINED ENV{BATCH_JOB_SIZE})
	set(BATCH_JOB_SIZE $ENV{BATCH_JOB_SIZE})
else()
	set(BATCH_JOB_SIZE 32768)
endif()

# Window size for w-NAF.
if(DEFINED ENV{WINDOW_SIZE})
  set(NAF_WINDOW_SIZE $ENV{WINDOW_SIZE})
else()
  set(NAF_WINDOW_SIZE 4)
endif()
message("Building with window size w=${NAF_WINDOW_SIZE}")

# Default allocated number of NAF digits for ECM stage 2.
set(NAF_STAGE2_DEFAULT_DIGITS 20)

# Use optimized precomputation
if(NOT DEFINED ENV{DISABLE_OPTIMIZED_PRECOMP})
  set(OPTIMIZE_PRECOMP 1)
endif()
if(NOT DEFINED OPTIMIZE_PRECOMP)
  message("Optimized point representation disabled")
endif()

# Choose a Montgomery product algorithm
if(DEFINED ENV{MON_PROD})
  if("$ENV{MON_PROD}" STREQUAL "CIOS")
    set(MON_PROD_CIOS 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "CIOS_XMAD")
    set(MON_PROD_CIOS_XMAD 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "FIPS")
    set(MON_PROD_FIPS 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "FIOS")
    set(MON_PROD_FIOS 1)
  endif()
else ()
  set(MON_PROD_FIPS 1)
endif()


set(COORDINATES_EXTENDED 1)
#set(COORDINATES_INVERTED 1)

# Set GPU architecture for HIP/ROCm
if(DEFINED ENV{GPU_ARCH})
    set(GPU_ARCHITECTURE $ENV{GPU_ARCH})
    message("Building for GPU architecture ${GPU_ARCHITECTURE}")
else()
  # Auto-detect using rocminfo, fallback to gfx1201 (RX 9070 XT)
  execute_process(
    COMMAND bash -c "rocminfo 2>/dev/null | grep -m1 'gfx[0-9]' | sed 's/.*\\(gfx[0-9a-z]*\\).*/\\1/'"
    OUTPUT_VARIABLE GPU_ARCHITECTURE
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if("${GPU_ARCHITECTURE}" STREQUAL "")
    set(GPU_ARCHITECTURE "gfx1201")
    message("Could not detect GPU architecture, using default: ${GPU_ARCHITECTURE}")
  else()
    message("Detected GPU architecture: ${GPU_ARCHITECTURE}")
  endif()
endif()

# Set CMake HIP architecture (required by CMake's HIP support)
set(CMAKE_HIP_ARCHITECTURES ${GPU_ARCHITECTURE})


# Build setup
# Do not edit

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(GMP REQUIRED)

# Compute MAX_SCALAR_LIMBS for build_config.h
math(EXPR MAX_SCALAR_LIMBS "${BITWIDTH} / ${LIMB_BITS}")

include(CTest)

# Generate version.h
include(${CMAKE_MODULE_PATH}/version.cmake)

# Generate build_config.h
configure_file(
        include/build_config.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/build_config.h
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)


# Concatenate all .cu files into a single HIP source file.
# Same strategy as the original CUDA build: all GPU code in one TU
# for better inlining and optimization.
function(add_cuda_executable TARGET)
    set(CURRENT_C_SOURCES ${ARGN})
    set(CURRENT_HIP_SOURCES ${ARGN})
    list(FILTER CURRENT_C_SOURCES EXCLUDE REGEX ".*\\.cu")
    list(FILTER CURRENT_HIP_SOURCES INCLUDE REGEX ".*\\.cu")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

    # Concatenate all .cu files into a single .hip.cpp file
    add_custom_command(
            OUTPUT generated/${TARGET}-hipkernel.hip.cpp
            COMMAND cat ${CURRENT_HIP_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET}-hipkernel.hip.cpp
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS ${CURRENT_HIP_SOURCES}
    )

    add_executable(${TARGET} ${CURRENT_C_SOURCES} generated/${TARGET}-hipkernel.hip.cpp)

    # Mark concatenated source as HIP language
    set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET}-hipkernel.hip.cpp
        PROPERTIES LANGUAGE HIP
    )

    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_ARCHITECTURE} -Wall -Wno-unknown-pragmas>
    )

    target_link_libraries(${TARGET} PRIVATE hip::device)
endfunction()


# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

find_package(Threads)
link_libraries(gmp)
link_libraries(${CMAKE_THREAD_LIBS_INIT})


# Set common source files (keep .cu extension - they get concatenated and compiled as HIP)
set(COMMON_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mp/mp.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mp/mp_montgomery.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/naf.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_common.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/ecm.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/factor_task.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/batch.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/stage1.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/stage2.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/handler.cu
        )

# Set coordinate specific ecc and ecm implementations
if(COORDINATES_EXTENDED)
  set(COMMON_CUDA_SOURCES
        ${COMMON_CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_extended.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/tw_ed_extended.cu
        )
elseif(COORDINATES_INVERTED)
  set(COMMON_CUDA_SOURCES
        ${COMMON_CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_inverted.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/tw_ed_inverted.cu
        )
endif()

set(COMMON_C_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gmp_conv/gmp_conversion.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/ini.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/log.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/file.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/tcp.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/parser.c
        )

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(resource)
