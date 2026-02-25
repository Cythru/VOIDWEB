cmake_minimum_required(VERSION 3.22)
project(oracle_kernels LANGUAGES CXX CUDA)

# ── Compiler requirements ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architecture detection ───────────────────────────────────────────────
# Auto-detect GPU or use ORACLE_ARCH override.
if(DEFINED ORACLE_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${ORACLE_ARCH})
else()
    # Default: compile for all modern NVIDIA GPUs (Ampere + Hopper + Ada).
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ── Release flags ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -ffast-math -funroll-loops")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --generate-line-info")

# Suppress CUDA deprecation noise.
add_compile_definitions(CUDA_IGNORE_DEPRECATED_PHASE1_API)

# ── Find dependencies ──────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Optional: CUTLASS (headers only, clone into third_party/cutlass).
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
    include_directories(third_party/cutlass/include)
    add_compile_definitions(ORACLE_HAVE_CUTLASS)
    message(STATUS "CUTLASS found — FP8/INT4 GEMM enabled")
else()
    message(STATUS "CUTLASS not found — using fallback BF16 GEMM")
endif()

# ── Kernel source files ───────────────────────────────────────────────────────
set(KERNEL_SOURCES
    kernels/attention/flash_attention.cu
    kernels/rope/rope.cu
    kernels/norm/rms_norm.cu
    kernels/gemm/gemm_fp8.cu
    kv_cache/src/block_manager.cpp
    loader/cpp/src/weight_loader.cpp
)

# ── Shared library: libkernels.so ─────────────────────────────────────────────
add_library(oracle_kernels SHARED ${KERNEL_SOURCES})

target_include_directories(oracle_kernels PRIVATE
    kernels/include
    kv_cache/include
    loader/cpp/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(oracle_kernels PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas          # used only for fallback path
    CUDA::cublasLt        # FP8 GEMM acceleration
)

# ── CUDA-specific compile options ────────────────────────────────────────────
set_target_properties(oracle_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
    OUTPUT_NAME                "kernels"
    PREFIX                     "lib"
    SUFFIX                     ".so"
)

target_compile_options(oracle_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        -Xcompiler=-O3
        -Xcompiler=-march=native
        # Hoist common expressions across warp divergence.
        --ptxas-options=-v
        # Maximum register use for throughput.
        -maxrregcount=128
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -march=native -ffast-math -fPIC
    >
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS oracle_kernels
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES kernels/include/kernels.h DESTINATION include/oracle)

# ── Optional: benchmark binary ───────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/bench/bench_kernels.cu")
    add_executable(bench_kernels bench/bench_kernels.cu)
    target_include_directories(bench_kernels PRIVATE kernels/include ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(bench_kernels PRIVATE oracle_kernels CUDA::cudart)
    set_target_properties(bench_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
