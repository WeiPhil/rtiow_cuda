#pragma once

#ifndef MACROS_H
#define MACROS_H

#include <iostream>
#include "cuda_runtime.h"

#define CUDART_NAMESPACE_BEGIN namespace cudart {
#define CUDART_NAMESPACE_END }

#if defined(__CUDACC__)
#define CUDART_FN __device__ __host__
#define CUDART_GPU_FN __device__
#define CUDART_CPU_FN __host__
#define CUDART_KERNEL __global__
#else
#define CUDART_FN
#define CUDART_GPU_FN
#define CUDART_CPU_FN
#define CUDART_KERNEL
#endif

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCuda(val) check_cuda_allocator((val), #val, __FILE__, __LINE__)

inline void check_cuda_allocator(cudaError_t result,
                                 char const *const func,
                                 const char *const file,
                                 int const line)
{
    if (result) {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << " at " << file
                  << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#endif