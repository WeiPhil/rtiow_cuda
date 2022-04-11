#pragma once

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

namespace cudart {

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCuda(val) check_cuda_allocator((val), #val, __FILE__, __LINE__)

void check_cuda_allocator(cudaError_t result,
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

template <typename T>
class Allocator {
public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    Allocator() {}

    /// Allows other (nested) types to use our own allocator
    template <class U>
    Allocator(const Allocator<U> &)
    {
    }

    value_type *allocate(size_t n)
    {
        // std::cout << "allocate " << typeid(T).name() << " x" << n << " = " << (sizeof(T) *
        // n) << " B" << std::endl;

        value_type *result = nullptr;
        checkCuda(cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal));

        return result;
    }

    void deallocate(value_type *ptr, size_t n)
    {
        // std::cout << "deallocate " << typeid(T).name() << " x" << n << std::endl;

        checkCuda(cudaFree(ptr));
    }
};

/// A reference is a managed pointer that can be shared.
template <class T1, class T2>
bool operator==(const Allocator<T1> &, const Allocator<T2> &)
{
    return true;
}

template <class T1, class T2>
bool operator!=(const Allocator<T1> &lhs, const Allocator<T2> &rhs)
{
    return !(lhs == rhs);
}

template <typename T>
using ref = std::shared_ptr<T>;

/// Convenience function to create ref objects easily.
template <typename T, class... Args>
ref<T> make_shared(Args &&...args)
{
    return std::allocate_shared<T, Allocator<T>>({}, args...);
}

}

#endif