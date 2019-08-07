#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <cooperative_groups.h>

namespace fht {
template<typename T>void show(const T&x) {
    size_t k = 0;
    for(const auto i: x) {
        std::fprintf(stderr, "%lf,", double(i)); std::fputc('\n', stderr);
        if(++k == 10) return;
    }
}
template<typename T>void show(const T *x, size_t n) {
    size_t k = 0;
    for(size_t i = 0; i < n; ++i) {
        auto v = x[n];
        std::fprintf(stderr, "%lf,", double(v)); std::fputc('\n', stderr);
        if(++k == 10) return;
    }
}

using std::size_t;

template<typename T>
size_t dumbfht(T *const ptr, const size_t l2) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t n = 1ull << l2;
    for(size_t i = 0; i < l2; ++i) {
#if VERBOSE_AF
        for(size_t i = 0; i < 10; ++i)
             std::fprintf(stderr, "%zu has %f\n", i, ptr[i]);
#endif
        size_t s1 = 1ull << i, s2 = s1 << 1;
        for(size_t j = 0; j < n; j += s2) {
            for(size_t k = 0; k < s1; ++k) {
                auto u = ptr[j + k], v = ptr[j + k + s1];
                ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return (end - start).count();
}
#if 0
template<typename T>
void dumbfake(T *ptr, size_t l2, size_t pt, size_t blocks, size_t threads_per_block) {
    T mv = 1e200, nv = -1e400;
}
#endif
template<typename T>
__global__ void dumbfht_gpu_kernel(T *ptr, size_t l2, int nthreads) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    for(int i = 0; i < l2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        int nthreads_active = min(n >> (i + 1), nthreads);
        int npert = n / nthreads_active;
        if(tid < nthreads_active) {
            for(int j = tid * npert, e = j + npert; j != e; j += s2) {
                #pragma unroll 8
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
                }
            }
        }
    }
}
template<typename T>
__global__ void mkdumbfht_gpu_kernel(T *ptr, size_t l2, int nthreads, int nkern) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    for(int i = 0; i < l2; ++i) {
        const int s1 = 1 << i, s2 = s1 << 1;
        const int nthreads_active = min(n >> (i + 1), nthreads);
        const int npert = n / nthreads_active;
        for(int kind = 0; kind < nkern; ++kind) {
            auto lptr = ptr + kind * n;
            if(tid < nthreads_active) {
                #pragma unroll 8
                for(int j = tid * npert, e = j + npert; j != e; j += s2) {
                    #pragma unroll 8
                    for(size_t k = 0; k < s1; ++k) {
                        auto u = lptr[j + k], v = lptr[j + k + s1];
                        lptr[j + k] = u + v, lptr[j + k + s1] = u - v;
                    }
                }
            }
        }
    }
}


template<typename T>
void call_dumbfht(T *ptr, size_t l2, size_t threadl2 = 8, int n=1) {
    size_t threads_per_block = 1ull << threadl2;
    size_t sz = 1ull << l2,
           nbytes = (n * sizeof(T)) << l2;
    size_t blocks = size_t(n) << (l2 - threadl2);
    int ptl2 = l2 - threadl2;
    size_t items_per_thread = 1ull << ptl2;
    cudaError_t rc;
    T *tmp;
    if(cudaMalloc((void **)&tmp, nbytes)) throw std::bad_alloc();
    if(cudaMemcpy(tmp, ptr, nbytes, cudaMemcpyHostToDevice)) throw std::runtime_error("Failed to copy to device");
    std::fprintf(stderr, "tpb: %zu. ipt: %zu sz: %zu. blocks: %zu\n", threads_per_block, items_per_thread, sz, blocks);
    auto start = std::chrono::high_resolution_clock::now();
#if 0
    cudaStream_t cs[16];
    std::for_each(std::begin(cs), std::end(cs), [](auto &x) {cudaStreamCreate(&x);});
    for(int i = 0; i < n; ++i) {
        dumbfht_gpu_kernel<<<blocks, threads_per_block, 0, cs[i % 16]>>>(tmp + (size_t(i) << l2), l2, threads_per_block * blocks);
    }
    std::for_each(std::begin(cs), std::end(cs), [](auto x) {cudaStreamDestroy(x);});
#else
    mkdumbfht_gpu_kernel<<<blocks, threads_per_block>>>(tmp, l2, threads_per_block * blocks, n);
#endif
    //dumbfake(tmp, l2, items_per_thread, blocks, threads_per_block);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    if(cudaMemcpy(ptr, tmp, nbytes, cudaMemcpyDeviceToHost)) throw "thing";
    std::fprintf(stderr, "gpu took %zu time for %zu \n", (end - start).count(), l2);
    cudaFree(tmp);
}


} // nameespace fht
