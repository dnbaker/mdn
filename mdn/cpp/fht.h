#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>

namespace fht {
using std::size_t;

template<typename T>
void dumbfht(T *ptr, size_t l2) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t n = 1ull << l2;
    for(size_t i = 0; i < l2; ++i) {
        size_t s1 = 1ull << i, s2 = s1 << 1;
        for(size_t j = 0; j < n; j += s2) {
            for(size_t k = 0; k < s1; ++k) {
                auto u = ptr[j + k], v = ptr[j + k + s1];
                ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "cpu took %zu time for %zu \n", (end - start).count(), l2);
}
#if 0
template<typename T>
void dumbfake(T *ptr, size_t l2, size_t pt, size_t blocks, size_t threads_per_block) {
    T mv = 1e200, nv = -1e400;
}
#endif
template<typename T>
__global__ void dumbfht_gpu_kernel(T *ptr, size_t l2, int per_thread, int ptl2) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int start_index = per_thread * tid, stop_index = start_index + per_thread;
    for(int i = 0; i < l2 - ptl2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        #pragma unroll
        for(int j = start_index; j < stop_index; j += s2) {
            #pragma unroll
            for(size_t k = 0; k < s1; ++k) {
                auto u = ptr[j + k], v = ptr[j + k + s1];
                ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
            }
        }
        __syncthreads();
    }
    for(int i = l2 - ptl2; i < l2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        if(tid < (1 << i)) {
            #pragma unroll
            for(int j = start_index; j < stop_index; j += s2) {
                #pragma unroll
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
                }
            }
        }
        __syncthreads();
    }
#if 0
    for(size_t i = 0; i < l2; ++i) {
        int max_thread_num = 1 << (l2 - i);
        if(i < max_thread_num) {
            size_t s1 = 1ull << i, s2 = s1 << 1;
            #pragma unroll
            for(int j = start_index; j < stop_index; j += s2) {
                #pragma unroll
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
                }
            }
        }
        __syncthreads();
    }
#endif
}

template<typename T>
void call_dumbfht(T *ptr, size_t l2, size_t threadl2 = 4, size_t blocksl2=4) {
    size_t threads_per_block = 1ull << threadl2;
    size_t sz = 1ull << l2,
           nbytes = sizeof(T) << l2;
    size_t blocks = 1ull << blocksl2;
    int ptl2 = l2 - threadl2 - blocksl2;
    size_t items_per_thread = 1ull << ptl2;
    if(l2 < threadl2 + blocksl2) throw 1;
    cudaError_t rc;
    T *tmp;
    if(cudaMalloc((void **)&tmp, nbytes)) throw std::bad_alloc();
    if(cudaMemcpy(tmp, ptr, nbytes, cudaMemcpyHostToDevice)) throw std::runtime_error("Failed to copy to device");
    std::fprintf(stderr, "tpb: %zu. ipt: %zu sz: %zu. blocks: %zu\n", threads_per_block, items_per_thread, sz, blocks);
    auto start = std::chrono::high_resolution_clock::now();
    dumbfht_gpu_kernel<<<blocks, threads_per_block>>>(tmp, l2, items_per_thread, ptl2);
    //dumbfake(tmp, l2, items_per_thread, blocks, threads_per_block);
    auto end = std::chrono::high_resolution_clock::now();
    if(cudaMemcpy(ptr, tmp, nbytes, cudaMemcpyDeviceToHost)) throw "thing";
    std::fprintf(stderr, "gpu took %zu time for %zu \n", (end - start).count(), l2);
    cudaFree(tmp);
}


} // nameespace fht
