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

template<typename T>
__global__ void dumbfht_gpu_kernel(T *ptr, size_t l2, size_t per_thread) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int start_index = per_thread * i, stop_index = per_thread * (i + 1);
    #pragma unroll
    for(size_t i = 0; i < l2; ++i) {
        size_t s1 = 1ull << i, s2 = s1 << 1;
        #pragma unroll
        for(int j = start_index; j < stop_index; j += s2) {
            for(size_t k = 0; k < s1; ++k) {
                auto u = ptr[j + k], v = ptr[j + k + s1];
                ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
            }
        }
        __syncthreads();
    }
}

template<typename T>
void call_dumbfht(T *ptr, size_t l2, size_t subl2 = 10) {
    if(l2 <= subl2 + 6) throw 1;
    size_t items_per_block = 1ull << subl2;
    size_t threads_per_block = 64;
    size_t items_per_thread = items_per_block / threads_per_block;
    size_t blocks = 1ull << (l2 - subl2 - int(std::log2(64)));
    size_t nbytes = sizeof(T) << l2;
    size_t sz = 1ull << l2;
    std::fprintf(stderr, "items per thread: %zu. blocks: %zu\n", items_per_thread, blocks);
    cudaError_t rc;
    T *tmp;
    if(cudaMalloc((void **)&tmp, nbytes)) throw std::bad_alloc();
    if(cudaMemcpy(tmp, ptr, nbytes, cudaMemcpyHostToDevice)) throw std::runtime_error("Failed to copy to device");
    auto start = std::chrono::high_resolution_clock::now();
    dumbfht_gpu_kernel<<<blocks, threads_per_block>>>(tmp, l2, items_per_thread);
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<T> t2(1ull << l2);
    if(cudaMemcpy(t2.data(), tmp, nbytes, cudaMemcpyDeviceToHost)) throw "thing";
    std::memcpy(ptr, t2.data(), nbytes);
    std::fprintf(stderr, "gpu took %zu time for %zu \n", (end - start).count(), l2);
    cudaFree(tmp);
}


} // nameespace fht
