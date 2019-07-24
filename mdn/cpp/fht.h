#include "./fht.h"
#include <cstdlib>
namespace fht {

template<typename T>
void dumbfht(T *ptr, size_t l2) {
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
}

template<typename T>
__global__ void dumbfht_gpu_kernel(T *ptr, size_t l2) {
    size_t n = 1ull << l2;
    int blockId = blockIdx.y* gridDim.x+ blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    size_t per_n = blockDim.x * gridDim.x;
    int start_index = per_n * threadId, stop_index = per_n * (threadId + 1);
    assert(!(gridDim.x & (gridDim.x - 1)));
    assert(!(blockDim.x & (blockDim.x - 1)));
    for(size_t i = 0; i < l2; ++i) {
        size_t s1 = 1ull << i, s2 = s1 << 1;
        #pragma unroll
        for(int j = start_index; j < stop_index; j += s2) {
            for(size_t k = 0; k < s1; ++k) {
                auto u = ptr[j + k], v = ptr[j + k + s1];
                ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
            }
        }
        __sync_threads();
    }
}

template<typename T>
void call_dumbfht(T *ptr, size_t l2) {
    if(l2 <= 6) throw 1;
    size_t nelem = 1ull << l2;
    size_t per_block = 1ull << 6;
    size_t blocks = 1ull << (l2 - 6);
    dumbfht_gpu_kernel<<<blocks, per_block>>>(ptr, l2);
}


} // nameespace fht
