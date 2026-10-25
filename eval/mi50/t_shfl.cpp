#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void k_xor32(float* o) { int l = threadIdx.x; o[l] = __shfl_xor_sync(0xffffffff, (float) l, 16); }
__global__ void k_xor64(float* o) { int l = threadIdx.x; o[l] = __shfl_xor_sync(~0ull, (float) l, 16); }
__global__ void k_xor64w32(float* o) { int l = threadIdx.x; o[l] = __shfl_xor_sync(~0ull, (float) l, 16, 32); }
__global__ void k_xor32w32(float* o) { int l = threadIdx.x; o[l] = __shfl_xor_sync(0xffffffff, (float) l, 16, 32); }
__global__ void k_sync32(float* o) { int l = threadIdx.x; o[l] = __shfl_sync(0xffffffff, (float) l, 3); }
__global__ void k_sync64w32(float* o) { int l = threadIdx.x; o[l] = __shfl_sync(~0ull, (float) l, 3, 32); }
__global__ void k_down32(float* o) { int l = threadIdx.x; float s = (float) l; for (int off = 16; off > 0; off /= 2) s += __shfl_down_sync(0xffffffff, s, off); o[l] = s; }
__global__ void k_down64w32(float* o) { int l = threadIdx.x; float s = (float) l; for (int off = 16; off > 0; off /= 2) s += __shfl_down_sync(~0ull, s, off, 32); o[l] = s; }
__global__ void k_ballot32(unsigned* o) { int l = threadIdx.x; o[l] = __ballot_sync(0xffffffff, l >= 32); }
__global__ void k_ballot64(unsigned* o) { int l = threadIdx.x; unsigned long long b = __ballot_sync(~0ull, l >= 32); o[l] = (unsigned) b; }

int main(int argc, char** argv)
{
    const char* name = argv[1];
    float* d; hipMalloc(&d, 64 * sizeof(float));
    void (*k)(float*) = nullptr;
    if (!strcmp(name, "xor32")) k = k_xor32;
    else if (!strcmp(name, "xor64")) k = k_xor64;
    else if (!strcmp(name, "xor64w32")) k = k_xor64w32;
    else if (!strcmp(name, "xor32w32")) k = k_xor32w32;
    else if (!strcmp(name, "sync32")) k = k_sync32;
    else if (!strcmp(name, "sync64w32")) k = k_sync64w32;
    else if (!strcmp(name, "down32")) k = k_down32;
    else if (!strcmp(name, "down64w32")) k = k_down64w32;
    else if (!strcmp(name, "ballot32")) { unsigned* o; hipMalloc(&o, 64 * sizeof(unsigned)); k_ballot32<<<1, 64>>>(o); }
    else if (!strcmp(name, "ballot64")) { unsigned* o; hipMalloc(&o, 64 * sizeof(unsigned)); k_ballot64<<<1, 64>>>(o); }
    else return 9;
    if (k) k<<<1, 64>>>(d);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("%-10s FAIL %s\n", name, hipGetErrorString(e)); return 1; }
    if (k) {
        float h[64]; hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost);
        printf("%-10s ok  [0]=%g [16]=%g [32]=%g [48]=%g\n", name, h[0], h[16], h[32], h[48]);
    } else printf("%-10s ok\n", name);
    return 0;
}
