// Does a raw buffer load honour a per-lane voffset on gfx906?  (exl3_gfx906_kernel.cuh relies on it)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

__global__ void k_lane_voff(const uint32_t* B, uint32_t* out)
{
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, 0x31004000);
    const int lane = threadIdx.x & 31;
    const uint32_t voff = (uint32_t) lane * 4u;
    const uint32_t soff = (uint32_t) blockIdx.x * 128u * 4u;
    out[blockIdx.x * 64 + threadIdx.x] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 6);
}

__global__ void k_lane_soff(const uint32_t* B, uint32_t* out)
{
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, 0x31004000);
    const int lane = threadIdx.x & 31;
    const uint32_t voff = 0u;
    const uint32_t soff = (uint32_t) (blockIdx.x * 32 + lane) * 4u;
    out[blockIdx.x * 64 + threadIdx.x] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 6);
}

// plain pointer load for reference
__global__ void k_plain(const uint32_t* B, uint32_t* out)
{
    const int lane = threadIdx.x & 31;
    out[blockIdx.x * 64 + threadIdx.x] = B[blockIdx.x * 32 + lane];
}

int main()
{
    const int n = 512;
    uint32_t h[n];
    for (int i = 0; i < n; i++) h[i] = 1000 + i;
    uint32_t *dB, *dout;
    hipMalloc(&dB, n * 4);
    hipMalloc(&dout, 128 * 4);
    hipMemcpy(dB, h, n * 4, hipMemcpyHostToDevice);
    uint32_t o[128];

    for (int which = 0; which < 3; which++)
    {
        hipMemset(dout, 0, 128 * 4);
        if (which == 0) k_lane_voff<<<2, 32>>>(dB, dout);
        if (which == 1) k_lane_soff<<<2, 32>>>(dB, dout);
        if (which == 2) k_plain<<<2, 32>>>(dB, dout);
        hipError_t e = hipDeviceSynchronize();
        hipMemcpy(o, dout, 64 * 4, hipMemcpyDeviceToHost);
        printf("%s (err %s): block0 = ", which == 0 ? "lane voffset " : which == 1 ? "lane soffset " : "plain load   ",
               hipGetErrorString(e));
        for (int i = 0; i < 8; i++) printf("%u ", o[i]);
        printf("... b0[31]=%u  block1[0]=%u\n", o[31], o[32]);
    }
    return 0;
}
