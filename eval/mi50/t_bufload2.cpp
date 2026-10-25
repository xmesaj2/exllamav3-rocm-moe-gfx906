// Which buffer-descriptor flags make raw_buffer_load_b32 work on gfx906?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

template <unsigned F>
__global__ void k(const uint32_t* B, uint32_t* out)
{
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, F);
    const int lane = threadIdx.x & 31;
    out[threadIdx.x] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, (uint32_t) lane * 4u, 0u, 6);
}

template <unsigned F> void run(const char* tag, const uint32_t* dB, uint32_t* dout)
{
    uint32_t o[64];
    hipMemset(dout, 0, 64 * 4);
    k<F><<<1, 32>>>(dB, dout);
    hipError_t e = hipDeviceSynchronize();
    hipMemcpy(o, dout, 32 * 4, hipMemcpyDeviceToHost);
    printf("flags 0x%08x %-10s err %-20s -> ", F, tag, hipGetErrorString(e));
    for (int i = 0; i < 4; i++) printf("%u ", o[i]);
    printf("| last %u %s\n", o[31], (o[0] == 1000 && o[31] == 1031) ? "OK" : "");
}

int main()
{
    const int n = 256;
    uint32_t h[n];
    for (int i = 0; i < n; i++) h[i] = 1000 + i;
    uint32_t *dB, *dout;
    hipMalloc(&dB, n * 4);
    hipMalloc(&dout, 64 * 4);
    hipMemcpy(dB, h, n * 4, hipMemcpyHostToDevice);
    run<0x31004000u>("author", dB, dout);
    run<0x00000000u>("zero", dB, dout);
    run<0x00004000u>("0x4000", dB, dout);
    run<0x0000c000u>("0xc000", dB, dout);
    run<0x0002c000u>("0x2c000", dB, dout);
    run<0x0003c000u>("0x3c000", dB, dout);
    run<0x000e8000u>("fmt=uint32", dB, dout);
    run<0x31000000u>("0x31000000", dB, dout);
    run<0x11004000u>("0x11004000", dB, dout);
    run<0x10004000u>("0x10004000", dB, dout);
    return 0;
}
