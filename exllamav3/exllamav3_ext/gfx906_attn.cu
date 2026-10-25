#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include "util.h"
#include "util.cuh"
#include "triton_kernel.h"
#include "gfx906_attn.h"

// Flash-decoding split kernel for AMD Vega 20 (gfx906: MI50 / MI60 / Radeon VII, wave64), a port of
// rdna3_attn.cu. Drop-in for the Triton _paged_attn_decode_split_kernel on the graphed decode/verify
// path (bc_attn.py): same launch arguments (plus Triton's two hidden scratch pointers), same
// program/split decomposition and the same partial_o / partial_ml layout, so the Triton combine
// kernel reduces its output.
//
// Per (kv head, split) block; each logical wave walks 32-token blocks of the split:
// - scores: lane <-> token. The lane reads its token's key row, fp16 or 8-bit (midpoint grid,
//   fp16 scale per 32 values, rotated domain); q rows are LDS broadcasts; v_dot2 per 32-group
// - online softmax with a wave-uniform running max per row
// - values: lane <-> dims (8 per lane at head_dim 256), p broadcast through LDS, fp32 FMAs
// - the four waves merge through LDS (decode), then one partial per (row, split) is stored
//
// Differences from the RDNA3 version:
// - wave64. The algorithm keeps its 32-lane "logical waves" and two of them share each 64-lane
//   hardware wave (lanes 0-31 and 32-63). Every cross-lane op stays inside one 32-lane half (shuffles
//   take width 32), so lane <-> token, LDS layout and partial layout are exactly as before. A block is
//   a whole number of hardware waves; with an odd number of logical waves the last half idles
// - no WMMA on gfx906: 8-bit verification (2 <= q_len <= 8) uses the LDS-staged v_dot2 kernel
//   (K/V tiles staged once per program) instead of the matrix-core kernel
// - dot2 is v_dot2_f32_f16 (gfx906 has it); other targets get a plain fp32 FMA fallback
//
// Scope: causal, no window / softcap / sinks, q_len <= 16, head_dim 128 or 256, cache fp16 or
// 8-bit for both K and V, softmax scale 1/sqrt(head_dim). Other configurations keep Triton.
//
// Build switches (-D):
//   GFX906_TK_WARP=64           threads per warp in the num_warps that TritonKernel launches with (see end of file)
//   GFX906_VERIFY_Q8_LDS=0      8-bit verify through the plain split kernel instead of the LDS-staged one
//   GFX906_NO_FDOT2             do not use __builtin_amdgcn_fdot2

#ifdef __HIP_PLATFORM_AMD__

// TritonKernel::launch() asks for <device warp size> * num_warps threads (triton_kernel_hip.cpp), so
// on gfx9 this is the wave size (64) and the wave count below is threads / 64
#ifndef GFX906_TK_WARP
#define GFX906_TK_WARP 64
#endif
#ifndef GFX906_VERIFY_Q8_LDS
#define GFX906_VERIFY_Q8_LDS 1
#endif
#if defined(__gfx906__) && !defined(GFX906_NO_FDOT2)
#define GFX906_FDOT2 1
#else
#define GFX906_FDOT2 0
#endif

namespace gfx906_attn {

constexpr int PAGE = 256;
constexpr int TB = 32;              // tokens per logical wave block (= lanes of a logical wave)

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c)
{
    #if GFX906_FDOT2
        return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
    #else
        const hv2 x = __builtin_bit_cast(hv2, a), y = __builtin_bit_cast(hv2, b);
        return fmaf((float) x.y, (float) y.y, fmaf((float) x.x, (float) y.x, c));
    #endif
}

// Two unsigned bytes of w (selected by v_perm; bytes 4-7 of the pair are 0x64) as a half2 of (b - 128):
// 0x64bb is 1024 + b
__device__ __forceinline__ uint32_t bytes_to_h2c(uint32_t w, uint32_t sel)
{
    uint32_t h = __builtin_amdgcn_perm(0x64646464u, w, sel);
    half2 v = *reinterpret_cast<half2*>(&h);
    v = __hsub2(v, __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&v);
}

// Reductions over one 32-lane logical wave (xor masks < 32 never leave a half of the 64-lane wave)
__device__ __forceinline__ float wave_max(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v = fmaxf(v, __shfl_xor(v, i, 32));
    return v;
}

__device__ __forceinline__ float wave_sum(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v += __shfl_xor(v, i, 32);
    return v;
}

}  // namespace gfx906_attn

// The launch must provide at least NTH threads (see the end of the file); any extra are dropped.
// Too few would silently compute garbage, so trap instead
#define GFX906_BLOCK_CHECK(NTH) \
    if (blockDim.x < (NTH)) __builtin_trap(); \
    if (threadIdx.x >= (NTH)) return;

// Block layout: QL == 1 (decode): TG = 4 logical waves split the tokens, each computes all GROUP rows,
// merged through LDS. QL > 1 (draft verification): one logical wave per query position (GROUP rows
// each) over all tokens of the split, no merge; the program for h_block 0 of each kv head computes
// every row and stores the partials of all its h_blocks, the other h_block programs exit (K/V read
// once). NTH rounds the logical waves up to whole 64-lane hardware waves
template <int QL> struct AttnGeom
{
    static constexpr int TG = QL == 1 ? 4 : 1;       // token groups (logical waves) per row group
    static constexpr int RG = QL;                    // row groups (logical waves): one per query position
    static constexpr int WAVES = TG * RG;
    static constexpr int NTH = ((WAVES + 1) / 2) * 64;
};

template <int QC, int HD, int QL, int NKV, int GROUP>
__global__ __launch_bounds__(AttnGeom<QL>::NTH)
void gfx906_attn_decode_split_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
    using namespace gfx906_attn;
    constexpr int TG = AttnGeom<QL>::TG;
    constexpr int WAVES = AttnGeom<QL>::WAVES;
    constexpr int NTH = AttnGeom<QL>::NTH;
    constexpr int BM = QL <= 1 ? 1 : QL <= 2 ? 2 : QL <= 4 ? 4 : QL <= 8 ? 8 : 16;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = BM * BH;                      // 16, partial layout rows
    constexpr int RW = GROUP;                                // rows per wave (one query position)
    constexpr int NROWS = QL * GROUP;                        // rows of this kv head, query-major
    constexpr int G = HD / 32;                               // 32-value groups per head
    constexpr int DPL = HD / 32;                             // value dims per lane
    constexpr int NQH = NKV * GROUP;
    constexpr int H_BLOCKS = (GROUP + BH - 1) / BH;

    GFX906_BLOCK_CHECK(NTH)

    __shared__ __attribute__((aligned(16))) uint32_t qs[NROWS][HD / 2];   // q rows as half2 (rotated for 8-bit keys)
    __shared__ float qsum[NROWS][G];             // per-group sums of qs (midpoint correction)
    __shared__ float pbuf[WAVES][TB][RW];        // softmax weights of the current block
    __shared__ float accm[TG > 1 ? RW : 1][TG > 1 ? HD : 1];    // cross-wave merge (decode)
    __shared__ float mw[WAVES][RW], lw[WAVES][RW];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;             // logical wave
    const int lane = threadIdx.x & 31;             // lane within the logical wave

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage q: one thread per (row, 32-group); 32-point Hadamard for rotated (8-bit) keys
    for (int task = threadIdx.x; task < NROWS * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        const int qi = gr / GROUP, hl = gr % GROUP;
        float x[32];
        const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
        if constexpr (QC > 0)
        {
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1)
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] *= 0.17677669529663687f;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            half2 v = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
            s += __low2float(v) + __high2float(v);
            qs[gr][g * 16 + i] = *reinterpret_cast<uint32_t*>(&v);
        }
        qsum[gr][g] = s;
    }
    if constexpr (TG > 1)
        for (int i = threadIdx.x; i < RW * HD; i += NTH) (&accm[0][0])[i] = 0.0f;
    __syncthreads();

    if constexpr (WAVES % 2 != 0)
        if (warp >= WAVES) return;                 // padding half of the last hardware wave

    const float sm_scale = QC > 0 ? rsqrtf((float) HD) / 128.0f : rsqrtf((float) HD);

    const int rg = warp / TG;                      // row group = query position
    const int tg = warp % TG;                      // token group
    const int row0 = rg * RW;
    const int q_abs = total_k_len - QL + rg;       // causal bound of this wave's rows

    float m[RW], l[RW], acc[RW][DPL], psum[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        m[r] = -INFINITY; l[r] = 0.0f; psum[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.0f;
    }

    const size_t row_elems = (size_t) NKV * HD;      // per token row: all kv heads
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const int t_last = min(n_end, q_abs + 1);        // causal: no keys past this wave's position

    for (int b0 = n_start + tg * TB; b0 < t_last; b0 += TG * TB)
    {
        const int t = b0 + lane;
        const bool tvalid = t < t_last;
        const int tc = tvalid ? t : t_last - 1;
        const int tok_row = bt[tc / PAGE] * PAGE + (tc % PAGE);
        const size_t kbase = (size_t) tok_row * row_elems + (size_t) kvh * HD;

        // Scores for this lane's token
        float s[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) s[r] = 0.0f;

        #pragma unroll 1
        for (int g = 0; g < G; ++g)
        {
            uint32_t kw[16];
            float gs = 1.0f;
            if constexpr (QC > 0)
            {
                const uint4* kp = (const uint4*) ((const uint8_t*) k_cache + kbase + g * 32);
                uint4 a = kp[0], b = kp[1];
                uint32_t w8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    kw[2 * i] = gfx906_attn::bytes_to_h2c(w8[i], 0x04010400u);
                    kw[2 * i + 1] = gfx906_attn::bytes_to_h2c(w8[i], 0x04030402u);
                }
                gs = __half2float(k_scales[(size_t) tok_row * (NKV * G) + kvh * G + g]);
            }
            else
            {
                const uint4* kp = (const uint4*) ((const half*) k_cache + kbase + g * 32);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 v = kp[i];
                    kw[4 * i] = v.x; kw[4 * i + 1] = v.y; kw[4 * i + 2] = v.z; kw[4 * i + 3] = v.w;
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const uint4* qp = (const uint4*) &qs[row0 + r][g * 16];
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 qv = qp[i];
                    d = gfx906_attn::dot2(kw[4 * i], qv.x, d);
                    d = gfx906_attn::dot2(kw[4 * i + 1], qv.y, d);
                    d = gfx906_attn::dot2(kw[4 * i + 2], qv.z, d);
                    d = gfx906_attn::dot2(kw[4 * i + 3], qv.w, d);
                }
                if constexpr (QC > 0) d = (d + 0.5f * qsum[row0 + r][g]) * gs;   // (b - 128) + 0.5 = b - 127.5
                s[r] += d;
            }
        }

        // Online softmax, wave-uniform running max per row
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            const float sc = tvalid ? s[r] * sm_scale : -INFINITY;
            const float m_new = fmaxf(m[r], gfx906_attn::wave_max(sc));
            const float alpha = m[r] == -INFINITY ? 0.0f : __expf(m[r] - m_new);
            const float p = tvalid ? __expf(sc - m_new) : 0.0f;
            l[r] = l[r] * alpha + p;
            psum[r] *= alpha;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] *= alpha;
            m[r] = m_new;
            pbuf[warp][lane][r] = p;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // Values: lane <-> dims
        const int nt = min(TB, t_last - b0);
        for (int tt = 0; tt < nt; ++tt)
        {
            const int vrow = __shfl(tok_row, tt, 32);
            const size_t vbase = (size_t) vrow * row_elems + (size_t) kvh * HD + lane * DPL;
            float vf[DPL];
            float vs = 1.0f;
            if constexpr (QC > 0)
            {
                const uint8_t* vp = (const uint8_t*) v_cache + vbase;
                #pragma unroll
                for (int j = 0; j < DPL; j += 4)
                {
                    uint32_t w = *(const uint32_t*) (vp + j);
                    vf[j] = (float) ((w >> 0) & 0xffu);
                    vf[j + 1] = (float) ((w >> 8) & 0xffu);
                    vf[j + 2] = (float) ((w >> 16) & 0xffu);
                    vf[j + 3] = (float) ((w >> 24) & 0xffu);
                }
                vs = __half2float(v_scales[(size_t) vrow * (NKV * G) + kvh * G + (lane * DPL) / 32]) * (1.0f / 128.0f);
            }
            else
            {
                const half2* vp = (const half2*) ((const half*) v_cache + vbase);
                #pragma unroll
                for (int j = 0; j < DPL / 2; ++j)
                {
                    half2 v = vp[j];
                    vf[2 * j] = __low2float(v);
                    vf[2 * j + 1] = __high2float(v);
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const float pp = pbuf[warp][tt][r] * vs;
                psum[r] += pp;
                #pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(pp, vf[j], acc[r][j]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }

    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        l[r] = gfx906_attn::wave_sum(l[r]);
        if constexpr (QC > 0)
        {
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] -= 127.5f * psum[r];
        }
    }

    // Row (qi, hl) -> partial slot of its h_block program: r16 = (hl % BH) * BM + qi
    auto slot = [&] (int qi, int hl, float*& po, float*& pm)
    {
        const int hb = hl / BH;
        const int r16 = (hl % BH) * BM + qi;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        po = partial_o + (pbase * BLOCK_ROWS + r16) * HD;
        pm = partial_ml + (pbase * BLOCK_ROWS + r16) * 2;
    };

    if constexpr (TG == 1)
    {
        if (split >= num_splits) return;
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float* po; float* pm;
            slot(rg, r, po, pm);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) po[lane * DPL + j] = acc[r][j];
            if (lane == 0) { pm[0] = m[r]; pm[1] = l[r]; }
        }
        // Padded / out-of-group rows of the partial tiles are never read back into valid outputs
    }
    else
    {
        // Merge the token-group waves through LDS
        #pragma unroll
        for (int r = 0; r < RW; ++r)
            if (lane == 0) { mw[warp][r] = m[r]; lw[warp][r] = l[r]; }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float mt = -INFINITY;
            #pragma unroll
            for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
            const float f = m[r] == -INFINITY ? 0.0f : __expf(m[r] - mt);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) atomicAdd(&accm[r][lane * DPL + j], acc[r][j] * f);
        }
        __syncthreads();
        if (split >= num_splits) return;
        for (int i = threadIdx.x; i < BLOCK_ROWS * HD; i += NTH)
        {
            const int r16 = i / HD;            // QL == 1: r16 = head within h_block 0
            float* po = partial_o + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * HD;
            po[i] = r16 < RW ? accm[r16][i % HD] : 0.0f;
        }
        if (threadIdx.x < BLOCK_ROWS)
        {
            const int r = threadIdx.x;
            float mt = -INFINITY, lt = 0.0f;
            if (r < RW)
            {
                #pragma unroll
                for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
                #pragma unroll
                for (int w = 0; w < TG; ++w)
                    if (mw[w][r] != -INFINITY) lt += lw[w][r] * __expf(mw[w][r] - mt);
            }
            float* pm = partial_ml + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * 2;
            pm[r * 2] = mt;
            pm[r * 2 + 1] = lt;
        }
    }
}

#if GFX906_VERIFY_Q8_LDS

// Verification (2 <= q_len <= 8), 8-bit cache: K/V tiles of TB tokens are staged in LDS once per
// program (register prefetch of the next tile overlaps the compute of the current one); one logical
// wave per query position computes its GROUP rows from LDS. Same partial layout as above. This is
// the kernel that replaces the WMMA one on gfx906
template <int HD, int QL, int NKV, int GROUP>
__global__ __launch_bounds__(((QL + 1) / 2) * 64)
void gfx906_attn_verify_q8_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
    using namespace gfx906_attn;
    constexpr int NTH = ((QL + 1) / 2) * 64;      // QL logical waves, rounded up to hardware waves
    constexpr int BM = QL <= 2 ? 2 : QL <= 4 ? 4 : 8;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = BM * BH;
    constexpr int RW = GROUP;
    constexpr int NROWS = QL * GROUP;
    constexpr int G = HD / 32;
    constexpr int DPL = HD / 32;
    constexpr int NQH = NKV * GROUP;
    constexpr int H_BLOCKS = (GROUP + BH - 1) / BH;
    constexpr int KROW = HD / 4 + 4;              // padded key row, words: conflict-free lane <-> token reads
    constexpr int VROW = HD / 4;
    constexpr int CPR = HD / 16;                  // 16-byte chunks per row
    constexpr int NCHUNK = 2 * TB * CPR;          // K and V
    constexpr int NSC = 2 * TB * G;               // K and V scales (halves)
    constexpr int PER_T = (NCHUNK + NTH - 1) / NTH;
    constexpr int PER_S = (NSC + NTH - 1) / NTH;

    GFX906_BLOCK_CHECK(NTH)

    __shared__ __attribute__((aligned(16))) uint32_t qs[NROWS][HD / 2];
    __shared__ float qsum[NROWS][G];
    __shared__ __attribute__((aligned(16))) uint32_t kt[TB][KROW];
    __shared__ __attribute__((aligned(16))) uint32_t vt[TB][VROW];
    __shared__ half sct[2][TB][G];
    __shared__ float pbuf[QL][TB][RW];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;            // logical wave = query position (>= QL: padding, only stages data)
    const int lane = threadIdx.x & 31;
    const bool active = warp < QL;

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    for (int task = threadIdx.x; task < NROWS * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        const int qi = gr / GROUP, hl = gr % GROUP;
        float x[32];
        const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
        #pragma unroll
        for (int h = 1; h < 32; h <<= 1)
            #pragma unroll
            for (int i = 0; i < 32; ++i)
                if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            half2 v = __floats2half2_rn(x[2 * i] * 0.17677669529663687f, x[2 * i + 1] * 0.17677669529663687f);
            sq += __low2float(v) + __high2float(v);
            qs[gr][g * 16 + i] = *reinterpret_cast<uint32_t*>(&v);
        }
        qsum[gr][g] = sq;
    }

    const float sm_scale = rsqrtf((float) HD) / 128.0f;
    const int row0 = warp * RW;
    const int q_abs = total_k_len - QL + warp;

    float m[RW], l[RW], acc[RW][DPL], psum[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        m[r] = -INFINITY; l[r] = 0.0f; psum[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.0f;
    }

    const size_t row_elems = (size_t) NKV * HD;
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const uint8_t* kc = (const uint8_t*) k_cache;
    const uint8_t* vc = (const uint8_t*) v_cache;

    // Register prefetch of one tile: 16-byte K/V chunks and scale halves
    uint4 pr[PER_T];
    half ps[PER_S];
    auto fetch = [&] (int tb)
    {
        #pragma unroll
        for (int i = 0; i < PER_T; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NCHUNK)
            {
                const int isv = c / (TB * CPR);
                const int cc = c % (TB * CPR);
                const int tk = cc / CPR, ch = cc % CPR;
                const int t = min(tb + tk, n_end - 1);
                const int tr = bt[t / PAGE] * PAGE + (t % PAGE);
                const uint8_t* src = (isv ? vc : kc) + ((size_t) tr * row_elems + (size_t) kvh * HD) + ch * 16;
                pr[i] = *(const uint4*) src;
            }
        }
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                const int tk = cc / G, g = cc % G;
                const int t = min(tb + tk, n_end - 1);
                const int tr = bt[t / PAGE] * PAGE + (t % PAGE);
                ps[i] = (isv ? v_scales : k_scales)[(size_t) tr * (NKV * G) + kvh * G + g];
            }
        }
    };
    auto commit = [&] ()
    {
        #pragma unroll
        for (int i = 0; i < PER_T; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NCHUNK)
            {
                const int isv = c / (TB * CPR);
                const int cc = c % (TB * CPR);
                const int tk = cc / CPR, ch = cc % CPR;
                uint32_t* dst = isv ? &vt[tk][ch * 4] : &kt[tk][ch * 4];
                *(uint4*) dst = pr[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                sct[isv][cc / G][cc % G] = ps[i];
            }
        }
    };

    // Every thread, padding wave included, takes part in staging and in the barriers; only the
    // arithmetic below is limited to the QL active waves. The tile loop bounds are block-uniform
    if (n_start < n_end) fetch(n_start);
    for (int tb = n_start; tb < n_end; tb += TB)
    {
        __syncthreads();              // previous tile consumed (and q staged, first pass)
        commit();
        __syncthreads();
        if (tb + TB < n_end) fetch(tb + TB);

        const int t = tb + lane;
        const bool tvalid = t < n_end && t <= q_abs;
        const int nt = active ? max(0, min(min(TB, n_end - tb), q_abs + 1 - tb)) : 0;
        if (nt == 0) continue;        // this wave's rows see none of the tile (or padding wave)

        float s[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) s[r] = 0.0f;
        #pragma unroll 2
        for (int g = 0; g < G; ++g)
        {
            const uint4 a = *(const uint4*) &kt[lane][g * 8];
            const uint4 b = *(const uint4*) &kt[lane][g * 8 + 4];
            const uint32_t w8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            uint32_t kw[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                kw[2 * i] = gfx906_attn::bytes_to_h2c(w8[i], 0x04010400u);
                kw[2 * i + 1] = gfx906_attn::bytes_to_h2c(w8[i], 0x04030402u);
            }
            const float gs = __half2float(sct[0][lane][g]);
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const uint4* qp = (const uint4*) &qs[row0 + r][g * 16];
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 qv = qp[i];
                    d = gfx906_attn::dot2(kw[4 * i], qv.x, d);
                    d = gfx906_attn::dot2(kw[4 * i + 1], qv.y, d);
                    d = gfx906_attn::dot2(kw[4 * i + 2], qv.z, d);
                    d = gfx906_attn::dot2(kw[4 * i + 3], qv.w, d);
                }
                s[r] += (d + 0.5f * qsum[row0 + r][g]) * gs;
            }
        }

        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            const float sc = tvalid ? s[r] * sm_scale : -INFINITY;
            const float m_new = fmaxf(m[r], gfx906_attn::wave_max(sc));
            const float alpha = m[r] == -INFINITY ? 0.0f : __expf(m[r] - m_new);
            const float p = tvalid ? __expf(sc - m_new) : 0.0f;
            l[r] = l[r] * alpha + p;
            psum[r] *= alpha;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] *= alpha;
            m[r] = m_new;
            pbuf[warp][lane][r] = p;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        #pragma unroll 4
        for (int tt = 0; tt < nt; ++tt)
        {
            float vf[DPL];
            #pragma unroll
            for (int j = 0; j < DPL; j += 4)
            {
                const uint32_t w = vt[tt][(lane * DPL + j) / 4];
                vf[j] = (float) (w & 0xffu);
                vf[j + 1] = (float) ((w >> 8) & 0xffu);
                vf[j + 2] = (float) ((w >> 16) & 0xffu);
                vf[j + 3] = (float) ((w >> 24) & 0xffu);
            }
            const float vs = __half2float(sct[1][tt][(lane * DPL) / 32]) * (1.0f / 128.0f);
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const float pp = pbuf[warp][tt][r] * vs;
                psum[r] += pp;
                #pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(pp, vf[j], acc[r][j]);
            }
        }
    }

    if (!active || split >= num_splits) return;
    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        const float lt = gfx906_attn::wave_sum(l[r]);
        const int hb = r / BH;
        const int r16 = (r % BH) * BM + warp;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        float* po = partial_o + (pbase * BLOCK_ROWS + r16) * HD;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) po[lane * DPL + j] = acc[r][j] - 127.5f * psum[r];
        if (lane == 0)
        {
            partial_ml[(pbase * BLOCK_ROWS + r16) * 2] = m[r];
            partial_ml[(pbase * BLOCK_ROWS + r16) * 2 + 1] = lt;
        }
    }
}

#endif  // GFX906_VERIFY_Q8_LDS

typedef void (*fp_gfx906_attn)(const half*, const void*, const void*, const int*, const int*, half*, float*, float*,
                               const half*, const half*, const half*, int, int, int, const float*, void*, void*);

template <int QC, int HD, int NKV, int GROUP>
static fp_gfx906_attn pick_ql(int ql)
{
    #if GFX906_VERIFY_Q8_LDS
    if constexpr (QC == 8)
    {
        switch (ql)
        {
            case 2: return gfx906_attn_verify_q8_kernel<HD, 2, NKV, GROUP>;
            case 3: return gfx906_attn_verify_q8_kernel<HD, 3, NKV, GROUP>;
            case 4: return gfx906_attn_verify_q8_kernel<HD, 4, NKV, GROUP>;
            case 5: return gfx906_attn_verify_q8_kernel<HD, 5, NKV, GROUP>;
            case 6: return gfx906_attn_verify_q8_kernel<HD, 6, NKV, GROUP>;
            case 7: return gfx906_attn_verify_q8_kernel<HD, 7, NKV, GROUP>;
            case 8: return gfx906_attn_verify_q8_kernel<HD, 8, NKV, GROUP>;
        }
    }
    #endif
    switch (ql)
    {
        case 1:  return gfx906_attn_decode_split_kernel<QC, HD, 1, NKV, GROUP>;
        case 2:  return gfx906_attn_decode_split_kernel<QC, HD, 2, NKV, GROUP>;
        case 3:  return gfx906_attn_decode_split_kernel<QC, HD, 3, NKV, GROUP>;
        case 4:  return gfx906_attn_decode_split_kernel<QC, HD, 4, NKV, GROUP>;
        case 5:  return gfx906_attn_decode_split_kernel<QC, HD, 5, NKV, GROUP>;
        case 6:  return gfx906_attn_decode_split_kernel<QC, HD, 6, NKV, GROUP>;
        case 7:  return gfx906_attn_decode_split_kernel<QC, HD, 7, NKV, GROUP>;
        case 8:  return gfx906_attn_decode_split_kernel<QC, HD, 8, NKV, GROUP>;
        case 9:  return gfx906_attn_decode_split_kernel<QC, HD, 9, NKV, GROUP>;
        case 10: return gfx906_attn_decode_split_kernel<QC, HD, 10, NKV, GROUP>;
        case 11: return gfx906_attn_decode_split_kernel<QC, HD, 11, NKV, GROUP>;
        case 12: return gfx906_attn_decode_split_kernel<QC, HD, 12, NKV, GROUP>;
        case 13: return gfx906_attn_decode_split_kernel<QC, HD, 13, NKV, GROUP>;
        case 14: return gfx906_attn_decode_split_kernel<QC, HD, 14, NKV, GROUP>;
        case 15: return gfx906_attn_decode_split_kernel<QC, HD, 15, NKV, GROUP>;
        case 16: return gfx906_attn_decode_split_kernel<QC, HD, 16, NKV, GROUP>;
    }
    return nullptr;
}

template <int QC, int HD>
static fp_gfx906_attn pick_heads(int nkv, int group, int ql)
{
    if (nkv == 4 && group == 6) return pick_ql<QC, HD, 4, 6>(ql);
    if (nkv == 8 && group == 4) return pick_ql<QC, HD, 8, 4>(ql);
    if (nkv == 4 && group == 4) return pick_ql<QC, HD, 4, 4>(ql);
    return nullptr;
}

#endif

std::shared_ptr<TritonKernel> gfx906_attn_split_kernel(int qc, int head_dim, int q_len, int n_kv_heads, int group)
{
    #ifdef __HIP_PLATFORM_AMD__
        fp_gfx906_attn k = nullptr;
        if (qc == 8 && head_dim == 256) k = pick_heads<8, 256>(n_kv_heads, group, q_len);
        if (qc == 0 && head_dim == 256) k = pick_heads<0, 256>(n_kv_heads, group, q_len);
        if (qc == 8 && head_dim == 128) k = pick_heads<8, 128>(n_kv_heads, group, q_len);
        if (qc == 0 && head_dim == 128) k = pick_heads<0, 128>(n_kv_heads, group, q_len);
        if (!k) return nullptr;
        hipFunction_t fn;
        cuda_check(hipGetFuncBySymbol(&fn, (const void*) k));

        // Threads per block: 4 logical waves for decode, one per query position otherwise, rounded up
        // to whole 64-lane hardware waves (== NTH in the kernels). TritonKernel launches
        // num_warps * <wave size> threads, so num_warps has to be NTH / 64 (GFX906_TK_WARP)
        const int logical = q_len == 1 ? 4 : q_len;
        const int threads = ((logical + 1) / 2) * 64;
        const int waves = threads / GFX906_TK_WARP;
        return std::make_shared<TritonKernel>((void*) fn, std::string("gfx906_attn_decode_split"), waves, 0);
    #else
        return nullptr;
    #endif
}
