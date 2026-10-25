#pragma once
 
#include <memory>
struct TritonKernel;
 
// HIP flash-decoding split kernel for gfx906 (Vega 20: MI50 / MI60 / Radeon VII, wave64), drop-in
// for the Triton decode split kernel and for rdna3_attn_split_kernel (same arguments), or
// nullptr when the configuration is not covered
std::shared_ptr<TritonKernel> gfx906_attn_split_kernel(int qc, int head_dim, int q_len, int n_kv_heads, int group);