# Unit test: ext.cuda_recurrent_gated_delta_rule vs the torch reference in gated_delta_rule.py,
# with a non-zero initial state (the decode/prefill-with-state path).
import sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import torch_recurrent_gated_delta_rule

def log(*a): print(*a, flush = True)
torch.manual_seed(0)

b, s = 1, 16
hk, hv, dk, dv = 16, 48, 128, 128
q = torch.randn(b, s, hk, dk, dtype = torch.float, device = "cuda") * 0.1
k = torch.randn(b, s, hk, dk, dtype = torch.float, device = "cuda") * 0.1
v = torch.randn(b, s, hv, dv, dtype = torch.float, device = "cuda") * 0.1
g = -torch.rand(b, s, hv, dtype = torch.float, device = "cuda") * 0.5
beta = torch.sigmoid(torch.randn(b, s, hv, dtype = torch.float, device = "cuda"))
st0 = torch.randn(2, 1, hv, dk, dv, dtype = torch.float, device = "cuda") * 0.01

# reference
# grouped heads: each k head serves num_v_heads / num_k_heads v heads
grp = hv // hk
q_e = q.repeat_interleave(grp, dim = 2)
k_e = k.repeat_interleave(grp, dim = 2)
ref, ref_state = torch_recurrent_gated_delta_rule(
    q_e, k_e, v, g, beta, st0[1].unsqueeze(0), True, use_qk_l2norm_in_kernel = True)

# ext kernel
mixed_qkv = torch.cat([q.reshape(b, s, -1), k.reshape(b, s, -1), v.reshape(b, s, -1)], dim = -1)
mixed_qkv = mixed_qkv.to(torch.bfloat16).contiguous()
state = st0.clone()
beta_bf = beta.to(torch.bfloat16)
slots = torch.tensor([1], dtype = torch.int32, device = "cuda")
out = torch.empty(b, s, hv, dv, dtype = torch.bfloat16, device = "cuda")
ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g.contiguous(), beta_bf, state, out,
                                    hk, hv, dk, dv, slots, False, None)
torch.cuda.synchronize()

o = out.float()
lb = ref.float().abs().max().clamp_min(1e-9)
d = (o - ref.float()).abs()
log(f"out: |ext|max {float(o.abs().max()):.5f} |ref|max {float(ref.abs().max()):.5f} "
    f"maxdiff {float(d.max()):.6f} meandiff {float(d.mean()):.8f} rel {float(d.max() / lb):.3e} "
    f"nonfinite {int((~torch.isfinite(o)).sum())}")
ds = (state[1] - ref_state[0]).abs()
log(f"state: |ext|max {float(state[1].abs().max()):.5f} |ref|max {float(ref_state[0].abs().max()):.5f} "
    f"maxdiff {float(ds.max()):.6f} rel {float(ds.max() / ref_state[0].abs().max().clamp_min(1e-9)):.3e}")
log(f"slot 0 untouched: {float((state[0] - st0[0]).abs().max()):.6f}")
log(f"ext[:2,:3] {[round(float(x), 4) for x in o[0, 0, 0, :4]]}")
log(f"ref[:2,:3] {[round(float(x), 4) for x in ref.float()[0, 0, 0, :4]]}")
