# Unit test: ext.cuda_causal_conv1d_update (bf16, K=4, state_size=K) vs a torch reference.
import sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3.ext import exllamav3_ext as ext

def log(*a): print(*a, flush = True)
torch.manual_seed(0)

def ref_conv(state, x, w, K, S):
    # kernel: win[0..K-2] = state[1..K-1], win[K-1] = x[s]
    y = torch.cat([state[:, 1:K], x], dim = -1)             # (dim, K-1+S)
    acc = sum(y[:, k:k + S] * w[:, k].view(-1, 1) for k in range(K))
    return torch.nn.functional.silu(acc), y[:, -K:]

for dim, K, S, hist in ((10240, 4, 16, False), (10240, 4, 1, False), (10240, 4, 16, True),
                        (6144, 4, 3, False), (10240, 4, 128, False)):
    nslots, state_size = 2, K
    x = (torch.randn(1, dim, S, dtype = torch.bfloat16, device = "cuda") * 0.1)
    st0 = (torch.randn(nslots, dim, state_size, dtype = torch.bfloat16, device = "cuda") * 0.1)
    conv_state = st0.clone()
    w = (torch.randn(dim, K, dtype = torch.bfloat16, device = "cuda") * 0.2)
    slots = torch.tensor([1], dtype = torch.int32, device = "cuda")
    out = torch.empty(1, S, dim, dtype = torch.bfloat16, device = "cuda")

    ref, new_state = ref_conv(st0[1].float(), x[0].float(), w.float(), K, S)
    ext.cuda_causal_conv1d_update(x, conv_state, slots, w, None, out, True, hist)
    torch.cuda.synchronize()
    o = out[0].float().T
    d = (o - ref).abs()
    log(f"dim={dim} S={S:<4d} hist={hist}: |out|max {float(o.abs().max()):.4f} |ref|max {float(ref.abs().max()):.4f} "
        f"maxdiff {float(d.max()):.5f} meandiff {float(d.mean()):.6f}")
    exp = new_state if not hist else torch.cat([st0[1].float()[:, state_size - S:], x[0].float()], dim = -1)[:, -state_size:] if False else new_state
    log(f"    state write: maxdiff {float((conv_state[1].float() - exp).abs().max()):.5f} "
        f"slot0 untouched {float((conv_state[0] - st0[0]).abs().max()):.5f}")
