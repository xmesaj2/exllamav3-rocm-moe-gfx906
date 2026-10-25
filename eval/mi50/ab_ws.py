# Inspect the gfx906 small-m kernel internals: transformed input xh and the per-split partial sums,
# compared against torch references built from the same weights.
# Run with EXL3_GFX906_TARGET_BLOCKS=160 -> splits = 2 for a 5120x10240 linear at rows = 1.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.ext import exllamav3_ext as ext

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)

found, seen = [], set()
def walk(obj, path, depth = 0):
    if id(obj) in seen: return
    if isinstance(obj, LinearEXL3):
        seen.add(id(obj)); found.append((path, obj)); return
    if depth > 2: return
    d = getattr(obj, "__dict__", None)
    if not d: return
    for k, v in d.items():
        if v is None or isinstance(v, (int, float, str, bool, bytes, torch.Tensor)): continue
        walk(v, path + "." + k, depth + 1)
for i, m in enumerate(model.modules):
    walk(m, f"mod{i}")
path, lin = next((p, l) for p, l in found if l.in_features == 5120 and l.out_features == 10240)
H, O = lin.in_features, lin.out_features
log(f"linear {path}: in {H} out {O}")
Wp = lin.get_inner_weight_tensor()
suh = lin.unpack_bf(lin.su).unsqueeze(1) if lin.su is not None else lin.suh.unsqueeze(1)
svh = lin.unpack_bf(lin.sv).unsqueeze(0) if lin.sv is not None else lin.svh.unsqueeze(0)

torch.manual_seed(0)
x = torch.randn(1, 1, H, dtype = torch.half, device = "cuda") * 0.1
xr = x.view(1, H)
xh_ref = torch.empty_like(xr)
ext.had_r_128(xr, xh_ref, suh, None, 1.0)

y = lin.forward(x.clone(), {}).float().view(1, O)
log(f"y: |y|max {float(y.abs().max()):.4f} nnz {int((y != 0).sum())}")

# --- transformed input, un-permuted from the kernel's LDS pair layout
order = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
inv = torch.argsort(torch.tensor(order, device = "cuda"))
xh_k = ext.exl3_gfx906_debug_xh(1, H, 0)
xh_nat = xh_k.view(1, -1, 16)[:, :, inv].reshape(1, H)
d = (xh_nat.float() - xh_ref.float()).abs()
log(f"xh: kernel vs had_r_128 -> maxdiff {float(d.max()):.6f} meandiff {float(d.mean()):.8f} "
    f"|ref|max {float(xh_ref.abs().max()):.6f} |ker|max {float(xh_nat.abs().max()):.6f}")
log(f"    ref[:8] {[round(float(v), 5) for v in xh_ref[0, :8].tolist()]}")
log(f"    ker[:8] {[round(float(v), 5) for v in xh_nat[0, :8].tolist()]}")

# --- per-split partial sums
splits = 2
ks0 = 160 * 16          # ks_per_split = 160 k-slices
ws = ext.exl3_gfx906_debug_ws(splits * O, 0).view(splits, 1, O)
ref0 = (xh_ref[:, :ks0].float() @ Wp[:ks0].float())
ref1 = (xh_ref[:, ks0:].float() @ Wp[ks0:].float())
for i, ref in ((0, ref0), (1, ref1)):
    part = ws[i]
    dd = (part - ref).abs()
    log(f"split {i}: |part|max {float(part.abs().max()):10.4f} |ref|max {float(ref.abs().max()):10.4f} "
        f"maxdiff {float(dd.max()):10.5f} meandiff {float(dd.mean()):10.6f} nnz {int((part != 0).sum())}")
    log(f"    part[:8] {[round(float(v), 5) for v in part[0, :8].tolist()]}")
    log(f"    ref [:8] {[round(float(v), 5) for v in ref[0, :8].tolist()]}")
