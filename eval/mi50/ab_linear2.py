# Absolute reference for one EXL3 linear: y = x @ W with W dequantized by the model's own reconstruct,
# compared against the on-the-fly quantized path (BC_LinearEXL3 -> exl3_gfx906_gemm) and the
# dequantize-per-call path (reconstruct_hgemm).
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
def cmp(tag, y, ref):
    a, b = y.float(), ref.float()
    d = (a - b).abs()
    log(f"  {tag:<12} maxdiff {float(d.max()):10.4f} meandiff {float(d.mean()):10.6f} "
        f"reldiff {float(d.max() / b.abs().max().clamp_min(1e-9)):.3e}  y[:4] {[round(v, 4) for v in a.flatten()[:4]]}")

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)
blk, inst, idx = next(fm for fm in model.fwd_modules if type(fm[0]) is TransformerBlock)
mlp = blk.mlp
log(f"mlp: {type(mlp).__name__}")
lins = []
for k, v in vars(mlp).items():
    inner = getattr(v, "inner", None)
    if inner is not None:
        log(f"  {k}: {type(v).__name__} inner {type(inner).__name__} "
            f"in {getattr(inner, 'in_features', '?')} out {getattr(inner, 'out_features', '?')}")
        lins.append((k, v, inner))

base = {"attn_mode": "flash_attn_nc", "layer_instance": inst}
rec = dict(base, reconstruct = True)

torch.manual_seed(0)
H = getattr(lins[0][2], "in_features")
for rows in (1, 16):
    x = torch.randn(1, rows, H, dtype = torch.half, device = "cuda") * 0.1
    log(f"--- rows {rows}")
    for name, outer, inner in lins:
        W = inner.get_weight_tensor()               # (in, out), original basis
        y_t = (x.view(rows, H).float() @ W.float()).view(1, rows, -1)
        y_q = outer.forward(x, base).clone()        # quantized kernel
        y_r = outer.forward(x, rec).clone()         # dequantize + hgemm
        log(f" {name}: |y_t|max {float(y_t.abs().max()):.4f}")
        cmp("quant vs ref", y_q, y_t)
        cmp("recon vs ref", y_r, y_t)
        cmp("recon vs quant", y_r, y_q)
