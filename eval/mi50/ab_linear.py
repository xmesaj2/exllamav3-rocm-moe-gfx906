# A/B the on-the-fly quantized matmul (exl3_gfx906_gemm) against the dequantize-then-matmul path,
# on the model's real weights. Same input, same weights: the two must agree.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
def cmp(tag, a, b):
    fa, fb = a.float(), b.float()
    d = (fa - fb).abs()
    log(f"  {tag}: |y|max {float(fa.abs().max()):.4f} |ref|max {float(fb.abs().max()):.4f} "
        f"maxdiff {float(d.max()):.4f} meandiff {float(d.mean()):.6f} "
        f"rel {float(d.max() / fb.abs().max().clamp_min(1e-9)):.3e}")

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)

blk, inst, idx = next(fm for fm in model.fwd_modules if type(fm[0]) is TransformerBlock)
mlp = blk.mlp
base = {"attn_mode": "flash_attn_nc", "layer_instance": inst}
rec = dict(base, reconstruct = True)

torch.manual_seed(0)
H = 5120
for name, x in (
    ("random rows=1", torch.randn(1, 1, H, dtype = torch.half, device = "cuda") * 0.1),
    ("random rows=16", torch.randn(1, 16, H, dtype = torch.half, device = "cuda") * 0.1),
    ("ones rows=1", torch.ones(1, 1, H, dtype = torch.half, device = "cuda")),
):
    log(name)
    cmp("mlp", mlp.forward(x, base), mlp.forward(x, rec))
    for nm in ("gate_proj", "up_proj"):
        g = getattr(mlp, nm)
        cmp(nm, g.forward(x, base), g.forward(x, rec))
    # down projection gets the same input in both cases (it is just another linear)
    cmp("down_proj", mlp.down_proj.forward(x, base), mlp.down_proj.forward(x, rec))
