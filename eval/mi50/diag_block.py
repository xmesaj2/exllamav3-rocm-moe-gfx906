# Isolate the first TransformerBlock: does its output depend on sequence length?
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
def stats(tag, t):
    f = t.detach().float()
    log(f"{tag}: shape {tuple(f.shape)} absmax {float(f.abs().max()):12.3f} mean {float(f.mean()):9.4f} "
        f"nf {int((~torch.isfinite(f)).sum())}")

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)

for i, (m, inst, idx) in enumerate(model.fwd_modules[:6]):
    ch = [n for n, _ in m.named_children()] if hasattr(m, "named_children") else []
    log(f"fwd_modules[{i}] {type(m).__name__} idx {idx} children {ch}")

blk, inst, idx = next(fm for fm in model.fwd_modules if type(fm[0]) is TransformerBlock)
log(f"first block idx {idx}, attn {type(blk.attn).__name__} mlp {type(blk.mlp).__name__}")
for name, child in (blk.named_children() if hasattr(blk, "named_children") else []):
    log(f"  child {name}: {type(child).__name__}")

ids = tok.encode("The capital of France is a city in Europe.")[:, :16]
x0 = model.prepare_inputs(ids, {"attn_mode": "flash_attn_nc"})
stats("embedding", x0)

for S in (1, 2, 4, 8, 16, 17, 32, 64, 128, 512, 2048):
    x = torch.randn(1, S, 5120, dtype = torch.half, device = "cuda") * 0.1
    p = {"attn_mode": "flash_attn_nc", "layer_instance": inst}
    y = blk.forward(x, p)
    stats(f"block0 S={S:<5d}", y)

for S in (8, 16, 32, 2048):
    x = torch.randn(1, S, 5120, dtype = torch.half, device = "cuda") * 0.1
    p = {"attn_mode": "flash_attn_nc", "layer_instance": inst}
    a = blk.attn.forward(x, p)
    stats(f"  attn S={S:<5d}", a)
for S in (8, 16, 2048):
    x = torch.randn(1, S, 5120, dtype = torch.half, device = "cuda") * 0.1
    p = {"attn_mode": "flash_attn_nc", "layer_instance": inst}
    a = blk.mlp.forward(x, p)
    stats(f"  mlp  S={S:<5d}", a)
