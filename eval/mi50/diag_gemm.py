# A linear layer's output rows must not depend on how many rows are in the batch.
# Compare MLP rows 0..15 computed at M=16 with the same rows computed at M=512.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)

blk, inst, idx = next(fm for fm in model.fwd_modules if type(fm[0]) is TransformerBlock)
mlp = blk.mlp
log(f"mlp {type(mlp).__name__} idx {idx}")

torch.manual_seed(0)
H = 5120
x16 = (torch.randn(1, 16, H, dtype = torch.half, device = "cuda") * 0.1)
p = {"attn_mode": "flash_attn_nc", "layer_instance": inst}

x512 = torch.zeros(1, 512, H, dtype = torch.half, device = "cuda")
x512[:, :16] = x16
ref = mlp.forward(x512, p).float()
log(f"ref M=512: absmax {float(ref.abs().max()):.3f}")

for S in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024):
    xs = torch.zeros(1, S, H, dtype = torch.half, device = "cuda")
    rows = min(S, 16)
    xs[:, :rows] = x16[:, :rows]
    y = mlp.forward(xs, p).float()
    d = (y[:, :rows] - ref[:, :rows]).abs()
    rel = d.max() / ref[:, :rows].abs().max().clamp_min(1e-9)
    log(f"M={S:<5d} rows {rows:<3d} absmax {float(y.abs().max()):10.3f} "
        f"rowdiff max {float(d.max()):10.4f} rel {float(rel):.3e}")

# also: are the rows past 16 of the M=512 run equal to a run of their own?
x496 = torch.zeros(1, 496, H, dtype = torch.half, device = "cuda")
x496[:, :] = x512[:, 16:]
y496 = mlp.forward(x496, p).float()
d2 = (y496 - ref[:, 16:]).abs()
log(f"M=512 rows 16..511 vs a separate M=496 run: max diff {float(d2.max()):.4f}")
