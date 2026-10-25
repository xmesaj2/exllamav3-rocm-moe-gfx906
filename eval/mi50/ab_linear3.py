# Absolute reference for one EXL3 linear.
#   y_t   = x @ W                     (W = get_weight_tensor(), original basis, torch fp32)
#   y_q   = LinearEXL3.forward(x, {})            -> exl3_gfx906_gemm (rows <= threshold)
#   y_rec = LinearEXL3.forward(x, {reconstruct}) -> dequantize + hgemm_recon
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3

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

log(f"found {len(found)} LinearEXL3")
for path, lin in found[:4]:
    log(f"  {path}: in {lin.in_features} out {lin.out_features} K {lin.K} key {getattr(lin, 'key', None)}")

# biggest linear (the MLP gate/up: 5120 -> 17408)
path, lin = max(found, key = lambda pl: pl[1].in_features * pl[1].out_features)
log(f"testing {path}: in {lin.in_features} out {lin.out_features}")
H, O = lin.in_features, lin.out_features
W = lin.get_weight_tensor()
log(f"  W {tuple(W.shape)} {W.dtype} |W|max {float(W.abs().max()):.4f}")

def run(name, x):
    rows = x.numel() // H
    y_t = (x.view(rows, H).float() @ W.float()).view(1, rows, O)
    y_q = lin.forward(x.clone(), {}).float().view(1, rows, O)
    y_r = lin.forward(x.clone(), {"reconstruct": True}).float().view(1, rows, O)
    for tag, y in (("quant", y_q), ("recon", y_r)):
        d = (y - y_t).abs()
        log(f"  {name} {tag:<6} |y|max {float(y.abs().max()):10.4f} maxdiff {float(d.max()):10.5f} "
            f"meandiff {float(d.mean()):10.6f} rel {float(d.max() / y_t.abs().max().clamp_min(1e-9)):.3e}")
    return y_t, y_q, y_r

torch.manual_seed(0)
for rows in (1, 3, 16):
    x = torch.randn(1, rows, H, dtype = torch.half, device = "cuda") * 0.1
    y_t, y_q, y_r = run(f"rand rows={rows:<4d}", x)

# one-hot: y must equal row 0 of W (a single input feature) -> shows which output columns are wrong
x = torch.zeros(1, 1, H, dtype = torch.half, device = "cuda")
x[0, 0, 0] = 1.0
y_t, y_q, y_r = run("onehot i=0     ", x)
for tag, y in (("quant", y_q[0, 0]), ("recon", y_r[0, 0])):
    err = (y - y_t[0, 0]).abs()
    blocks = err.view(-1, 128).max(-1).values
    log(f"  onehot {tag}: per-128-block max err -> {[round(float(v), 4) for v in blocks[:10].tolist()]} ... "
        f"worst block {int(blocks.argmax())}")
    log(f"    y[:8]     {[round(float(v), 5) for v in y[:8].tolist()]}")
    log(f"    ref[:8]   {[round(float(v), 5) for v in y_t[0, 0][:8].tolist()]}")

# a second one-hot further along the input
x = torch.zeros(1, 1, H, dtype = torch.half, device = "cuda")
x[0, 0, 129] = 1.0
y_t, y_q, y_r = run("onehot i=129   ", x)
