# Which stage of the gfx906 small-m kernel is wrong?  Reconstruct every candidate intermediate in
# torch from the same weights and see which one the kernel output matches.
#   correct      : y = had(xh @ W') * svh      (xh = had(x, suh), W' = rotated-basis weights)
#   no_out_had   : y = (xh @ W') * svh
#   no_in_had    : y = had(x @ W') * svh
#   raw_W        : y = had(xh @ W) * svh        (W = original-basis weights)
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
log(f"linear {path}: in {lin.in_features} out {lin.out_features} K {lin.K}")
H, O = lin.in_features, lin.out_features

Wp = lin.get_inner_weight_tensor()          # rotated basis (what the kernel decodes)
log(f"W' {tuple(Wp.shape)} {Wp.dtype} |W'|max {float(Wp.abs().max()):.4f}")
suh = lin.unpack_bf(lin.su).unsqueeze(1) if lin.su is not None else lin.suh.unsqueeze(1)
svh = lin.unpack_bf(lin.sv).unsqueeze(0) if lin.sv is not None else lin.svh.unsqueeze(0)
log(f"suh {tuple(suh.shape)} svh {tuple(svh.shape)}")

def had(x, post = None):
    y = torch.empty_like(x)
    ext.had_r_128(x, y, None, post, 1.0)
    return y

def candidates(x, rows):
    x = x.view(rows, H)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    red = (xh.float() @ Wp.float())
    out = {}
    out["correct"] = had(red.half(), svh).float()
    out["no_out_had"] = red * svh.float()
    out["no_in_had"] = had((x.float() @ Wp.float()).half(), svh).float()
    out["in_had_twice"] = had((had(xh.half()).float() @ Wp.float()).half(), svh).float()
    return out

def check(tag, x, rows):
    y_q = lin.forward(x.clone(), {}).float().view(rows, O)
    cand = candidates(x, rows)
    log(f"--- {tag}: |y_q|max {float(y_q.abs().max()):.4f}  nnz {int((y_q != 0).sum())}/{y_q.numel()}")
    for name, c in cand.items():
        d = (y_q - c).abs()
        log(f"    {name:<14} maxdiff {float(d.max()):10.5f} meandiff {float(d.mean()):10.6f} "
            f"|cand|max {float(c.abs().max()):.4f}")
    log(f"    y_q[:8]  {[round(float(v), 5) for v in y_q[0, :8].tolist()]}")
    log(f"    y_q[120:128] {[round(float(v), 5) for v in y_q[0, 120:128].tolist()]}")

torch.manual_seed(0)
x = torch.zeros(1, 1, H, dtype = torch.half, device = "cuda"); x[0, 0, 0] = 1.0
check("onehot i=0", x, 1)
x = torch.zeros(1, 1, H, dtype = torch.half, device = "cuda"); x[0, 0, 129] = 1.0
check("onehot i=129", x, 1)
x = torch.randn(1, 1, H, dtype = torch.half, device = "cuda") * 0.1
check("random rows=1", x, 1)
