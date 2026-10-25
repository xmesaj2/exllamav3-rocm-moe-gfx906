# Where does the on-the-fly EXL3 matmul break?  Compare it against the reconstruct path
# ({"reconstruct": True}, proven exact vs a torch x @ W reference) for a range of row counts.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3, AUTO_RECONSTRUCT_THRESHOLD

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
log(f"AUTO_RECONSTRUCT_THRESHOLD = {AUTO_RECONSTRUCT_THRESHOLD}")
ip = config.infer_params
log(f"infer_params.no_reconstruct = {getattr(ip, 'no_reconstruct', '?')}")

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
H = lin.in_features
log(f"linear {path}: in {lin.in_features} out {lin.out_features} K {lin.K}")

torch.manual_seed(0)
for rows in (1, 16, 64, 128, 144, 145, 160, 192, 256, 512, 1024, 2048):
    x = torch.randn(1, rows, H, dtype = torch.half, device = "cuda") * 0.1
    y_q = lin.forward(x.clone(), {}).float().view(rows, -1)
    y_r = lin.forward(x.clone(), {"reconstruct": True}).float().view(rows, -1)
    d = (y_q - y_r).abs()
    rel = float(d.max() / y_r.abs().max().clamp_min(1e-9))
    log(f"rows {rows:<5d} |y_q|max {float(y_q.abs().max()):9.4f} |y_r|max {float(y_r.abs().max()):9.4f} "
        f"maxdiff {float(d.max()):10.5f} rel {rel:.3e} {'OK' if rel < 5e-3 else '*** WRONG'}")
