import torch, sys, time
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3.ext import exllamav3_ext as ext

rows = int(sys.argv[1]); dim = int(sys.argv[2]); xd = sys.argv[3]; wd = sys.argv[4]; yd = sys.argv[5]
dt = {"half": torch.half, "bf16": torch.bfloat16, "float": torch.float32}
label = f"rows={rows} dim={dim} x={xd} w={wd} y={yd}"
torch.manual_seed(0)
x = torch.randn(rows, dim, dtype = dt[xd], device = "cuda")
w = torch.randn(dim, dtype = dt[wd], device = "cuda")
y = torch.empty(rows, dim, dtype = dt[yd], device = "cuda")
eps = 1e-6
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim = True) + eps) * w.float()).to(dt[yd])
torch.cuda.synchronize()
threads = min(1024, -(-(dim // 4) // 32) * 32)
print(f"launch {label} (threads={threads})", flush = True)
t = time.perf_counter()
ext.rms_norm(x, w, y, eps, 0.0, 1.0, False, False, 1)
torch.cuda.synchronize()
err = (y.float() - ref.float()).abs().max().item()
rel = err / ref.float().abs().max().item()
print(f"OK {label} {time.perf_counter() - t:.4f} s  maxabs={err:.3e} rel={rel:.1e}", flush = True)
