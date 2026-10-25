import torch, sys, time
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3.ext import exllamav3_ext as ext

rows = int(sys.argv[1]); dim = int(sys.argv[2]); xd = sys.argv[3]; wd = sys.argv[4]; yd = sys.argv[5]
dt = {"half": torch.half, "bf16": torch.bfloat16, "float": torch.float32}
label = f"rows={rows} dim={dim} x={xd} w={wd} y={yd}"
x = torch.randn(rows, dim, dtype = dt[xd], device = "cuda")
w = torch.randn(dim, dtype = dt[wd], device = "cuda")
y = torch.empty(rows, dim, dtype = dt[yd], device = "cuda")
torch.cuda.synchronize()
threads = min(1024, -(-(dim // 4) // 32) * 32)
print(f"launch {label} (threads={threads})", flush = True)
t = time.perf_counter()
ext.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False, 1)
torch.cuda.synchronize()
print(f"OK {label} {time.perf_counter() - t:.3f} s", flush = True)
