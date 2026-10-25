import torch, sys, time
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3.ext import exllamav3_ext as ext

rows = int(sys.argv[1]); dim = int(sys.argv[2])
wd = {"half": torch.half, "bf16": torch.bfloat16}[sys.argv[3]]
yd = {"half": torch.half, "float": torch.float32}[sys.argv[4]]
label = f"rows={rows} dim={dim} w={sys.argv[3]} y={sys.argv[4]}"

x = torch.randn(rows, dim, dtype = torch.float32, device = "cuda")
w = torch.randn(dim, dtype = wd, device = "cuda")
y = torch.empty(rows, dim, dtype = yd, device = "cuda")
torch.cuda.synchronize()
print("launch", label, flush = True)
t = time.perf_counter()
ext.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False, 1)
torch.cuda.synchronize()
print(f"OK {label} {time.perf_counter() - t:.3f} s y00={float(y[0, 0]):.5f}", flush = True)
