# Stage timing: load, no-cache prefill, cached prefill, decode steps.
import os, sys, time, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
t0 = time.perf_counter()
model.load(progressbar = False)
torch.cuda.synchronize()
log(f"load {time.perf_counter() - t0:.1f} s")
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :144]
log("ids", tuple(ids.shape))

t0 = time.perf_counter()
lg = model.forward(ids[:, :128], {"attn_mode": "flash_attn_nc"})
torch.cuda.synchronize()
log(f"no-cache prefill 128 tok {time.perf_counter() - t0:.2f} s", tuple(lg.shape))

p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
t0 = time.perf_counter()
model.prefill(input_ids = ids[:, :128], params = p)
torch.cuda.synchronize()
log(f"cached prefill 128 tok {time.perf_counter() - t0:.2f} s")
rs = p.get("recurrent_states")

for i in range(8):
    pos = 128 + i
    q = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096),
         "recurrent_states": rs}
    t0 = time.perf_counter()
    lg = model.forward(input_ids = ids[:, pos:pos + 1], params = q)
    torch.cuda.synchronize()
    log(f"decode step {i} {time.perf_counter() - t0:.3f} s argmax {lg[0, -1].argmax().item()}")
log("done")
