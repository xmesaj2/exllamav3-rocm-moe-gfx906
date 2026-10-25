# Single decode step only, for kernel attribution via AMD_LOG_LEVEL
import os, sys, time, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 1024, max_batch_size = 1)
model.load(progressbar = False)
torch.cuda.synchronize()
log("loaded")
ids = torch.tensor([[1234]], dtype = torch.long)
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 1024)}
log("decode 1 token ...")
lg = model.forward(input_ids = ids, params = p)
torch.cuda.synchronize()
log("decode ok", tuple(lg.shape))
