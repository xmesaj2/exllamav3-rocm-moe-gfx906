# Cached prefill + decode, printing finiteness and top-1 of every step. Env selects the attention path.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
log(f"env: NOGRAPH={os.environ.get('EXL3_NOGRAPH','')} BC_ATTN={os.environ.get('EXL3_BC_ATTN','1')} "
    f"GFX906_ATTN={os.environ.get('EXL3_GFX906_ATTN','1')}")

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = CacheLayer_fp16)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]

@torch.inference_mode()
def run():
    st = [cache.get_new_state()]
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096),
         "recurrent_states": st}
    lg = model.forward(ids[:, :16], p).float()[0]
    nf = int((~torch.isfinite(lg)).sum())
    log(f"prefill 16: nf {nf} |max| {float(lg.abs().max()):.3f} "
        f"top1 {[tok.decode(torch.tensor([[int(i)]])) for i in lg[-2:].argmax(-1)]}")
    for i in range(16, 20):
        q = {"attn_mode": "flash_attn", "cache": cache, "past_len": i, "batch_shape": (1, 4096),
             "recurrent_states": st}
        o = model.forward(ids[:, i:i + 1], q).float()[0]
        nf = int((~torch.isfinite(o)).sum())
        log(f"decode @{i}: nf {nf} |max| {float(o.abs().max()):.3f} top1 {int(o.argmax(-1))} "
            f"{tok.decode(torch.tensor([[int(o.argmax(-1))]]))!r}  (ref {tok.decode(ids[0, i+1:i+2])!r})")
    cache.release_state(st[0])

run()
