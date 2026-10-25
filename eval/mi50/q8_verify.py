# Q8 KV cache: decode (q_len 1) and verify (q_len 2..8) split kernels vs no-cache prefill reference.
import os, sys, math, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
P = 64
CHUNK = 4
STEPS = 16
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1,
              layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :P + STEPS + 8]
N = ids.shape[1]
log("ids", tuple(ids.shape))

ref = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float().cpu()[0]
params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
model.prefill(input_ids = ids[:, :P], params = params)
rs = params.get("recurrent_states")

outs, pos = [], P
while pos < N:
    c = min(CHUNK, N - pos)
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096),
         "recurrent_states": rs}
    outs.append(model.forward(input_ids = ids[:, pos:pos + c], params = p).float().cpu()[0])
    log(f"cached forward q_len={c} at pos {pos} ok")
    pos += c
torch.cuda.synchronize()
dec = torch.cat(outs, 0)
r = ref[P:N]

agree = (dec.argmax(-1) == r.argmax(-1)).float().mean().item()
lpr, lpd = torch.log_softmax(r, -1), torch.log_softmax(dec, -1)
kl = (lpr.exp() * (lpr - lpd)).sum(-1)
tgt = ids[0, P + 1:N]
log(f"Q8 cache, chunk {CHUNK}: top1 agree {agree:.3f}, KL mean {kl.mean().item():.4f} max {kl.max().item():.4f}")
log(f"ppl ref {math.exp(-lpr[:-1].gather(1, tgt.unsqueeze(1)).mean().item()):.3f} "
    f"dec {math.exp(-lpd[:-1].gather(1, tgt.unsqueeze(1)).mean().item()):.3f}")
