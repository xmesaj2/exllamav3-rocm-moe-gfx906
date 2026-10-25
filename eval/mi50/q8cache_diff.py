# Cached decode with a quantized (Q8) KV cache vs no-cache prefill reference.
# Exercises the q_cache kernels (their __shfl_sync absolute-lane reads were the wave64 fix).
import os, sys, math, argparse, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw")
ap.add_argument("--prefix", type = int, default = 200)
ap.add_argument("--steps", type = int, default = 24)
ap.add_argument("--chunk", type = int, default = 1)
ap.add_argument("--kb", type = int, default = 8)
ap.add_argument("--vb", type = int, default = 8)
ap.add_argument("--text", default = "/root/wikitext2_test.txt")
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1,
              layer_type = CacheLayer_quant, k_bits = args.kb, v_bits = args.vb)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
text = open(args.text).read()[5000:40000]
ids = tok.encode(text)[:, :args.prefix + args.steps]
N = ids.shape[1]

ref = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float().cpu()[0]
P = args.prefix
params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
model.prefill(input_ids = ids[:, :P], params = params)
rs = params.get("recurrent_states")
outs = []
pos = P
while pos < N:
    c = min(args.chunk, N - pos)
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096),
         "recurrent_states": rs}
    outs.append(model.forward(input_ids = ids[:, pos:pos + c], params = p).float().cpu()[0])
    pos += c
torch.cuda.synchronize()
dec = torch.cat(outs, 0)
r = ref[P:N]
agree = (dec.argmax(-1) == r.argmax(-1)).float().mean().item()
lpr = torch.log_softmax(r, -1); lpd = torch.log_softmax(dec, -1)
kl = (lpr.exp() * (lpr - lpd)).sum(-1)
tgt = ids[0, P + 1:N]
nll_r = -lpr[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
nll_d = -lpd[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
print(f"KV {args.kb}/{args.vb} chunk {args.chunk}: top1 agree {agree:.3f}, KL mean {kl.mean().item():.4f} "
      f"max {kl.max().item():.4f}, ppl ref {math.exp(nll_r):.3f} dec {math.exp(nll_d):.3f}", flush = True)
