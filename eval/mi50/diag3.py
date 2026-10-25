# Cached (KV-cache) forward vs a no-cache reference: prefill rows and single-token decode rows,
# with and without the graph-captured decode paths.  Q8 and fp16 caches.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ["EXL3_NOGRAPH"] = "attn,mlp,gdn"
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.cache.fp16 import CacheLayer_fp16

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]
N = ids.shape[1]
log(f"ids {N} tokens")

lg_nc = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
def report(tag, lg):
    nf = int((~torch.isfinite(lg)).sum())
    d = (lg - lg_nc).abs()
    ok = (lg.argmax(-1) == lg_nc.argmax(-1))
    log(f"{tag}: nonfinite {nf} maxdiff {float(d.max()):9.4f} "
        f"top1 {int(ok.sum())}/{N} rows {[i for i, v in enumerate(ok.tolist()) if not v][:6]}")
    return d

def cached(cache, npre, graphs):
    os.environ["EXL3_NOGRAPH"] = "" if graphs else "attn,mlp,gdn"
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
    rows = [model.forward(ids[:, :npre], p).float()[0]]
    rs = p.get("recurrent_states")
    for i in range(npre, N):
        q = {"attn_mode": "flash_attn", "cache": cache, "past_len": i,
             "batch_shape": (1, 4096), "recurrent_states": rs}
        rows.append(model.forward(ids[:, i:i + 1], q).float()[0])
    return torch.cat(rows, 0)

for tag, kind, kw in (("fp16", CacheLayer_fp16, {}), ("Q8", CacheLayer_quant, dict(k_bits = 8, v_bits = 8))):
    for graphs in (False, True):
        cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = kind, **kw)
        lg = cached(cache, 16, graphs)
        d = report(f"{tag} cache, graphs {'on ' if graphs else 'off'}", lg)
        per_row = d.max(-1).values
        log(f"    per-row maxdiff: " + " ".join(f"{float(v):.3f}" for v in per_row))
        del cache
        torch.cuda.empty_cache()
