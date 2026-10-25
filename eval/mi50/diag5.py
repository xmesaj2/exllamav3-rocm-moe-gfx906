# Cached forward vs no-cache reference: prefill rows + single-token decode rows, for fp16/Q8 caches
# with and without the graph-captured decode paths. Caches are created BEFORE model.load() so the
# recurrent (GDN) state tensors get allocated.
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
cases = [
    ("fp16 graphs off", CacheLayer_fp16, {}, False),
    ("fp16 graphs on ", CacheLayer_fp16, {}, True),
    ("Q8   graphs off", CacheLayer_quant, dict(k_bits = 8, v_bits = 8), False),
    ("Q8   graphs on ", CacheLayer_quant, dict(k_bits = 8, v_bits = 8), True),
]
caches = [(tag, Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = lt, **kw), gr)
          for tag, lt, kw, gr in cases]
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]
N = ids.shape[1]

lg_nc = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
log(f"no-cache reference: logits {tuple(lg_nc.shape)} |max| {float(lg_nc.abs().max()):.3f} "
    f"top1 {[tok.decode(torch.tensor([[int(i)]])) for i in lg_nc[:4].argmax(-1)]}")

@torch.inference_mode()
def run_case(tag, cache, graphs):
    os.environ["EXL3_NOGRAPH"] = "" if graphs else "attn,mlp,gdn"
    st = [cache.get_new_state()]
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096),
         "recurrent_states": st}
    rows = [model.forward(ids[:, :16], p).float()[0]]
    for i in range(16, N):
        q = {"attn_mode": "flash_attn", "cache": cache, "past_len": i, "batch_shape": (1, 4096),
             "recurrent_states": st}
        rows.append(model.forward(ids[:, i:i + 1], q).float()[0])
    lg = torch.cat(rows, 0)
    d = (lg - lg_nc).abs()
    per_row = d.max(-1).values
    ok = (lg.argmax(-1) == lg_nc.argmax(-1))
    log(f"{tag}: nonfinite {int((~torch.isfinite(lg)).sum())} maxdiff {float(d.max()):9.4f} top1 {int(ok.sum())}/{N}")
    log("    per-row maxdiff: " + " ".join(f"{float(v):.3f}" for v in per_row))
    log(f"    decoded: {[tok.decode(torch.tensor([[int(i)]])) for i in lg[16:20].argmax(-1)]} "
        f"vs ref {[tok.decode(torch.tensor([[int(i)]])) for i in lg_nc[16:20].argmax(-1)]}")
    cache.release_state(st[0])

for tag, cache, graphs in caches:
    run_case(tag, cache, graphs)
