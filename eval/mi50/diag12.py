# Bisect the BC attention path at the first full-attention layer: compare the K/V cache it writes and
# its output against the working (non-BC) path, on identical input.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ.setdefault("EXL3_NOGRAPH", "attn,mlp,gdn")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
import exllamav3.modules.attn as attn_mod
import exllamav3.modules.attention_fn.bc_attn as bcmod

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
LAYER = 3
def log(*a): print(*a, flush = True)
def stats(t, tag):
    f = t.detach().float()
    nf = int((~torch.isfinite(f)).sum())
    log(f"    {tag}: {tuple(t.shape)} nf={nf} max={float(f.abs().max()) if nf < f.numel() else float('nan'):.4f}")

cap = {}
_orig = attn_mod.Attention.forward
def hooked(self, x, params, *a, **kw):
    y = _orig(self, x, params, *a, **kw)
    if self.layer_idx == LAYER and "x" not in cap:
        cl = self.cache_layers[0]
        cap["x"] = x.detach().clone()
        cap["y"] = y.detach().float().clone()
        cap["k"] = cl.k[:1].detach().clone()
        cap["v"] = cl.v[:1].detach().clone()
    return y
attn_mod.Attention.forward = hooked

config = Config.from_directory(M)
model = Model.from_config(config)
caches = [Cache(model, max_num_tokens = 2048, max_batch_size = 1, layer_type = CacheLayer_fp16) for _ in range(2)]
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]

@torch.inference_mode()
def run(cache, bc, gfx):
    attn_mod._bc_attn_enable = bc
    bcmod._gfx906_attn_enable = gfx
    cap.clear()
    st = [cache.get_new_state()]
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 2048),
         "recurrent_states": st}
    lg = model.forward(ids[:, :16], p).float()[0]
    nf = int((~torch.isfinite(lg)).sum())
    out = dict(cap)
    out["logits_nf"] = nf
    out["logits_max"] = float(lg.abs().max()) if nf < lg.numel() else float("nan")
    cache.release_state(st[0])
    return out

runs = [
    ("BC on  (gfx906 kernels)", caches[0], True, True),
    ("BC on  (Triton kernels)", caches[1], True, False),
    ("BC off (reference)     ", caches[1], False, False),
]
res = {}
for tag, cache, bc, gfx in runs:
    log(tag)
    r = run(cache, bc, gfx)
    res[tag] = r
    log(f"    logits nf={r['logits_nf']} max={r['logits_max']:.4f}")
    for key in ("x", "y", "k", "v"):
        if key in r: stats(r[key], key)

ref = res["BC off (reference)     "]
log("=== diffs vs reference")
for tag, r in res.items():
    if tag.startswith("BC off"): continue
    log(f"  {tag}")
    for key in ("x", "y", "k", "v"):
        if key in r and key in ref:
            d = (r[key].float() - ref[key].float()).abs()
            nd = int((r[key].float() - ref[key].float()).ne(0).sum())
            log(f"    {key}: maxdiff {float(d.max()):.6f} differing {nd}/{d.numel()}")
