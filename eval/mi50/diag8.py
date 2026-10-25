# Cached prefill: record input/output stats for every GDN / MLP / block, in execution order,
# stopping at the first non-finite value.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/OMV/ai/exl3/exllamav3-rocm-moe")
os.environ.setdefault("EXL3_NOGRAPH", "attn,mlp,gdn")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.mlp import GatedMLP
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/OMV/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

rec = []
def stats(t):
    f = t.detach().float()
    nf = int((~torch.isfinite(f)).sum())
    return nf, (float(f.abs().max()) if nf < f.numel() else float("nan"))

def hook(cls, tag):
    orig = cls.forward
    def fwd(self, x, params, *a, **kw):
        ni, ai = stats(x)
        y = orig(self, x, params, *a, **kw)
        z = y[0] if isinstance(y, tuple) else y
        no, ao = stats(z)
        rec.append((tag, getattr(self, "key", "?"), ni, ai, no, ao))
        return y
    cls.forward = fwd
for c, t in ((GatedDeltaNet, "gdn"), (GatedMLP, "mlp"), (TransformerBlock, "blk")):
    hook(c, t)

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
    log(f"cached prefill logits nf {int((~torch.isfinite(lg)).sum())}")
    for i, (tag, key, ni, ai, no, ao) in enumerate(rec):
        flag = " <<<" if (ni or no or ai > 1e4 or ao > 1e4) else ""
        log(f"  [{i:3d}] {tag} {key[-28:]:<28} in nf={ni:<6d} max={ai:10.3f} | out nf={no:<6d} max={ao:10.3f}{flag}")
        if flag: break
    cache.release_state(st[0])

run()
