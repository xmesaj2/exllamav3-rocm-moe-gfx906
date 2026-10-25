# Inside a real cached prefill, compare each full-attention layer's cached output with a no-cache
# recompute of the very same input (valid at past_len 0).
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ.setdefault("EXL3_NOGRAPH", "attn,mlp,gdn")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.attn import Attention

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
def stats(t):
    f = t.detach().float()
    nf = int((~torch.isfinite(f)).sum())
    return nf, (float(f.abs().max()) if nf < f.numel() else float("nan"))

out = []
_orig = Attention.forward
def hooked(self, x, params, *a, **kw):
    y = _orig(self, x, params, *a, **kw)
    if len(out) < 3:
        ni, ai = stats(x)
        no, ao = stats(y)
        # no-cache recompute of the same input (full causal attention, equivalent at past_len 0)
        ref = _orig(self, x.clone(), {"attn_mode": "flash_attn_nc", "layer_instance": params.get("layer_instance", 0),
                                      "batch_shape": tuple(x.shape[:2])})
        nr, ar = stats(ref)
        d = float("nan")
        if no == 0 and nr == 0:
            d = float((y.float() - ref.float()).abs().max())
        out.append((self.key[-24:], tuple(x.shape), ni, ai, no, ao, nr, ar, d))
    return y
Attention.forward = hooked

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = CacheLayer_fp16)
model.load(progressbar = False)
out.clear()
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]

@torch.inference_mode()
def run():
    st = [cache.get_new_state()]
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096),
         "recurrent_states": st}
    lg = model.forward(ids[:, :16], p).float()[0]
    log(f"cached prefill logits nf {int((~torch.isfinite(lg)).sum())}")
    for k, shape, ni, ai, no, ao, nr, ar, d in out:
        log(f"  {k:<24} x{shape} in nf={ni} max={ai:8.3f} | cached out nf={no} max={ao:10.3f} "
            f"| nc out nf={nr} max={ar:8.3f} | maxdiff {d:.5f}")
    cache.release_state(st[0])

run()
