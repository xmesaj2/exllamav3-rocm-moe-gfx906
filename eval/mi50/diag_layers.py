# Per-layer diagnostics: find the first module whose output goes non-finite / explodes.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

rec = []
_count = [0]
_orig = TransformerBlock.forward
def _hooked(self, x, params):
    i = _count[0]; _count[0] += 1
    y = _orig(self, x, params)
    f = y.detach().float()
    bad = int((~torch.isfinite(f)).sum())
    rec.append(dict(layer = i, shape = tuple(y.shape), nonfinite = bad,
                    absmax = float(f.abs().max()), mean = float(f.mean()),
                    last = [round(v, 3) for v in f[0, -1, :5].tolist()]))
    return y
TransformerBlock.forward = _hooked

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1,
              layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :17]

def show(tag):
    log(f"===== {tag}")
    prev = None
    for r in rec:
        flag = " <<<" if r["nonfinite"] or r["absmax"] > 1e4 else ""
        log(f"  L{r['layer']:2d} {str(r['shape']):>16} nf={r['nonfinite']:<8d} absmax={r['absmax']:11.3f} "
            f"mean={r['mean']:9.4f} last={r['last']}{flag}")
        prev = r
    rec.clear()

def logits_report(tag, lg):
    f = lg.float()
    bad = int((~torch.isfinite(f)).sum())
    lg0 = f[0, -1]
    am = int(lg0.argmax())
    log(f"{tag}: logits {tuple(lg.shape)} nonfinite {bad} absmax {float(f.abs().max()):.3f} "
        f"argmax {am} {tok.decode(torch.tensor([[am]]), decode_special_tokens = True)!r}")

# --- A: no-cache prefill (dispatch path)
_count[0] = 0
lg = model.forward(ids[:, :16], {"attn_mode": "flash_attn_nc"})
show("No-cache prefill, 16 tokens")
logits_report("no-cache", lg)

# --- B: cached prefill
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
_count[0] = 0
model.prefill(input_ids = ids[:, :16], params = p)
show("Cached prefill, 16 tokens")

# --- C: one decode step
q = {"attn_mode": "flash_attn", "cache": cache, "past_len": 16, "batch_shape": (1, 4096),
     "recurrent_states": p.get("recurrent_states")}
_count[0] = 0
lg = model.forward(input_ids = ids[:, 16:17], params = q)
show("Decode 1 token")
logits_report("decode", lg)

# --- D: comparison of the same position via a long cached prefill
cache2 = Cache(model, max_num_tokens = 4096, max_batch_size = 1)
p2 = {"attn_mode": "flash_attn", "cache": cache2, "past_len": 0, "batch_shape": (1, 4096)}
lg2 = model.prefill(input_ids = ids[:, :17], params = p2)
lg2 = model.forward(input_ids = ids[:, :17], params = p2)
logits_report("cached forward 17 tokens (fp16 cache)", lg2)
