# Which submodule goes non-finite in the cached (KV-cache + recurrent state) path?
# Hooks GatedDeltaNet / GatedMLP / TransformerBlock outputs; one fp16 cache, graphs off.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ.setdefault("EXL3_NOGRAPH", "attn,mlp,gdn")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.mlp import GatedMLP
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

rec = []
def hook(cls, tag):
    orig = cls.forward
    def fwd(self, x, params, *a, **kw):
        y = orig(self, x, params, *a, **kw)
        z = y[0] if isinstance(y, tuple) else y
        f = z.detach().float()
        nf = int((~torch.isfinite(f)).sum())
        rec.append((tag, self.layer_idx, tuple(z.shape), nf, float(f.abs().max()) if nf < f.numel() else float("nan")))
        return y
    cls.forward = fwd
for c, t in ((GatedDeltaNet, "gdn"), (GatedMLP, "mlp")):
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
    log(f"cached prefill logits: nonfinite {int((~torch.isfinite(lg)).sum())} |max| {float(lg.abs().max()):.3f}")
    show("prefill")
    rec.clear()
    q = {"attn_mode": "flash_attn", "cache": cache, "past_len": 16, "batch_shape": (1, 4096),
         "recurrent_states": st}
    lg2 = model.forward(ids[:, 16:17], q).float()[0]
    log(f"decode logits: nonfinite {int((~torch.isfinite(lg2)).sum())} |max| {float(lg2.abs().max()):.3f} "
        f"top1 {int(lg2.argmax(-1))} {tok.decode(torch.tensor([[int(lg2.argmax(-1))]]))!r}")
    show("decode")

def show(phase):
    log(f"--- {phase}: first non-finite submodule outputs")
    n = 0
    for tag, idx, shape, nf, am in rec:
        if nf or am > 1e4:
            log(f"  {tag} L{idx} {shape} nf={nf} absmax={am:.3f}")
            n += 1
            if n > 6: break
    if n == 0: log("  (none)")
    log(f"  calls recorded: {len(rec)}")
    return

def _unused():
    log("first non-finite GDN outputs:")
    n = 0
    for tag, idx, shape, nf, am in rec:
        if nf or am > 1e4:
            log(f"  {tag} L{idx} {shape} nf={nf} absmax={am:.3f}")
            n += 1
            if n > 6: break
    if n == 0: log("  (none)")
    log(f"GDN calls recorded: {len(rec)}, L0 shapes {[r[2] for r in rec[:4]]}")

run()
