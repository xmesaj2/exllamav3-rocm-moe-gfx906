# Which causal_conv1d_update call faults, and with what shapes?
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ["EXL3_NOGRAPH"] = "attn,mlp,gdn"
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
import exllamav3.modules.gated_delta_net_fn.conv1d as conv1d_mod

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

_orig = conv1d_mod.causal_conv1d_update
_n = [0]
def wrapped(mixed_qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias, history = False, params = None):
    _n[0] += 1
    if _n[0] < 6 or (mixed_qkv.shape[1] > 100000):
        log(f"  conv#{_n[0]} x {tuple(mixed_qkv.shape)} {mixed_qkv.dtype} contig {mixed_qkv.is_contiguous()} "
            f"state {None if conv_state is None else tuple(conv_state.shape)} slots {None if recurrent_slots is None else tuple(recurrent_slots.shape)} "
            f"hist {history}")
    return _orig(mixed_qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias, history, params)
conv1d_mod.causal_conv1d_update = wrapped
import exllamav3.modules.gated_delta_net as gdn_mod
gdn_mod.causal_conv1d_update = wrapped

config = Config.from_directory(M)
model = Model.from_config(config)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :24]

log("A: no-cache prefill 24")
lg = model.forward(ids, {"attn_mode": "flash_attn_nc"})
torch.cuda.synchronize()
log(f"   ok, logits {tuple(lg.shape)} |max| {float(lg.float().abs().max()):.4f}")
_n[0] = 0
log("B: cached prefill 16 (fp16 cache)")
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = CacheLayer_fp16)
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
lg = model.forward(ids[:, :16], p)
torch.cuda.synchronize()
log(f"   ok, logits {tuple(lg.shape)} |max| {float(lg.float().abs().max()):.4f}")
_n[0] = 0
log("C: decode 1 token")
q = {"attn_mode": "flash_attn", "cache": cache, "past_len": 16, "batch_shape": (1, 4096),
     "recurrent_states": p.get("recurrent_states")}
lg = model.forward(ids[:, 16:17], q)
torch.cuda.synchronize()
log(f"   ok, logits {tuple(lg.shape)} |max| {float(lg.float().abs().max()):.4f}")
