# Same MLP, same input, different params: which params key makes it produce NaN?
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
os.environ.setdefault("EXL3_NOGRAPH", "attn,mlp,gdn")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.transformer import TransformerBlock

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1, layer_type = CacheLayer_fp16)
model.load(progressbar = False)

blk, inst, idx = next(fm for fm in model.fwd_modules if type(fm[0]) is TransformerBlock)
mlp = blk.mlp
log(f"mlp idx {idx} instance {inst}")

@torch.inference_mode()
def run():
    st = [cache.get_new_state()]
    torch.manual_seed(0)
    x = (torch.randn(1, 16, 5120, dtype = torch.half, device = "cuda") * 0.1)
    variants = [
        ("nc minimal", {"attn_mode": "flash_attn_nc", "layer_instance": inst}),
        ("nc + batch_shape", {"attn_mode": "flash_attn_nc", "layer_instance": inst,
                              "batch_shape": (1, 4096), "past_len": 0}),
        ("cache", {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                   "batch_shape": (1, 4096), "layer_instance": inst}),
        ("cache + states", {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                            "batch_shape": (1, 4096), "layer_instance": inst, "recurrent_states": st}),
        ("nc + states", {"attn_mode": "flash_attn_nc", "layer_instance": inst, "recurrent_states": st}),
        ("cache again", {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                         "batch_shape": (1, 4096), "layer_instance": inst}),
    ]
    for tag, p in variants:
        y = mlp.forward(x.clone(), dict(p))
        f = y.float()
        nf = int((~torch.isfinite(f)).sum())
        log(f"  {tag:<18} shape {tuple(y.shape)} nf={nf:<7d} absmax={float(f.abs().max()):12.3f} "
            f"y[0,0,:4]={[round(float(v), 4) for v in f[0, 0, :4]]}")
    cache.release_state(st[0])

run()
