# Locate non-finite logits: no-cache prefill vs Q8 cached decode.
import os, sys, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)
def nan_report(tag, x):
    bad = ~torch.isfinite(x)
    rows = bad.any(-1).nonzero().flatten().tolist()
    log(f"{tag}: shape {tuple(x.shape)} nonfinite {int(bad.sum())} rows {rows[:12]}"
        f"{' ...' if len(rows) > 12 else ''}")
    if rows:
        r0 = rows[0]
        row = x[r0]
        log(f"  row {r0}: finite {int(torch.isfinite(row).sum())}/{row.numel()} "
            f"min {row[torch.isfinite(row)].min().item():.2f} max {row[torch.isfinite(row)].max().item():.2f}")

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096, max_batch_size = 1,
              layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
P, CH = 64, 4
ids = tok.encode(open("/root/wikitext2_test.txt").read()[5000:30000])[:, :P + 16]
N = ids.shape[1]

ref = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float().cpu()[0]
nan_report("no-cache prefill", ref)

params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, 4096)}
model.prefill(input_ids = ids[:, :P], params = params)
rs = params.get("recurrent_states")
outs, pos = [], P
while pos < N:
    c = min(CH, N - pos)
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": (1, 4096),
         "recurrent_states": rs}
    outs.append(model.forward(input_ids = ids[:, pos:pos + c], params = p).float().cpu()[0])
    pos += c
dec = torch.cat(outs, 0)
nan_report("cached decode", dec)
a, b = ref[P:N], dec
fin = torch.isfinite(a) & torch.isfinite(b)
log(f"rows compared {a.shape[0]}, finite entry pairs {int(fin.sum())}/{fin.numel()}")
if fin.any():
    d = (a - b)[fin].abs()
    log(f"max abs diff {d.max().item():.4f}, mean {d.mean().item():.6f}")
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    log(f"top1 agree {top1:.3f}")
