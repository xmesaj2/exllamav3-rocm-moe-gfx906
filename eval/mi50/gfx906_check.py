# gfx906 bring-up check: load, prefill, cached decode vs no-cache logits, free-run continuation.
import os, sys, time, argparse, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw")
ap.add_argument("--cache", type = int, default = 8192)
ap.add_argument("--chunk", type = int, default = 2048)
ap.add_argument("--len", type = int, default = 1024)
ap.add_argument("--steps", type = int, default = 24)
ap.add_argument("--gen", type = int, default = 48)
ap.add_argument("--notext", action = "store_true")
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)
q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1, **q8)
t0 = time.perf_counter()
model.load(progressbar = False, max_chunk_size = args.chunk)
torch.cuda.synchronize()
print(f"LOAD ok in {time.perf_counter() - t0:.1f} s", flush = True)

tok = Tokenizer.from_config(config)
text = open("/mnt/storage/ai/exl3/tabbyAPI/docs/03.-Usage.md").read()[2000:12000]
ids = tok.encode(text)[:, :args.len + args.steps + 8]
print("prompt tokens", ids.shape, flush = True)

# 1. prefill without cache participation, all positions at once
t0 = time.perf_counter()
logits_pf = model.forward(ids[:, :args.len + args.steps], {"attn_mode": "flash_attn_nc"})
torch.cuda.synchronize()
print(f"PREFILL {args.len + args.steps} tok in {time.perf_counter() - t0:.2f} s", flush = True)

# 2. cached decode of the same positions
gen = Generator(model = model, cache = cache, tokenizer = tok, max_chunk_size = args.chunk)
job = Job(input_ids = ids[:, :args.len], max_new_tokens = 1, sampler = ComboSampler(temperature = 0.0, top_k = 1),
          stop_conditions = [])
# feed the decode steps manually through the model to compare logits position by position
model.forward(ids[:, :args.len], {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                                  "batch_shape": (1, args.len)})
torch.cuda.synchronize()
maxerr = 0.0
for i in range(args.steps):
    pos = args.len + i
    lg = model.forward(ids[:, pos:pos + 1], {"attn_mode": "flash_attn", "cache": cache, "past_len": pos,
                                             "batch_shape": (1, 1)})
    torch.cuda.synchronize()
    ref = logits_pf[:, pos]
    err = (lg[:, -1].float() - ref.float()).abs().max().item()
    maxerr = max(maxerr, err)
    if i < 4 or i == args.steps - 1:
        am_ref = ref.argmax().item(); am = lg[0, -1].argmax().item()
        print(f"  step {i}: maxabs {err:.4f}  argmax prefill {am_ref} decode {am} "
              f"({tok.decode_special(torch.tensor([am_ref]))!r} vs {tok.decode_special(torch.tensor([am]))!r})",
              flush = True)
print(f"DECODE-vs-PREFILL max |dlogit| over {args.steps} steps: {maxerr:.4f}", flush = True)

if not args.notext:
    job = Job(input_ids = ids[:, :args.len], max_new_tokens = args.gen,
              sampler = ComboSampler(temperature = 0.7, top_k = 50, top_p = 0.9), stop_conditions = [])
    t0 = time.perf_counter()
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            if res.get("token_ids") is not None and res["token_ids"].numel():
                out.append(res["token_ids"][0, -1].item())
    dt = time.perf_counter() - t0
    print(f"GENERATE {len(out)} tok in {dt:.2f} s ({len(out) / dt:.1f} tok/s):", flush = True)
    print(tok.decode(torch.tensor(out)), flush = True)
