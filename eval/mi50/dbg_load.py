import os, sys, traceback, time, argparse
sys.path.insert(0, "/mnt/omv/ai/exl3/exllamav3-rocm-moe")
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = "/mnt/storage/omv/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw")
ap.add_argument("--cache", type = int, default = 8192)
ap.add_argument("--chunk", type = int, default = 2048)
ap.add_argument("--prefill", type = int, default = 0)
ap.add_argument("--dumpall", action = "store_true")
args = ap.parse_args()

from exllamav3.ext import exllamav3_ext as ext

def wrap(name):
    f = getattr(ext, name)
    def w(*a, **kw):
        if torch.is_tensor(a[0]):
            x = a[0]
            big = x.numel() > 4_000_000
            if big or args.dumpall:
                print(f"[{name}] x={tuple(x.shape)} {x.dtype} contig={x.is_contiguous()}"
                      f" w={tuple(a[1].shape) if torch.is_tensor(a[1]) else a[1]}"
                      f" y={tuple(a[2].shape) if torch.is_tensor(a[2]) else None} {a[2].dtype if torch.is_tensor(a[2]) else ''}",
                      flush = True)
            if big:
                traceback.print_stack()
        return f(*a, **kw)
    setattr(ext, name, w)

for n in ("rms_norm", "rms_norm_res_in", "rms_norm_had", "gated_rms_norm"):
    if hasattr(ext, n):
        wrap(n)
    else:
        print("no such ext fn:", n)

config = Config.from_directory(args.model)
model = Model.from_config(config)
q8 = dict(layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1, **q8)
t0 = time.perf_counter()
model.load(progressbar = False, max_chunk_size = args.chunk)
torch.cuda.synchronize()
print(f"loaded in {time.perf_counter() - t0:.1f} s", flush = True)

if args.prefill:
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_chunk_size = args.chunk)
    ids = torch.randint(1000, 200000, (1, args.prefill), dtype = torch.long, device = "cuda")
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = ComboSampler(temperature = 0.0, top_k = 1), stop_conditions = [])
    gen.enqueue(job)
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            pass
    torch.cuda.synchronize()
    print(f"prefill {args.prefill} tokens in {time.perf_counter() - t0:.2f} s", flush = True)
