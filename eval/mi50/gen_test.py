# End-to-end generation through the Generator (the path tabbyAPI drives), Q8 KV cache, greedy.
import os, sys, time, torch
sys.path.insert(0, "/mnt/storage/ai/exl3/exllamav3-rocm-moe")
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
from exllamav3.cache import CacheLayer_quant

M = "/mnt/storage/ai/exl3/tabbyAPI/models/Qwen3.8-27B-EXL3-3.5bpw"
def log(*a): print(*a, flush = True)

config = Config.from_directory(M)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192, max_batch_size = 1,
              layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
t0 = time.perf_counter()
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
log(f"loaded in {time.perf_counter() - t0:.1f} s")
gen = Generator(model = model, cache = cache, tokenizer = tok)

for prompt, n in (
    ("The capital of France is", 24),
    ("A list of three colors:\n1.", 32),
    ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"", 48),
):
    ids = tok.encode(prompt)
    job = Job(input_ids = ids, sampler = GreedySampler(), max_new_tokens = n,
              stop_conditions = [], decode_special_tokens = False)
    gen.enqueue(job)
    t0, out, steps = time.perf_counter(), "", 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            steps += 1
            if "text" in r: out += r["text"]
    dt = time.perf_counter() - t0
    log(f"--- {prompt!r} ({ids.shape[-1]} tok prompt, {steps} iters, {dt:.2f} s)")
    log(repr(out))
