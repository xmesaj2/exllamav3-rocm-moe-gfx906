import os, re, sys

ROOT = "/mnt/storage/ai/exl3/exllamav3-rocm-moe/exllamav3/exllamav3_ext"
SYNC = {"__shfl_sync": 3, "__shfl_xor_sync": 3, "__shfl_up_sync": 3, "__shfl_down_sync": 3}
PLAIN = {"__shfl": 2, "__shfl_xor": 2, "__shfl_up": 2, "__shfl_down": 2}
ALL = {}
ALL.update(SYNC); ALL.update(PLAIN)

# longest names first so __shfl_sync is not matched as __shfl
names = sorted(ALL, key = len, reverse = True)
pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in names) + r")\s*\(")


def split_args(s, start):
    """s[start] is just after '('. Returns (args, end_index_of_close_paren)."""
    depth = 0
    args, cur = [], []
    i = start
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                args.append("".join(cur))
                return args, i
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    raise RuntimeError("unbalanced")


total = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in sorted(filenames):
        if not fn.endswith((".cu", ".cuh", ".h", ".cpp")):
            continue
        if fn.endswith("_hip.cuh"):
            continue
        path = os.path.join(dirpath, fn)
        src = open(path, encoding = "utf8", errors = "surrogateescape").read()
        out, pos, changed = [], 0, []
        while True:
            m = pat.search(src, pos)
            if not m:
                out.append(src[pos:])
                break
            name = m.group(1)
            args, close = split_args(src, m.end())
            want = ALL[name]
            out.append(src[pos:m.end()])
            if len(args) == want:
                out.append(",".join(args) + ", 32")
                changed.append((src.count("\n", 0, m.start()) + 1, name, ",".join(a.strip() for a in args)))
                pos = close
            else:
                out.append(",".join(args))
                pos = close
        if changed:
            new = "".join(out)
            # sanity: only whitespace/width differences
            assert len(new) == len(src) + 4 * len(changed), (path, len(new) - len(src), len(changed))
            open(path, "w", encoding = "utf8", errors = "surrogateescape").write(new)
            for ln, name, args in changed:
                print(f"{os.path.relpath(path, ROOT)}:{ln}: {name}({args})")
            total += len(changed)
print("total sites patched:", total)
