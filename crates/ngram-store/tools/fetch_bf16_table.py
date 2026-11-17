#!/usr/bin/env python3
"""Original-precision (bf16) n-gram table of Qwen3.8-Flash-Next -> one raw row file (exact,
unquantized n-gram rows). Downloads only the 33 safetensors files that hold
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight (128 x [2,500,012, 160] bf16),
one file at a time: curl (resumable) -> sha256 against the LFS oid -> each shard pwritten at
shard * 2,500,012 rows into rows.bin (320 B rows, 102.4 GB, row r = global n-gram row r) -> the source file deleted.
A done-list makes it restartable. Refuses to download while / would drop below --min-free-gb.
Run at idle priority:  ionice -c 3 nice -n 19 python3 fetch_bf16_table.py [--out DIR]
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next"
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
SHARDS, SHARD_ROWS, DIM = 128, 2_500_012, 160
ROW = DIM * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/models/qwen3.8-flash-next-ngram-bf16"))
    ap.add_argument("--min-free-gb", type=float, default=100)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "dl"), exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/{REV}/"
    idx_path = os.path.join(a.out, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        urllib.request.urlretrieve(base + "model.safetensors.index.json", idx_path)
    wm = json.load(open(idx_path))["weight_map"]
    files = sorted({f for k, f in wm.items() if k.startswith(PREFIX)})
    req = urllib.request.Request(f"https://huggingface.co/api/models/{REPO}/paths-info/{REV}",
                                 data=json.dumps({"paths": files}).encode(), headers={"Content-Type": "application/json"})
    info = {e["path"]: e for e in json.load(urllib.request.urlopen(req))}
    rows_path = os.path.join(a.out, "rows.bin")
    total = SHARDS * SHARD_ROWS * ROW
    fd = os.open(rows_path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != total:
        os.ftruncate(fd, total)   # sparse until written
    done_path = os.path.join(a.out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    for f in files:
        if f in done:
            continue
        size, oid = info[f]["size"], info[f]["lfs"]["oid"]
        src = os.path.join(a.out, "dl", f)
        have = os.path.getsize(src) if os.path.exists(src) else 0
        free = shutil.disk_usage(a.out).free
        # rows.bin is sparse: what it still has to allocate counts against the floor too
        pending = total - os.fstat(fd).st_blocks * 512
        if free - (size - have) - max(pending, 0) < a.min_free_gb * 1e9:
            sys.exit(f"stop: {free / 1e9:.0f} GB free, {f} needs {(size - have) / 1e9:.1f} GB + rows.bin {pending / 1e9:.0f} GB "
                     f"still to allocate; floor {a.min_free_gb} GB")
        print(f"{f}: {size / 1e9:.2f} GB", flush=True)
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "20", "--retry-all-errors", "-C", "-", "-o", src, base + f], check=True)
        h = hashlib.sha256()
        with open(src, "rb") as s:
            while b := s.read(16 << 20):
                h.update(b)
        if h.hexdigest() != oid:
            sys.exit(f"{f}: sha256 {h.hexdigest()} != LFS oid {oid} (file kept for inspection)")
        with open(src, "rb") as s:
            n = struct.unpack("<Q", s.read(8))[0]
            hdr = json.loads(s.read(n))
            data0 = 8 + n
            for k, t in hdr.items():
                if not k.startswith(PREFIX):
                    continue
                shard = int(k[len(PREFIX):].split(".")[0])
                if t["dtype"] != "BF16" or t["shape"] != [SHARD_ROWS, DIM]:
                    sys.exit(f"{k}: {t['dtype']} {t['shape']}, expected BF16 [{SHARD_ROWS}, {DIM}]")
                b0, b1 = t["data_offsets"]
                assert b1 - b0 == SHARD_ROWS * ROW
                s.seek(data0 + b0)
                at, dst = 0, shard * SHARD_ROWS * ROW
                while at < b1 - b0:
                    buf = s.read(min(64 << 20, b1 - b0 - at))
                    os.pwrite(fd, buf, dst + at)
                    at += len(buf)
                print(f"  shard {shard} -> rows {shard * SHARD_ROWS}..{(shard + 1) * SHARD_ROWS - 1}", flush=True)
        os.fsync(fd)
        os.remove(src)
        with open(done_path, "a") as d:
            d.write(f + "\n")
        done.add(f)
    if len(done) == len(files):
        with open(rows_path + ".complete", "w") as c:   # the engine's PLE_FORMAT=auto (default) looks for it
            c.write(f"{REPO}@{REV}\n")
    print("done" if len(done) == len(files) else f"{len(done)}/{len(files)} files", flush=True)


if __name__ == "__main__":
    main()
