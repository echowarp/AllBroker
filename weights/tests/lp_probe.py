#!/usr/bin/env python3
"""Teacher-forced logprob probe (keeper identity check), synthetic prompts only (no corpus / customer data).
  lp_probe.py dump <out.json> [url]     prompt logprobs of a ~1.5k-token synthetic text + top-20 first-token logprobs of 8 prompts
  lp_probe.py cmp <a.json> <b.json>     max / mean |diff| and exact-equal share
Prefill only (max_tokens=1), so speculative decoding is not involved."""
import json, sys, urllib.request

TOPICS = ["page caches", "B-trees", "TCP congestion control", "garbage collection", "RAID levels", "SSD wear levelling",
          "consistent hashing", "the Linux scheduler"]
LONG = " ".join(f"Section {i}: {t} matter because systems trade latency for throughput; when {t} misbehave, operators "
                f"measure queue depths, tail latencies and error rates, then change one setting at a time and record "
                f"the result next to the baseline." for i, t in enumerate(TOPICS * 4))


def post(url, body):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def dump(out, url):
    res = {"first": []}
    for t in TOPICS:
        d = post(url, {"model": "qwen3.8-27b", "prompt": f"Explain {t} to a new engineer in three sentences.",
                       "max_tokens": 1, "temperature": 0, "logprobs": 20})
        res["first"].append(d["choices"][0]["logprobs"]["top_logprobs"][0])
    try:
        d = post(url, {"model": "qwen3.8-27b", "prompt": LONG, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
        res["prompt_raw"] = d["choices"][0].get("prompt_logprobs") or []
    except Exception as e:  # noqa
        res["prompt_error"] = repr(e)
    json.dump(res, open(out, "w"))
    print(f"lp dump: {len(res['first'])} first-token rows, prompt logprobs: "
          f"{len(res['prompt_raw']) if res.get('prompt_raw') else res.get('prompt_error')}")


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    first, prompt, top1 = [], [], [0, 0]
    for x, y in zip(A["first"], B["first"]):
        first += [abs(x[k] - y[k]) for k in set(x) & set(y)]
        top1[0] += 1; top1[1] += max(x, key=x.get) == max(y, key=y.get)
    for x, y in zip(A.get("prompt_raw") or [], B.get("prompt_raw") or []):
        if x and y:
            prompt += [abs(x[k]["logprob"] - y[k]["logprob"]) for k in set(x) & set(y) if x[k].get("rank") == 1 or len(x) == 1]
            if len(x) > 1:          # the prompt token itself (not the top-1 alternative)
                k = [k for k in x if x[k].get("rank") != 1]
                prompt += [abs(x[k[0]]["logprob"] - y[k[0]]["logprob"])] if k and k[0] in y else []
    def st(d):
        d = sorted(d); n = len(d)
        return (f"n {n}, exact {100 * sum(1 for v in d if v == 0) / max(n, 1):.1f}%, mean {sum(d) / max(n, 1):.5f}, "
                f"median {d[n // 2] if n else 0:.5f}, p99 {d[int(n * .99)] if n else 0:.4f}, max {d[-1] if n else 0:.4f}")
    print(f"first-token top-20: {st(first)}; argmax same {top1[1]}/{top1[0]}")
    print(f"prompt tokens (teacher-forced): {st(prompt)}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8001")
    else:
        cmp(sys.argv[2], sys.argv[3])
