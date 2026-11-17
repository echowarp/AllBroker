#!/usr/bin/env python3
"""Port the lend manager, the lazy vision module and their CPU tests from the internal tree into llmt_kv.

    tools/lend-port/port.py --src <tree> --commit <sha> --lend-dir <dir> --lazyvis-dir <dir> [--accept]

Reads, at <commit> of the internal git tree (git show, so uncommitted edits there are never picked up):
    <lend dir>/llmt_lora.py                   -> crates/py/python/llmt_kv/lend.py
    <lazyvis dir>/llmt_lazyvis.py             -> crates/py/python/llmt_kv/lazyvis.py
    <lend dir>/tests/test_*_cpu.py            -> crates/py/tests/lend/
Edits, in order (each one must apply, or the port stops):
  1. functions named in hooked.py / hooked_lazyvis.py replace the source's functions of the same name (the
     scheduler / block-pool / model-runner / worker wrappers become llmt_kv.vllm_hooks registrations). The source text of
     every replaced function is checked against MANIFEST.json: if the source changed it, the port stops and shows the
     diff, so the hooked version can be updated first (then --accept records the new hashes).
  2. the module docstring is replaced by lend_header.txt / lazyvis_header.txt.
  3. exact-text edits (EDITS below; each must match exactly as often as stated).
  4. comment scrub (SCRUB regexes: dates, quotes, lab provenance), then a gate (GATE) that fails on anything left.
Writes SYNC.json (source commit, md5 of every input and output)."""
import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(ROOT, "crates/py/python/llmt_kv")

LAZY_NAMES = '("exl3xpu.lazyvis", "llmt_lazyvis")'
LAZY_NAMES_NEW = '("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis")'

# (old, new, expected count)
EDITS = {
    "lend": [
        (LAZY_NAMES, LAZY_NAMES_NEW, 4),
    ],
    "lazyvis": [],
    "test": [
        ("import torch, llmt_lora as L", "import torch\nfrom llmt_kv import lend as L", None),
        ("import numpy as np, torch, llmt_lora as L", "import numpy as np, torch\nfrom llmt_kv import lend as L", None),
        ("import llmt_lora as L", "from llmt_kv import lend as L", None),
        # the wrappers are llmt_kv.vllm_hooks registrations now (native hooks, or fallback wrappers on stock vLLM)
        ('assert Scheduler.schedule.__module__ == "llmt_lora" and BlockPool.get_new_blocks.__module__ == "llmt_lora"',
         'from llmt_kv import vllm_hooks as H\n'
         'assert H.native() is not None or (hasattr(Scheduler.schedule, "__llmt_hook_orig__")\n'
         '                                  and hasattr(BlockPool.get_new_blocks, "__llmt_hook_orig__"))\n'
         'if H.native() is not None:   # the stub above replaced the hooked method: apply the fold as vLLM would\n'
         '    _stub = gw.Worker.determine_available_memory\n'
         '    gw.Worker.determine_available_memory = lambda self, *a, **k: H.native().fold(\n'
         '        "worker.available_kv_memory", _stub(self, *a, **k), self)', None),
    ],
}

_DATE = r"(?:20\d\d-\d\d-\d\d|\b1[01]-[0-3]\d(?:\.\.[0-3]?\d)?\b)(?:\s+\d\d:\d[\dx](?::\d\d)?)?"
_CTX = r"(?:(?:Arc|5090|RTX 5090|combined deploy|router trace|Router trace)\s+)?(?:lab\s+)?" \
       r"(?:(?:AP1|C2|C3|S1g|S2|lp3|step-5)\s+(?:recheck\s+)?)?"
SCRUB = [
    # quoted remarks of the person who asked for a feature, with or without a date / "via ..."
    (r"\s*\(\s*(?:" + _DATE + r",?\s*)?(?:user|User)(?: via \w+)?(?: " + _DATE + r")?[^\"()]*?:?\s*\"[^\"]*\"\s*\)", ""),
    (r"(?:" + _DATE + r",?\s*)?(?:[Uu]ser)(?: via \w+)?(?:,? " + _DATE + r")?:\s*\"[^\"]*\"\.?\s*", ""),
    (r"\(\s*(?:[Uu]ser),?\s*" + _DATE + r"[^)]*\)", ""),
    (r"\b(?:user|User),?\s+" + _DATE + r"[,:]?\s*", ""),
    # dated lab / run references
    (r"\s*\(\s*" + _CTX + _DATE + r"\s*\)", ""),
    (r"\(\s*" + _CTX + _DATE + r"[:,;]\s*", "("),
    (_CTX + _DATE + r"[:,;]?\s*", ""),
    (r"\s*\(lab\s*\)", ""),
    # tidy up what the removals leave behind
    (r"\(\s*[,;:]\s*\)", ""),
    (r"#\s*[,;:]\s+", "# "),
    (r",\s*\)", ")"),
    (r"(?<=\w) +:", ":"),
    (r" \(\s*\)", ""),
    (r"# -- ", "# "),
]
GATE_BASE = (r"20\d\d-\d\d-\d\d|\b1[01]-1\d\b|\b[Uu]ser\b|another agent|via main|/srv/|/mnt/|/home/|"
             r"patches/[a-z]|AP1|session [0-9a-f]{6}")
GATE = re.compile(GATE_BASE)
# lines the gate may keep (generic code, not provenance)
GATE_OK = []


def git_show(src, commit, path):
    return subprocess.run(["git", "-C", src, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


def md5(b):
    return hashlib.md5(b if isinstance(b, bytes) else b.encode()).hexdigest()


def funcs(text):
    """name -> (start, end) offsets of top-level defs (decorators included)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    off = [0]
    for ln in lines:
        off.append(off[-1] + len(ln))
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or (
                isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            name = n.name if hasattr(n, "name") else n.targets[0].id
            first = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            out[name] = (off[first - 1], off[n.end_lineno])
    return out


def replace_funcs(text, hooked_text, manifest, key, accept):
    src = funcs(text)
    new = funcs(hooked_text)
    errs = []
    appended = []
    for name, (a, b) in sorted(new.items(), key=lambda kv: -src.get(kv[0], (-1, -1))[0]):
        repl = hooked_text[a:b]
        if name not in src:
            appended.append(repl)
            continue
        sa, sb = src[name]
        old = text[sa:sb]
        h = md5(old)
        want = manifest.setdefault(key, {}).get(name)
        if want != h:
            if not accept:
                errs.append(f"{key}:{name} changed in the source (md5 {h}, manifest {want}); bring hooked*.py in line "
                            f"with it, then --accept. Source vs hooked version:\n" + "".join(difflib.unified_diff(
                                old.splitlines(True), repl.splitlines(True), "source", "hooked")))
                continue
            manifest[key][name] = h
        text = text[:sa] + repl + text[sb:]
    if errs:
        sys.exit("\n".join(errs))
    if appended:
        at = text.index("\ndef register(") if "\ndef register(" in text else len(text)
        text = text[:at] + "\n\n" + "\n\n".join(s.rstrip("\n") + "\n" for s in appended) + text[at:]
    return text


def set_docstring(text, header):
    tree = ast.parse(text)
    d = tree.body[0]
    assert isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant), "no module docstring"
    lines = text.splitlines(keepends=True)
    return header.rstrip("\n") + "\n" + "".join(lines[d.end_lineno:])


def edits(text, kind, name):
    for old, new, n in EDITS[kind]:
        c = text.count(old)
        if n is not None and c != n:
            sys.exit(f"{name}: edit {old!r} matched {c} times, expected {n}")
        text = text.replace(old, new)
    return text


def _scrub_text(t):
    for pat, rep in SCRUB:
        t = re.sub(pat, rep, t)
    return t


def _py_spans(text):
    """(start, end) offsets of comments and string literals (tokenize), so the scrub never touches code."""
    import io
    import tokenize
    lines = text.splitlines(keepends=True)
    off = [0]
    for ln in lines:
        off.append(off[-1] + len(ln))
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            yield off[tok.start[0] - 1] + tok.start[1], off[tok.end[0] - 1] + tok.end[1]


def _c_spans(text):
    for m in re.finditer(r"/\*.*?\*/|//[^\n]*", text, flags=re.S):
        yield m.start(), m.end()


def scrub(text, name):
    spans = list(_py_spans(text) if name.endswith(".py") else _c_spans(text))
    for a, b in reversed(spans):
        text = text[:a] + _scrub_text(text[a:b]) + text[b:]
    bad = [(i + 1, ln) for i, ln in enumerate(text.splitlines()) if GATE.search(ln) and ln.strip() not in GATE_OK]
    return text, [f"{name}:{i}: {ln.strip()}" for i, ln in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="internal git tree")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--lend-dir", required=True, help="dir (in the tree) with llmt_lora.py, tests/")
    ap.add_argument("--lazyvis-dir", required=True, help="dir (in the tree) with llmt_lazyvis.py")
    ap.add_argument("--accept", action="store_true", help="record changed source functions in MANIFEST.json")
    ap.add_argument("--private-words", default="",
                    help="regex of private names (hosts, internal project names): removed in parentheses, else gated")
    ap.add_argument("--fixups", default="",
                    help="exact-text comment rewrites applied before the scrub: {file: [[old, new], ...]}")
    a = ap.parse_args()
    global GATE
    if a.private_words:
        pw = (r"\s*\((?:" + a.private_words + r")[^()]*\)", "")
        SCRUB.insert(0, pw)
        SCRUB.insert(len(SCRUB) - 4, pw)          # again after the date removals (before the tidy-up rules)
        GATE = re.compile(GATE_BASE + "|" + a.private_words, re.I)
    commit = subprocess.run(["git", "-C", a.src, "rev-parse", a.commit], check=True, capture_output=True,
                            text=True).stdout.strip()
    mpath = os.path.join(HERE, "MANIFEST.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    fix = json.load(open(a.fixups)) if os.path.exists(a.fixups) else {}
    sync = {"source_commit": commit, "inputs": {}, "outputs": {}}
    out = {}
    gate = []

    def apply_fix(name, text):
        for old, new in fix.get(name, []):
            if text.count(old) != 1:
                gate.append(f"{name}: fixup does not match once: {old[:70]!r}")
                continue
            text = text.replace(old, new)
        return text

    LEND_DIR, LAZY_DIR = a.lend_dir.strip("/"), a.lazyvis_dir.strip("/")
    for kind, sdir, fname, dst, hooked, header in (
            ("lend", LEND_DIR, "llmt_lora.py", "lend.py", "hooked.py", "lend_header.txt"),
            ("lazyvis", LAZY_DIR, "llmt_lazyvis.py", "lazyvis.py", "hooked_lazyvis.py", "lazyvis_header.txt")):
        raw = git_show(a.src, commit, f"{sdir}/{fname}")
        sync["inputs"][fname] = md5(raw)
        t = raw.decode()
        t = replace_funcs(t, open(os.path.join(HERE, hooked)).read(), manifest, kind, a.accept)
        t = set_docstring(t, open(os.path.join(HERE, header)).read())
        t = edits(t, kind, fname)
        t = apply_fix(dst, t)
        t, bad = scrub(t, dst)
        gate += bad
        ast.parse(t)
        out[os.path.join(PKG, dst)] = t

    names = subprocess.run(["git", "-C", a.src, "ls-tree", "--name-only", commit, f"{LEND_DIR}/tests/"],
                           check=True, capture_output=True, text=True).stdout.split()
    for p in sorted(names):
        b = os.path.basename(p)
        if not re.fullmatch(r"test_\w+_cpu\.py", b):
            continue           # test_whxkey.py compares internal deploy copies: not portable
        raw = git_show(a.src, commit, p)
        sync["inputs"]["tests/" + b] = md5(raw)
        t = edits(raw.decode(), "test", b).replace("llmt_lora.py", "llmt_kv/lend.py")
        t = apply_fix(b, t)
        t, bad = scrub(t, b)
        gate += bad
        out[os.path.join(ROOT, "crates/py/tests/lend", b)] = t

    if gate:
        sys.exit("scrub gate: provenance left (add a fixup in fixups.json):\n" + "\n".join(gate))
    for p, t in out.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(t)
        sync["outputs"][os.path.relpath(p, ROOT)] = md5(t)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(HERE, "SYNC.json"), "w") as f:
        json.dump(sync, f, indent=1, sort_keys=True)
    print(f"ported {len(out)} files from {commit[:12]}")


if __name__ == "__main__":
    main()
