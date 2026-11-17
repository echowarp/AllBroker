#!/bin/bash
# CPU tests of the lend manager, lazy vision, hooks and keeper loader, inside an image that has vLLM + torch
# (no GPU device, no network). Usage: run-cpu.sh <image> [cuda|xpu] [<echowarp/vllm worktree> <base ref>]
#   run-cpu.sh my-vllm-cuda-image cuda                       stock vLLM in the image: fallback wrappers
#   run-cpu.sh my-vllm-cuda-image cuda ~/vllm v0.26.0        the worktree's files changed since <base ref> mounted
#                                                            over the image's vllm package: native extension hooks
# The repo is mounted read-only; llmt_kv comes from crates/py/python (the Rust client is not needed).
set -euo pipefail
img=$1; kind=${2:-cuda}; tree=${3:-}; base=${4:-}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
mounts=()
if [[ -n $tree ]]; then
  site=$(docker run --rm --memory 2g --network none --entrypoint /bin/bash "$img" -c \
    'p=$(test -x /opt/venv/bin/python3 && echo /opt/venv/bin/python3 || command -v python3); $p -c "import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec(\"vllm\").origin)))"')
  while read -r f; do mounts+=(-v "$tree/$f:$site/$f:ro"); done < <(git -C "$tree" diff --name-only "$base" -- vllm/)
fi
exec docker run --rm --memory 2g --network none -e PYTHONPATH=/repo/crates/py/python -e KIND="$kind" \
  -e TRITON_INTERPRET=1 "${mounts[@]}" -v "$repo":/repo:ro -w /tmp --entrypoint /bin/bash "$img" -c '
set -u; py=$(test -x /opt/venv/bin/python3 && echo /opt/venv/bin/python3 || command -v python3); pass=0; fail=0; failed=""
run() { local t=$1; shift; if (cd /tmp && timeout 600 $py "$@" >/tmp/out 2>&1); then pass=$((pass+1)); echo "PASS $t"; else
        fail=$((fail+1)); failed="$failed $t"; echo "FAIL $t"; tail -25 /tmp/out; fi; }
for f in /repo/crates/py/tests/lend/test_*_cpu.py; do
  b=$(basename $f)
  case $b in
    test_whx_lazyvis_cpu.py) run $b $f /repo/crates/py/python/llmt_kv/whx_loader.py ;;
    *) run $b $f ;;
  esac
done
for f in /repo/crates/py/tests/test_vllm_hooks.py /repo/weights/tests/test_whx_loader.py /repo/weights/tests/test_whx_lorapaths.py; do run $(basename $f) $f; done
echo "== $KIND: $pass passed, $fail failed$failed"; [ $fail = 0 ]'
