#!/bin/bash
# keeper-fdstore-test.sh: verifies that weights held by llmt-keeper survive keeper restarts through systemd's fd store,
# with real GPU memory. It runs the keeper as a TRANSIENT unit (systemd-run, gone after the test), never installs one.
# Needs: the llmt-keeper binary (KEEPER=), an engine-side client that can export / import device memory: by default
# weights/tests/gpu_compat.py in a container with the GPU (CLIENT="docker run ... python3 .../gpu_compat.py"), and no
# other user of the GPU memory you measure. Starting a unit on a production host needs the operator's OK: the script
# refuses unless KEEPER_FDSTORE_TEST_OK=1.
#   KEEPER=/path/llmt-keeper CLIENT="<cmd taking: socket phase>" VRAM="<cmd printing used MiB>" [VRAM_SETTLE=s] keeper-fdstore-test.sh
# The keeper's socket directory ($D below) must be visible to the client (bind-mount it into a client container).
# Steps (each must pass; the script stops at the first failure and cleans up):
#   1 start the keeper unit (Type=notify, FileDescriptorStoreMax=64, FileDescriptorStorePreserve=yes)
#   2 client put: 3 x 64 MiB of patterned device memory, the exporting process exits    -> VRAM stays up
#   3 systemctl restart (clean stop + start)                                          -> entry restored, client check
#   4 SIGKILL the keeper (Restart=always brings it back)                               -> entry restored, client check
#   5 systemctl stop, wait, start (fds parked in the store while the unit is down)     -> entry restored, client check
#   6 client drop                                                                      -> fd store empty, VRAM back down
#   7 stop the unit, clean its fd store
set -u
[ "${KEEPER_FDSTORE_TEST_OK:-0}" = 1 ] || { echo "refusing: set KEEPER_FDSTORE_TEST_OK=1 (starts a transient systemd unit)"; exit 2; }
: "${KEEPER:?llmt-keeper binary}" "${CLIENT:?client command}"
VRAM=${VRAM:-"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0"}; W=${VRAM_SETTLE:-2}   # s before a VRAM reading (Arc telemetry: 5)
U=llmt-keeper-fdtest; D=/run/llmt-whx-fdtest; ST=/var/lib/llmt-whx-fdtest; SOCK=$D/keeper.sock
fail() { echo "FAIL: $*"; cleanup; exit 1; }
cleanup() { systemctl stop $U 2>/dev/null; systemctl clean --what=fdstore $U 2>/dev/null; systemctl reset-failed $U 2>/dev/null; rm -rf $D $ST; }
vram() { eval "$VRAM" | head -1 | tr -d ' '; }
restored() { journalctl -u $U --since "@$1" --no-pager -o cat | grep -oE "fd store on, [0-9]+ entries restored" | tail -1; }
up() { for i in $(seq 50); do [ -S $SOCK ] && systemctl is-active -q $U && return 0; sleep 0.2; done; return 1; }
mkdir -p $D $ST; v0=$(vram); echo "VRAM before: $v0 MiB"
systemd-run --unit=$U -p Type=notify -p NotifyAccess=main -p FileDescriptorStoreMax=64 \
  -p FileDescriptorStorePreserve=yes -p Restart=always -p RestartSec=1 \
  "$KEEPER" --socket $SOCK --fdstore $ST/keeper-state.json || fail "systemd-run"
up || fail "keeper did not come up"
$CLIENT $SOCK put || fail "put"
sleep $W; v1=$(vram); echo "VRAM held after the exporter exited: $v1 MiB (+$((v1 - v0)))"; [ $((v1 - v0)) -ge 150 ] || fail "memory not held"
t=$(date +%s); systemctl restart $U; up || fail "restart"; sleep 0.5
echo "restart: $(restored $t)"; $CLIENT $SOCK check || fail "check after restart"
t=$(date +%s); systemctl kill -s KILL $U; sleep 2; up || fail "no auto-restart after SIGKILL"
echo "SIGKILL: $(restored $t)"; $CLIENT $SOCK check || fail "check after SIGKILL"
t=$(date +%s); systemctl stop $U; sleep $((W + 2)); v2=$(vram); echo "unit stopped: VRAM $v2 MiB (fd store keeps it)"
[ $((v2 - v0)) -ge 150 ] || fail "memory freed while the unit was stopped (FileDescriptorStorePreserve)"
systemctl start $U; up || fail "start after stop"; sleep 0.5
echo "stop/start: $(restored $t)"; $CLIENT $SOCK check || fail "check after stop/start"
$CLIENT $SOCK drop || fail "drop"
sleep $W; v3=$(vram); echo "after drop: VRAM $v3 MiB"; [ $((v3 - v0)) -lt 100 ] || fail "memory not freed after drop"
cleanup; echo "keeper fd-store test OK"
