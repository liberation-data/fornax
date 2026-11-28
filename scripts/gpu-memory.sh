#!/usr/bin/env bash
# Runs a command and prints the most GPU memory it took: the peak of the amdgpu driver's VRAM plus GTT use, sampled
# every 0.2 s, above where it stood when the command started. AMD GPUs on Linux only; on a unified-memory APU (Strix
# Halo) GTT is where models live. Other GPU work running at the same time counts too, so run it alone.
#
#   scripts/gpu-memory.sh <command...>
set -euo pipefail
gtt=(/sys/class/drm/card*/device/mem_info_gtt_used)
[[ -e ${gtt[0]} ]] || { echo "gpu-memory: no amdgpu device" >&2; exit 1; }
dev=${gtt[0]%/*}
used() { echo $(( ($(cat "$dev/mem_info_gtt_used") + $(cat "$dev/mem_info_vram_used")) / 1048576 )); }
base=$(used); peak=$base
"$@" & pid=$!
while kill -0 "$pid" 2>/dev/null; do u=$(used); (( u > peak )) && peak=$u; sleep 0.2; done
rc=0; wait "$pid" || rc=$?
echo "gpu-memory: peak $((peak - base)) MB above the start" >&2
exit $rc
