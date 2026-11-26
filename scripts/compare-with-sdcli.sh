#!/usr/bin/env bash
# Compare the program stable-diffusion.cpp compiles for a Fornax request with the one sd-cli compiles for the
# same settings. A difference here is a Fornax bug (or an option Fornax forgot) found in seconds, without
# rendering: sd.cpp logs the whole compiled DGL program at debug level before it runs anything, and both
# runs are stopped as soon as that program has been printed.
#
#   scripts/compare-with-sdcli.sh <fornax-gradle-task> "<fornax --args>" "<sd-cli args>" [-P... fornax properties]
#
# Example (Endless Country's two-stage LTX recipe at a small size):
#   scripts/compare-with-sdcli.sh video \
#     "out/x.mp4 \"a prompt\" keyframe.png" \
#     "-M vid_gen --diffusion-model ... --hires --hires-upscaler <name> ... -p \"a prompt\" -o /tmp/x.webm" \
#     -Precipe=two-stage -Pwidth=336 -Pheight=224 -Pframes=49
#
# Needs sd-cli built from engines-src/stable-diffusion.cpp with the same patches:
#   cmake -S engines-src/stable-diffusion.cpp -B engines-src/stable-diffusion.cpp/build-cli -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release -DSD_VULKAN=ON -DSD_BUILD_EXAMPLES=ON && cmake --build ... --target sd-cli
set -euo pipefail

TASK=$1 FORNAX_ARGS=$2 CLI_ARGS=$3; shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLI=${SD_CLI:-$ROOT/engines-src/stable-diffusion.cpp/build-cli/bin/sd-cli}
OUT=$(mktemp -d)

# Print the program block from a log, then stop reading.
program() { awk '/Executing DGL program:/{f=1;next} f&&/--- [a-z_]+\.cpp:[0-9]+$/{exit} f' "$1"; }

# Run a command until its log contains the whole program, then stop it.
until_program() {  # until_program <log> <command...>
    local log=$1; shift
    "$@" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 600); do
        if grep -q 'Executing DGL program:' "$log" && program "$log" | grep -q '^fn main'; then break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

echo "fornax: capturing its program..." >&2
until_program "$OUT/fornax.log" env FORNAX_VERBOSE=debug "$ROOT/gradlew" -q -p "$ROOT" "$TASK" --args="$FORNAX_ARGS" "$@"
echo "sd-cli: capturing its program..." >&2
eval "set -- $CLI_ARGS"
until_program "$OUT/cli.log" "$CLI" --log-level debug "$@"

program "$OUT/fornax.log" > "$OUT/fornax.dgl"
program "$OUT/cli.log" > "$OUT/cli.dgl"
for side in fornax cli; do
    [ -s "$OUT/$side.dgl" ] || { echo "no program captured from $side; see $OUT/$side.log" >&2; exit 2; }
done
if diff -u "$OUT/cli.dgl" "$OUT/fornax.dgl" > "$OUT/program.diff"; then
    echo "same program ($OUT)"
else
    echo "programs differ (- sd-cli, + fornax): $OUT/program.diff"
    cat "$OUT/program.diff"
    exit 1
fi
