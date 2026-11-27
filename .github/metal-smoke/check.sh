#!/usr/bin/env bash
# The Metal smoke check, shared by metal-smoke.yml (the released natives) and release.yml (the natives a release is
# about to upload): chats a few tokens from Qwen3 0.6B through LlamaModel and fails unless llama.cpp put the model on a
# Metal device with every layer offloaded.
#
#   .github/metal-smoke/check.sh <classpath> [native library dir]     (Qwen3 0.6B must be in the Hugging Face cache)
set -euo pipefail
cp=$1; libs=${2:-}
model=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/*/Qwen3-0.6B-Q8_0.gguf | head -1)
# FORNAX_VERBOSE: llama.cpp's full log, which says where the layers went (Fornax shows only errors otherwise)
FORNAX_VERBOSE=1 java --enable-native-access=ALL-UNNAMED ${libs:+"-Dfornax.library.path=$libs"} -cp "$cp" \
    "$(dirname "$0")/MetalSmoke.java" "$model" 2> llama.log \
    || { echo "::error::MetalSmoke failed"; tail -60 llama.log; exit 1; }
echo "--- llama.cpp:"; grep -E 'using device|GPU name|has unified memory|recommendedMaxWorkingSetSize|offloaded' llama.log | head -12 || true
offloaded=$(sed -n 's/.*offloaded \([0-9]*\)\/\([0-9]*\) layers to GPU.*/\1 \2/p' llama.log | head -1)
[ -n "$offloaded" ] || { echo "::error::llama.cpp reported no layers offloaded"; tail -40 llama.log; exit 1; }
set -- $offloaded
[ "$1" = "$2" ] && [ "$1" -gt 0 ] || { echo "::error::only $1 of $2 layers on the GPU"; exit 1; }
# The offload count is arithmetic once any GPU backend is there; this line says the model's device is Metal.
grep -qE 'using device MTL[0-9]' llama.log || { echo "::error::llama.cpp didn't put the model on a Metal device"; exit 1; }
echo "Metal: $1/$2 layers offloaded"
