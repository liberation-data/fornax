#!/usr/bin/env bash
# Converts Qwen/Qwen3-TTS-12Hz-1.7B-Base to GGUF (backbone + mmproj) with the convert_hf_to_gguf.py of the llama.cpp
# Kiln pins, so that the mmproj carries the reference encoder (the speech tokenizer's Mimi encoder, a.gen.enc.*).
# ggml-org's published mmproj lacks it, and without it llama.cpp clones from the speaker embedding only and ignores
# the reference transcript. Load the output by path:
#   ./gradlew speak --args='out.wav "text" @clip.wav' -Pmodel=<out>/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
#       -Pmmproj=<out>/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
# (or new TtsModel(model, mmproj)), with clip.wav.txt beside the clip holding what it says.
#
#   scripts/convert-qwen3-tts.sh [out-dir]          (default: ~/models/qwen3-tts-clone)
#
# Env: PYTHON  a python with torch, safetensors, transformers, numpy and sentencepiece (gguf comes from llama.cpp's gguf-py)
#      LLAMA_CPP  llama.cpp checkout at the pinned commit (default: engines-src/llama.cpp, as native/build.sh fetches)
#      HF_HUB_CACHE, HF_HOME  where the Hugging Face cache lives (as hf does: default ~/.cache/huggingface/hub)
#      OUTTYPE  backbone and mmproj type (default q8_0; the converter keeps the encoder's convs, codebooks and
#               transformer in F32/F16 regardless)
# The weights must be in the HF cache first: hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
set -euo pipefail

case "${1-}" in -*) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; exit 2 ;; esac
[ $# -le 1 ] || { echo "usage: $0 [out-dir]" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$HOME/models/qwen3-tts-clone}
PYTHON=${PYTHON:-python3}
LLAMA_CPP=${LLAMA_CPP:-$ROOT/engines-src/llama.cpp}
OUTTYPE=${OUTTYPE:-q8_0}
TYPE=$(echo "$OUTTYPE" | tr a-z A-Z)  # not ${OUTTYPE^^}: macOS has bash 3.2
REPO=Qwen/Qwen3-TTS-12Hz-1.7B-Base
NAME=Qwen3-TTS-12Hz-1.7B-Base

pin=$(grep '^llama.cpp.commit=' "$ROOT/native/engines.properties" | cut -d= -f2)
[ -f "$LLAMA_CPP/convert_hf_to_gguf.py" ] || { echo "no llama.cpp at $LLAMA_CPP (run native/build.sh vulkan tts, or set LLAMA_CPP)" >&2; exit 1; }
head=$(git -C "$LLAMA_CPP" rev-parse HEAD 2>/dev/null || echo "an unknown commit")
[ "$head" = "$pin" ] || echo "warning: $LLAMA_CPP is at $head, Kiln pins $pin" >&2

# The snapshot hf download last fetched (refs/main), not whichever older one sorts first.
cache=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--${REPO//\//--}
snap=$cache/snapshots/$(cat "$cache/refs/main" 2>/dev/null || true)
[ -f "$snap/speech_tokenizer/model.safetensors" ] || { echo "not in the HF cache: hf download $REPO" >&2; exit 1; }

mkdir -p "$OUT"
# convert_hf_to_gguf.py puts its own gguf-py first on sys.path, so the converter and gguf always match.
"$PYTHON" "$LLAMA_CPP/convert_hf_to_gguf.py" "$snap" --outtype "$OUTTYPE" --outfile "$OUT/$NAME-$TYPE.gguf"
"$PYTHON" "$LLAMA_CPP/convert_hf_to_gguf.py" "$snap" --outtype "$OUTTYPE" --mmproj --outfile "$OUT/mmproj-$NAME-$TYPE.gguf"

# The reference encoder's last tensor: present means voice cloning with a transcript (ICL) works.
PYTHONPATH="$LLAMA_CPP/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$OUT/mmproj-$NAME-$TYPE.gguf" <<'EOF'
import sys, gguf
names = {t.name for t in gguf.GGUFReader(sys.argv[1]).tensors}
enc = sorted(n for n in names if n.startswith(("a.gen.enc.", "a.seanet.", "a.downsample.")))
print(f"reference encoder: {len(enc)} tensors")
sys.exit(0 if "a.gen.enc.quant.rest.codebook.weight" in names else "a.gen.enc.quant.rest.codebook missing: no reference encoder")
EOF
ls -l "$OUT"
