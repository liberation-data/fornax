#!/usr/bin/env bash
# Writes fornax-llama/src/test/resources/embeddings/<model>.txt, the reference EmbeddingsIT (qwen3-embedding-0.6b) and
# EmbeddingGemmaIT (embeddinggemma-2) compare Fornax's embeddings with: llama.cpp's own llama-embedding, built from the
# pinned llama.cpp (native/engines.properties), on the same model and texts. One line per text: the text (with \n for
# a newline), a tab, then the normalised embedding.
#
#   scripts/embedding-reference.sh [path/to/llama-embedding] [qwen3-embedding-0.6b|embeddinggemma-2]
#
# Build llama-embedding with the same backend as the shim, for example (Vulkan):
#   cmake -S engines-src/llama.cpp -B engines-src/llama.cpp/build-ref-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=ON $CMAKE_EXTRA
#   cmake --build engines-src/llama.cpp/build-ref-vulkan --target llama-embedding
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-$ROOT/engines-src/llama.cpp/build-ref-vulkan/bin/llama-embedding}
NAME=${2:-qwen3-embedding-0.6b}
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
OUT=$ROOT/fornax-llama/src/test/resources/embeddings/$NAME.txt
SEP='<#kiln#>'

# The texts each IT embeds, prefixes written out; keep them in step.
case $NAME in
    qwen3-embedding-0.6b)
        MODEL=$(ls "$HUB"/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/*/Qwen3-Embedding-0.6B-Q8_0.gguf | head -1)
        TEXTS=(
            "The kiln fires clay into stoneware at about 1200 degrees Celsius."
            "Nashville is the capital of Tennessee and the home of country music."
            $'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:How hot does a kiln get?'
        ) ;;
    embeddinggemma-2)
        MODEL=$(ls "$HUB"/models--unsloth--embeddinggemma-2-GGUF/snapshots/*/embeddinggemma-2-Q8_0.gguf | head -1)
        TEXTS=(
            "title: none | text: The kiln fires clay into stoneware at about 1200 degrees Celsius."
            "title: none | text: Nashville is the capital of Tennessee and the home of country music."
            "task: search result | query: How hot does a kiln get?"
        ) ;;
    *) echo "unknown model: $NAME" >&2; exit 2 ;;
esac
prompt=$(printf "%s$SEP" "${TEXTS[@]}"); prompt=${prompt%"$SEP"}

# -c 8192 -ub 8192: the same context and batch as LlamaModel's EMBED default
"$BIN" -m "$MODEL" -ngl 99 -c 8192 -b 8192 -ub 8192 --embd-normalize 2 --embd-output-format array \
    --embd-separator "$SEP" -p "$prompt" 2>/dev/null > "$OUT.json"
python3 -I - "$OUT.json" "$OUT" "${TEXTS[@]}" <<'PY'
import json, sys
src, out, texts = sys.argv[1], sys.argv[2], sys.argv[3:]
vecs = json.load(open(src))
assert len(vecs) == len(texts), (len(vecs), len(texts))
with open(out, "w") as f:
    for t, v in zip(texts, vecs):
        f.write(t.replace("\n", "\\n") + "\t" + " ".join("%.6f" % x for x in v) + "\n")
PY
rm "$OUT.json"
echo "wrote $OUT"
