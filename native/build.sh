#!/usr/bin/env bash
# Builds Fornax's native libraries: one self-contained shared library per engine, with ggml linked in
# statically and its symbols kept private, so several engines can share one JVM even though each
# pins a different ggml.
#
#   native/build.sh [vulkan|hip|cuda|metal|cpu] [engine...]
#
# Engines: llama tts sd acestep yue2 whisper demucs (default: all). Output: build/native/<os>-<arch>-<backend>/.
# Env: JOBS, HIP_ARCH (default gfx1151), CMAKE_EXTRA (extra cmake args for every engine).
set -euo pipefail

BACKEND=${1:-vulkan}; shift || true
ENGINES=${*:-llama tts sd acestep yue2 whisper demucs}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/engines-src
OS=$(uname -s | tr A-Z a-z); ARCH=$(uname -m)
[ "$OS" = darwin ] && OS=macos
OUT=$ROOT/build/native/$OS-$ARCH-$BACKEND
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
mkdir -p "$SRC" "$OUT"

prop() { grep "^$1=" "$ROOT/native/engines.properties" | cut -d= -f2-; }

fetch() {  # fetch <dir-name>
    local dir=$SRC/$1 url commit want
    url=$(prop "$1.url"); commit=$(prop "$1.commit")
    [ -d "$dir/.git" ] || git clone -q "$url" "$dir"
    # When the pin or any patch changed since the last fetch, drop the old patches (tracked files only: the build
    # dirs stay) so the new ones apply to a clean checkout. The submodules too: patches to a vendored ggml land there.
    want="$commit $({ cat "$ROOT/native/patches/$1"/*.patch 2>/dev/null || true; } | git hash-object --stdin)"
    [ "$(cat "$dir/.git/fornax-patches" 2>/dev/null)" = "$want" ] || { git -C "$dir" reset -q --hard; git -C "$dir" submodule foreach -q --recursive git reset -q --hard; }
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
        git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
        git -C "$dir" checkout -q "$commit"
    fi
    git -C "$dir" submodule update -q --init --recursive
    # Fornax's patches to the engine (native/patches/<engine>/*.patch), each applied once.
    for p in "$ROOT/native/patches/$1"/*.patch; do
        [ -e "$p" ] || continue
        if git -C "$dir" apply --reverse --check "$p" 2>/dev/null; then continue; fi
        git -C "$dir" apply "$p" || { echo "patch failed: $p" >&2; exit 1; }
    done
    echo "$want" > "$dir/.git/fornax-patches"
}

# ggml backend switches, plus the system libraries a static ggml backend needs at link time.
case $BACKEND in
    vulkan) GGML="-DGGML_VULKAN=ON"; SD="-DSD_VULKAN=ON"; LIBS="-lvulkan" ;;
    hip)    GGML="-DGGML_HIP=ON -DGPU_TARGETS=${HIP_ARCH:-gfx1151}"; SD="-DSD_HIPBLAS=ON -DGPU_TARGETS=${HIP_ARCH:-gfx1151}"
            # /opt/rocm from AMD's packages; Fedora's own ROCm lives in /usr (libraries in /usr/lib64).
            ROCM_LIB=${ROCM_PATH:-/opt/rocm}/lib; [ -d "$ROCM_LIB" ] || ROCM_LIB=/usr/lib64
            LIBS="-L$ROCM_LIB -lhipblas -lrocblas -lamdhip64" ;;
    cuda)   GGML="-DGGML_CUDA=ON"; SD="-DSD_CUDA=ON"; LIBS="-L${CUDA_PATH:-/usr/local/cuda}/lib64 -lcudart -lcublas -lcublasLt -lcuda" ;;
    metal)  GGML="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"; SD="-DSD_METAL=ON"
            LIBS="-framework Metal -framework Foundation -framework MetalKit -framework Accelerate" ;;
    cpu)    GGML=""; SD=""; LIBS="" ;;
    *) echo "unknown backend: $BACKEND" >&2; exit 1 ;;
esac

COMMON="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF ${CMAKE_EXTRA:-}"
CXX=${CXX:-c++}

if [ "$OS" = macos ]; then
    SHARED="-dynamiclib"; EXT=dylib; PRIVATE="-Wl,-undefined,error"; OMP=""
    whole() { for a in "$@"; do printf -- '-Wl,-force_load,%s ' "$a"; done; }
    group() { printf -- '%s ' "$@"; }  # ld64 resolves archives in any order
else
    SHARED="-shared"; EXT=so; PRIVATE="-Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--no-undefined -Wl,--version-script=$ROOT/native/exports.map"; OMP="-fopenmp"
    whole() { printf -- '-Wl,--whole-archive %s -Wl,--no-whole-archive ' "$*"; }
    group() { printf -- '-Wl,--start-group %s -Wl,--end-group ' "$*"; }
fi

ggml_archives() {  # every static ggml lib under a build dir, as one link group (they reference each other)
    group $(find "$1" -name 'libggml*.a' | sort)
}

cmake_build() {  # cmake_build <src> <build> <targets> <args...>
    local src=$1 bld=$2 targets=$3; shift 3
    cmake -S "$src" -B "$bld" $COMMON "$@" > "$bld.configure.log" 2>&1 || { tail -30 "$bld.configure.log"; exit 1; }
    cmake --build "$bld" -j "$JOBS" --target $targets > "$bld.build.log" 2>&1 || { tail -30 "$bld.build.log"; exit 1; }
}

build_llama() {  # text and vision: links libmtmd (images in, through an mmproj) the way build_tts does
    fetch llama.cpp
    local s=$SRC/llama.cpp b=$SRC/llama.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" "llama llama-common mtmd" $GGML -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_MTMD=ON
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_llama.$EXT" "$ROOT/fornax-llama/native/fornax_llama.cpp" \
        -I"$s/include" -I"$s/ggml/include" -I"$s/common" -I"$s/vendor" -I"$s/tools/mtmd" \
        $(group "$(find "$b" -name libmtmd.a)" "$(find "$b" -name libllama-common.a)" "$(find "$b" -name libllama-common-base.a)" \
                $(find "$b/vendor" -name '*.a' 2>/dev/null) "$(find "$b" -name libllama.a)") $(ggml_archives "$b") \
        $PRIVATE $OMP $LIBS -lpthread
}

build_tts() {  # llama.cpp's text to speech (libmtmd's audio generation), in the same build tree as llama
    fetch llama.cpp
    local s=$SRC/llama.cpp b=$SRC/llama.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" "llama llama-common mtmd" $GGML -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_MTMD=ON
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_tts.$EXT" "$ROOT/fornax-tts/native/fornax_tts.cpp" \
        -I"$s/include" -I"$s/ggml/include" -I"$s/common" -I"$s/vendor" -I"$s/tools/mtmd" \
        $(group "$(find "$b" -name libmtmd.a)" "$(find "$b" -name libllama-common.a)" "$(find "$b" -name libllama-common-base.a)" \
                $(find "$b/vendor" -name '*.a' 2>/dev/null) "$(find "$b" -name libllama.a)") $(ggml_archives "$b") \
        $PRIVATE $OMP $LIBS -lpthread
}

build_sd() {
    fetch stable-diffusion.cpp
    local s=$SRC/stable-diffusion.cpp b=$SRC/stable-diffusion.cpp/build-$BACKEND
    mkdir -p "$b"
    # sd.cpp's own shared library already links ggml statically.
    cmake_build "$s" "$b" stable-diffusion $SD -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_EXAMPLES=OFF
    cp "$(find "$b" -name "libstable-diffusion.$EXT" | head -1)" "$OUT/"
}

build_acestep() {
    fetch acestep.cpp
    local s=$SRC/acestep.cpp b=$SRC/acestep.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" "acestep-core yyjson" $GGML
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_acestep.$EXT" "$ROOT/fornax-acestep/native/fornax_acestep.cpp" \
        -I"$s/src" -I"$s" -I"$s/ggml/include" -I"$s/vendor/yyjson" \
        $(whole "$(find "$b" -name libacestep-core.a)") "$(find "$b" -name libyyjson.a)" $(ggml_archives "$b") \
        $PRIVATE $OMP $LIBS -lpthread
}

build_yue2() {
    fetch yue2.cpp
    local s=$SRC/yue2.cpp b=$SRC/yue2.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" ggml $GGML
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_yue2.$EXT" "$ROOT/fornax-yue2/native/fornax_yue2.cpp" \
        "$s/src/request.cpp" "$s/src/model-store.cpp" "$s/vendor/yyjson/yyjson.c" \
        -I"$s/src" -I"$s" -I"$b" -I"$s/ggml/include" -I"$s/vendor/yyjson" -I"$s/vendor" \
        $(ggml_archives "$b") $PRIVATE $OMP $LIBS -lpthread
}

build_whisper() {
    fetch whisper.cpp
    local s=$SRC/whisper.cpp b=$SRC/whisper.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" whisper $GGML -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_whisper.$EXT" "$ROOT/fornax-whisper/native/fornax_whisper.cpp" \
        -I"$s/include" -I"$s/ggml/include" \
        "$(find "$b" -name libwhisper.a)" $(ggml_archives "$b") $PRIVATE $OMP $LIBS -lpthread
}

build_demucs() {  # stems.cpp's HTDemucs, through its libstems C ABI (src/libstems.cpp), compiled into the shim
    fetch stems.cpp
    local s=$SRC/stems.cpp b=$SRC/stems.cpp/build-$BACKEND
    mkdir -p "$b"
    cmake_build "$s" "$b" stems $GGML -DSTEMS_BUILD_TOOLS=OFF -DBUILD_TESTING=OFF
    $CXX -std=c++17 -O2 -fPIC $SHARED -o "$OUT/libfornax_demucs.$EXT" "$ROOT/fornax-demucs/native/fornax_demucs.cpp" \
        "$s/src/libstems.cpp" -DSTEMS_VERSION_STRING="\"$(sed -n 's/^project(stems_cpp VERSION \([^ ]*\).*/\1/p' "$s/CMakeLists.txt")\"" \
        -I"$s/src" -I"$s/ggml/include" \
        "$(find "$b" -name libstems_core.a)" $(ggml_archives "$b") $PRIVATE $OMP $LIBS -ldl -lpthread
}

for e in $ENGINES; do
    echo "== $e ($BACKEND)"
    "build_$e"
done
echo "built into $OUT:"; ls -1 "$OUT"
