#!/usr/bin/env bash
# The logo stinger: two first/last-frame LTX segments, k1 → k2 (the fire catches) and k2 → k3 (the arch becomes the logo).
#   stinger/render.sh [draft width] [draft height]   (default 352×192, a quick look; 672×384 gives the 1344×768 cut)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); kiln=$(cd "$here/../../.." && pwd)
W=${1:-352} H=${2:-192}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/projects/studio/.cache/hf}
seg() {  # seg <name> <first> <last> <prompt>
    "$kiln/gradlew" -q -p "$kiln" video -Pkiln.backend=${KILN_BACKEND:-hip} --args="$here/out/$1-${W}x${H}.mp4 \"$4\" $here/$2 $here/$3" \
        -Pltx.dir=$HOME/projects/studio/models -Precipe=two-stage -Paudio=true -Pwidth=$W -Pheight=$H -Pframes=57 -Pfps=24 -Pseed=${SEED:-1}
}
mkdir -p "$here/out"
seg a k1.png k2.png "A dark stone kiln arch at night. The glowing embers on its floor catch, and a single flame rises and fills the arch, curling into a bright flame shape. Sparks drift up. The camera is still. Audio: a deep whoomph as the fire ignites, then a roaring crackle."
seg b k2.png k3.png "The burning stone arch transforms: the stones melt into a smooth polished metal frame around the flame, the arch rises and floats, and the word kiln4j forms in glowing letters below it, reflected on a dark floor. Sparks settle. Audio: a rising shimmer that resolves into a warm metallic ring."
