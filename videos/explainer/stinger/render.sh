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

# The video's ending: both segments joined, the fire's ring tuned to the bed's E♭ (LTX made it in E here: one semitone
# down), then the logo held for HOLD seconds and faded out.
if [ "$W" = 672 ]; then
    HOLD=${HOLD:-3}
    ffmpeg -v error -y -i "$here/out/a-${W}x${H}.mp4" -i "$here/out/b-${W}x${H}.mp4" -filter_complex \
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a];[v]tpad=stop_mode=clone:stop_duration=$HOLD[vh];[a]rubberband=pitch=0.943874:pitchq=quality,apad=pad_dur=$HOLD[ah]" \
        -map "[vh]" -map "[ah]" -c:v libx264 -crf 16 -c:a aac -b:a 256k "$here/out/stinger-hold-tmp.mp4"
    len=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$here/out/stinger-hold-tmp.mp4")
    ffmpeg -v error -y -i "$here/out/stinger-hold-tmp.mp4" \
        -vf "fade=t=out:st=$(echo "$len - 1.2" | bc):d=1.2" -af "afade=t=out:st=$(echo "$len - 2" | bc):d=2" \
        -c:v libx264 -crf 16 -c:a aac -b:a 256k "$here/out/stinger-1344x768-hold.mp4"
    rm "$here/out/stinger-hold-tmp.mp4"
fi
