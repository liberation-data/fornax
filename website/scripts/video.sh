#!/usr/bin/env bash
# The explainer, for the page: from the video build's mp4 (arg 1, default the Kiln repo's out/explainer.mp4), writes
# public/video/:
#   explainer.mp4          the web copy: H.264 CRF 24, AAC 160k, moov first (faststart), the fallback where HLS can't play
#   hls/                   the web copy cut into 4 s fmp4 segments (index.m3u8, init.mp4, 000.m4s...), copied, not
#                          re-encoded, with the same settings as worlds-landing's scripts/pull-videos.mjs
#   explainer-poster.jpg   the title card the video opens on (0 to 4 s), at 2 s
#   explainer-end.jpg      the kiln4j logo the video holds at its end, 1.2 s before the end (after it, it fades to black)
#   explainer.vtt          the build's captions, where they are beside the mp4
# public/video/ is gitignored: the video isn't cleared to publish yet, so none of this is committed. Run from website/.
set -euo pipefail

SRC=${1:-$HOME/projects/kiln/videos/explainer/out/explainer.mp4}
OUT=public/video
[[ -f $SRC ]] || { echo "no video at $SRC" >&2; exit 1; }
SRC=$(realpath "$SRC")
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
BUILT=$(dirname "$SRC")

echo "web copy"
ffmpeg -v error -y -i "$SRC" -c:v libx264 -crf 24 -preset slow -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart "$OUT/explainer.mp4.part.mp4"
mv "$OUT/explainer.mp4.part.mp4" "$OUT/explainer.mp4"

echo "HLS segments"
rm -rf "$OUT/hls.part"
mkdir -p "$OUT/hls.part"
ffmpeg -v error -i "$OUT/explainer.mp4" -c copy -f hls -hls_time 4 -hls_playlist_type vod -hls_segment_type fmp4 \
    -hls_fmp4_init_filename init.mp4 -hls_segment_filename "$OUT/hls.part/%03d.m4s" "$OUT/hls.part/index.m3u8"
rm -rf "$OUT/hls"
mv "$OUT/hls.part" "$OUT/hls"

echo "poster and end frame"
ffmpeg -v error -y -ss 2 -i "$SRC" -frames:v 1 -q:v 2 "$OUT/explainer-poster.jpg"
ffmpeg -v error -y -sseof -1.2 -i "$SRC" -frames:v 1 -q:v 3 "$OUT/explainer-end.jpg"

if [[ -f $BUILT/explainer.vtt ]]; then
    echo "captions"
    cp "$BUILT/explainer.vtt" "$OUT/explainer.vtt"
fi

seconds=$(printf '%.0f' "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/explainer.mp4")")
printf 'done: %s, %d:%02d long (the page reads it from the playlist)\n' \
    "$OUT" "$(( seconds / 60 ))" "$(( seconds % 60 ))"
