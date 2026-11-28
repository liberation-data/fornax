#!/usr/bin/env bash
# Runs Fornax's tests on a Mac over SSH, against the Metal backend: copies this working tree there (committed or not),
# builds the Metal libraries with native/build.sh, runs ./gradlew test, and brings the test reports back.
#
#   scripts/test-on-mac.sh <host> [gradle args...]     e.g. scripts/test-on-mac.sh my-mac.local --tests '*LlamaModelIT'
#   ./gradlew testOnMac -Pfornax.macHost=<host>           the same, from Gradle (or set fornax.macHost in ~/.gradle/gradle.properties);
#                                                         -Pfornax.macTestArgs='--tests *LlamaModelIT' passes Gradle args
#
# The Mac needs Xcode (or its command line tools), JDK 25 (SDKMAN's newest 22+ is used when the PATH has none), CMake and Ninja on its PATH or in
# ~/.local/bin. The integration tests skip any model that isn't in its Hugging Face cache; scripts/test-models.sh, run
# on the Mac, fetches the ones they use, all small enough for a 32 GB Mac.
#
# Env: FORNAX_MAC_LOCK_WAIT, how long to wait for the Mac's ~/.local/bin/heavy-lock, where installed (seconds; empty:
# forever). The copy below happens before the lock is taken, so two runs at once need different FORNAX_MAC_DIRs.
# Env: FORNAX_MAC_DIR, the checkout's folder on the Mac (default ~/fornax-test/src). The engine sources and the native build
# stay there between runs, so only the first run builds everything.
set -euo pipefail

usage='usage: scripts/test-on-mac.sh <host> [gradle args...]'
HOST=${1:?$usage}; shift
case $HOST in -*) echo "$usage" >&2; exit 1;; esac  # -h, --help, or an ssh option where the host should be
DIR=${FORNAX_MAC_DIR:-fornax-test/src}
RDIR=$(printf '%q' "${DIR#\~/}")  # for the remote shell: a path with spaces works; ~/x is taken from the home folder
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPORTS=$ROOT/build/mac-test

# The tracked and the untracked-but-not-ignored files: what a commit of this tree would hold, sent as a tar (macOS's
# openrsync lacks the options this would need). Everything else there goes first, so a file deleted here is gone there
# too, except build/, engines-src/ and .gradle/, so a run reuses their engine checkouts, libraries and Gradle caches.
echo "==> copying the working tree to $HOST:$DIR"
(cd "$ROOT" && git ls-files -co --exclude-standard -z | tar -cf - --null --no-recursion --ignore-failed-read -T - 2> /dev/null) |
    ssh "$HOST" "mkdir -p $RDIR && cd $RDIR && find . -mindepth 1 -maxdepth 1 ! -name build ! -name engines-src ! -name .gradle -exec rm -rf {} + && tar -xf -"

# A login shell's PATH isn't there over SSH: SDKMAN's java and ~/.local/bin are put on it by hand. Gradle's own test
# exit status is kept, so the reports come back even when a test fails.
status=0
args=$( (( $# )) && printf '%q ' "$@" || true )  # quoted for the remote shell; none at all when there are none
# A shared Mac runs one GPU job at a time: where heavy-lock is installed, the build and the tests queue behind it as a
# --gpu job (they run models on Metal) of about 20 GB at peak, models included, as the Mac's memory is the GPU's; a Mac
# too small to keep heavy-lock's reserve free beside that runs it alone. FORNAX_MAC_LOCK_WAIT seconds at most, else
# forever; heavy-lock exits 125 when the wait runs out.
lock='if [ -x "$HOME/.local/bin/heavy-lock" ]; then exec env HEAVY_LOCK_WAIT="${1:-}" "$HOME/.local/bin/heavy-lock" --gpu --mem 20 fornax-mac-test -- "${@:2}"; else exec "${@:2}"; fi'
ssh "$HOST" "cd $RDIR && bash -c $(printf '%q' "$lock") locked ${FORNAX_MAC_LOCK_WAIT:-''} bash -s -- $args" <<'REMOTE' || status=$?
set -uo pipefail
[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ] && { set +u; source "$HOME/.sdkman/bin/sdkman-init.sh"; set -u; }
export PATH="$HOME/.local/bin:$PATH"
# No java 22+ on the PATH (SDKMAN with no default, or an older one): use the newest SDKMAN JDK 22+ for this run only,
# leaving the Mac's default java as it is.
major=$( (java -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p' | head -1) || true)
if [ "${major:-0}" -lt 22 ] 2>/dev/null || [ -z "${major:-}" ]; then
    for jdk in $(ls -d "$HOME"/.sdkman/candidates/java/2[2-9]* "$HOME"/.sdkman/candidates/java/[3-9][0-9]* 2>/dev/null | sort -V -r); do
        export JAVA_HOME=$jdk PATH="$jdk/bin:$PATH"; break
    done
fi
for tool in java cmake ninja xcrun; do command -v "$tool" > /dev/null || { echo "missing on the Mac: $tool" >&2; exit 2; }; done
echo "==> $(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, $(java -version 2>&1 | head -1)"
echo "==> native/build.sh metal"
native/build.sh metal > build-native.log 2>&1 || { tail -40 build-native.log; echo "native build failed (build-native.log on the Mac)" >&2; exit 3; }
echo "==> ./gradlew test -Pfornax.backend=metal $*"
./gradlew --no-daemon test -Pfornax.backend=metal --continue "$@"
REMOTE
[ "$status" -ne 125 ] || { echo "==> gave up waiting for $HOST's heavy-lock; nothing ran" >&2; exit 125; }

echo "==> test reports to $REPORTS"
rm -rf "$REPORTS" && mkdir -p "$REPORTS"
ssh "$HOST" "cd $RDIR && find . -path '*/build/reports/tests' -o -path '*/build/test-results' | COPYFILE_DISABLE=1 tar -cf - -T -" | tar -xf - -C "$REPORTS" || true
exit $status
