#!/bin/sh
# Installs the fornax command (https://fornax4j.org/docs/guides/cli/) for one user, on Linux x86_64 or an Apple-silicon
# Mac. Read it before you run it:
#
#   curl -fsSLO https://fornax4j.org/install.sh
#   less install.sh
#   sh install.sh
#
# What it does, and nothing else:
#
#   1. picks this machine's natives: linux-x86_64-vulkan or macos-aarch64-metal;
#   2. downloads SHA256SUMS and fornax-cli-<version>-<platform>.zip from a Fornax release on GitHub
#      (https://github.com/liberation-data/fornax/releases): the latest, or the one FORNAX_VERSION names;
#   3. checks the zip's SHA-256 against SHA256SUMS, and stops if they differ;
#   4. unpacks it into ~/.local/share/fornax/fornax-cli-<version>, points ~/.local/share/fornax/current at it,
#      deletes the versions it installed there before, and links ~/.local/bin/fornax to current/bin/fornax;
#   5. looks for Java 22 or newer, which fornax runs on, and if there is none says how to get one. It installs no Java.
#
# It needs no root, writes only into those two folders and edits no shell startup file: when ~/.local/bin isn't on
# PATH, it prints the line to add. Run it again to update fornax. To uninstall, delete ~/.local/share/fornax (or the
# FORNAX_HOME below) and ~/.local/bin/fornax.
#
#   FORNAX_VERSION   the release to install, such as 0.3.0 (default: the latest)
#   FORNAX_HOME      the folder the versions go in (default: $XDG_DATA_HOME/fornax, else ~/.local/share/fornax)
#   FORNAX_BIN_DIR   the folder the fornax link goes in (default: ~/.local/bin)
#   FORNAX_DIST_URL  the folder SHA256SUMS and the zips are read from instead of the release, https:// or file://
#                    (for a mirror, or to try a locally built zip)
#
# fornax itself reads FORNAX_JAVA, JAVA_HOME or java on PATH for its Java, as this script does to check it.

set -eu

REPO=https://github.com/liberation-data/fornax

say() { printf '%s\n' "$*"; }
warn() { printf 'fornax install: %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }
has() { command -v "$1" > /dev/null 2>&1; }

# This machine's natives, as the release names them.
detect_platform() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os-$arch" in
        Linux-x86_64 | Linux-amd64) echo linux-x86_64-vulkan ;;
        Darwin-arm64) echo macos-aarch64-metal ;;
        Darwin-x86_64)
            # An Apple-silicon Mac whose shell runs under Rosetta reports x86_64.
            if [ "$(sysctl -n sysctl.proc_translated 2> /dev/null || true)" = 1 ]; then
                echo macos-aarch64-metal
            else
                die "Fornax has no natives for Intel Macs yet ($REPO/issues/123). See $REPO#building to build them."
            fi
            ;;
        *) die "Fornax has no natives for $os $arch; it ships them for Linux x86_64 (Vulkan) and Apple-silicon Macs (Metal). See $REPO#building." ;;
    esac
}

# fetch URL FILE: https:// with curl or wget, or file:// by copying. Fails, rather than dies, on a file that isn't
# there, so the caller can say what that means.
fetch() {
    case "$1" in
        file://*) cp "${1#file://}" "$2" || return 1 ;;
        https://*)
            if has curl; then
                if [ -t 2 ] && [ "${3:-}" = progress ]; then bar=--progress-bar; else bar=--silent; fi
                curl --fail --location --show-error $bar --proto '=https' --tlsv1.2 --output "$2" "$1" || return 1
            # --https-only keeps a redirect off plain http. BusyBox's wget has no such option and would fail here, which
            # the caller would report as a missing file.
            elif has wget && wget --help 2>&1 | grep -q -e --https-only; then
                wget --quiet --https-only --output-document="$2" "$1" || return 1
            else
                die "needs curl, or a wget with --https-only (BusyBox's has none), to download $1"
            fi
            ;;
        *) die "not an https:// or file:// URL: $1" ;;
    esac
}

sha256() {
    if has sha256sum; then sha256sum "$1" | cut -d ' ' -f 1
    elif has shasum; then shasum -a 256 "$1" | cut -d ' ' -f 1
    elif has openssl; then openssl dgst -sha256 -r "$1" | cut -d ' ' -f 1
    else die "needs sha256sum, shasum or openssl to check the download"
    fi
}

# unpack ZIP DIR
unpack() {
    if has unzip; then unzip -q "$1" -d "$2"
    elif has bsdtar; then bsdtar -xf "$1" -C "$2"
    elif has python3; then python3 -m zipfile -e "$1" "$2"
    else die "needs unzip (or bsdtar, or python3) to unpack $1"
    fi
}

# The major version of the java at $1 ("25", or "8" for 1.8), or nothing if it doesn't run.
java_major() {
    "$1" -version 2>&1 | sed -n -e 's/.* version "1\.\([0-9][0-9]*\).*/\1/p' -e 's/.* version "\([0-9][0-9]*\).*/\1/p' | head -n 1
}

# Says whether the Java fornax will find is 22 or newer, and how to get one if it isn't. Never fails the install.
check_java() {
    if [ -n "${FORNAX_JAVA:-}" ]; then java=$FORNAX_JAVA; from=FORNAX_JAVA
    elif [ -n "${JAVA_HOME:-}" ]; then java=$JAVA_HOME/bin/java; from=JAVA_HOME
    else java=$(command -v java || true); from=PATH
    fi
    major=
    if [ -n "$java" ] && has "$java"; then major=$(java_major "$java" || true); fi
    if [ -n "$major" ] && [ "$major" -ge 22 ]; then
        say "Java:    $major, at $java (from $from)"
        return 0
    fi
    say
    if [ -n "$major" ]; then
        say "fornax needs Java 22 or newer, and the one it would run on is Java $major ($java, from $from)."
    elif [ -n "$java" ]; then
        say "fornax needs Java 22 or newer, and $java (from $from) doesn't run."
    else
        say "fornax needs Java 22 or newer, and found none (no FORNAX_JAVA or JAVA_HOME, and no working java on PATH)."
    fi
    if [ "$(uname -s)" = Darwin ] && other=$(/usr/libexec/java_home -v 22+ 2> /dev/null); then
        say "This Mac has one at $other: export JAVA_HOME=\"$other\""
    else
        say "Any of these gets you one; this script installs none itself:"
        say "  SDKMAN (https://sdkman.io):  sdk install java           # then it's on PATH"
        say "  JBang (https://jbang.dev):   jbang jdk install 25 && export JAVA_HOME=\"\$(jbang jdk home 25)\""
        if [ "$(uname -s)" = Darwin ]; then
            say "  Homebrew:                    brew install openjdk && export JAVA_HOME=\"\$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home\""
        else
            say "  Your package manager's OpenJDK, 22 or newer (Fedora: java-25-openjdk-headless; Debian and Ubuntu: openjdk-25-jre-headless where offered)"
        fi
        say "  A download from https://adoptium.net, with JAVA_HOME set to it"
    fi
    say "fornax uses FORNAX_JAVA if set, else \$JAVA_HOME/bin/java, else java on PATH."
}

# The engines on Linux need the Vulkan loader; say so if it isn't there. Never fails the install.
check_vulkan() {
    for ldconfig in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
        if has "$ldconfig"; then
            "$ldconfig" -p 2> /dev/null | grep -q 'libvulkan\.so\.1' && return 0
            say
            say "No Vulkan loader (libvulkan.so.1) found. fornax's engines need it, and a Vulkan driver for your GPU:"
            say "  Fedora: vulkan-loader and mesa-vulkan-drivers; Debian and Ubuntu: libvulkan1 and mesa-vulkan-drivers"
            return 0
        fi
    done
}

main() {
    platform=$(detect_platform)
    data=${XDG_DATA_HOME:-$HOME/.local/share}
    home=${FORNAX_HOME:-$data/fornax}
    bin=${FORNAX_BIN_DIR:-$HOME/.local/bin}
    version=${FORNAX_VERSION:-}
    version=${version#v}
    if [ -n "${FORNAX_DIST_URL:-}" ]; then base=${FORNAX_DIST_URL%/}
    elif [ -n "$version" ]; then base=$REPO/releases/download/v$version
    else base=$REPO/releases/latest/download
    fi

    mkdir -p "$home" "$bin"
    home=$(cd "$home" && pwd -P)

    # Refuse before downloading anything to replace a fornax this script didn't put there (Fornax Studio's launcher,
    # say). Its own link points at $home/current/bin/fornax.
    link=$bin/fornax
    if [ -e "$link" ] || [ -L "$link" ]; then
        if [ ! -L "$link" ] || [ "$(readlink "$link")" != "$home/current/bin/fornax" ]; then
            die "$link is already there and isn't this script's link (it may be Fornax Studio's fornax). Remove it, or set FORNAX_BIN_DIR to another folder."
        fi
    fi
    if [ -e "$home/current" ] && [ ! -L "$home/current" ]; then
        die "$home/current is there and isn't a link; set FORNAX_HOME to another folder"
    fi

    work=$(mktemp -d "$home/.install.XXXXXX")
    trap 'rm -rf "$work"' EXIT
    trap 'exit 130' INT TERM

    say "Fornax:  ${version:-latest} for $platform, from $base"
    fetch "$base/SHA256SUMS" "$work/SHA256SUMS" ||
        die "no SHA256SUMS at $base. Releases before 0.3.0 have no fornax-cli download; other ways to get fornax: https://fornax4j.org/docs/guides/cli/#installing-it"

    # The zip for this platform: "<sha256>  fornax-cli-<version>-<platform>.zip" (or "*fornax-cli-..." in binary mode)
    line=$(awk -v suffix="-$platform.zip" '{ f = $2; sub(/^\*/, "", f) }
        f ~ /^fornax-cli-/ && length(f) > length(suffix) && substr(f, length(f) - length(suffix) + 1) == suffix { print $1, f; exit }' "$work/SHA256SUMS")
    [ -n "$line" ] || die "$base has no fornax-cli download for $platform (SHA256SUMS lists none)"
    want=$(echo "${line%% *}" | tr 'A-F' 'a-f')
    zip=${line#* }
    found=${zip#fornax-cli-}
    found=${found%-"$platform".zip}
    echo "$found" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$' || die "not a version: $found (from $zip)"
    echo "$want" | grep -Eq '^[0-9a-f]{64}$' || die "not a SHA-256 for $zip in SHA256SUMS: $want"
    [ -z "$version" ] || [ "$version" = "$found" ] || die "asked for $version, but $base has $zip"
    version=$found

    say "Downloading $zip"
    fetch "$base/$zip" "$work/$zip" progress || die "couldn't download $base/$zip"
    got=$(sha256 "$work/$zip")
    [ "$got" = "$want" ] || die "$zip's SHA-256 is $got, but SHA256SUMS says $want: not installing it"
    say "SHA-256: $got, as SHA256SUMS says"

    mkdir "$work/unpacked"
    unpack "$work/$zip" "$work/unpacked"
    dist=$work/unpacked/fornax-cli-$version
    if [ ! -f "$dist/bin/fornax" ] || [ ! -f "$dist/lib/fornax-cli-$version.jar" ]; then
        die "$zip doesn't hold fornax-cli-$version/bin/fornax and lib/fornax-cli-$version.jar"
    fi
    chmod +x "$dist/bin/fornax"  # python3's zipfile doesn't keep the mode

    target=$home/fornax-cli-$version
    rm -rf "$target"
    mv "$dist" "$target"
    ln -sfn "fornax-cli-$version" "$home/current"
    for old in "$home"/fornax-cli-*; do
        if [ -d "$old" ] && [ "$old" != "$target" ]; then rm -rf "$old"; fi
    done
    ln -sfn "$home/current/bin/fornax" "$link"
    say "Installed fornax $version in $target, linked from $link"

    check_java
    case "$platform" in linux-*) check_vulkan ;; esac

    say
    case ":$PATH:" in
        *":$bin:"*)
            first=$(command -v fornax || true)
            if [ -n "$first" ] && [ "$first" != "$link" ]; then
                say "Another fornax comes first on PATH: $first. Run this one as $link, or put $bin earlier on PATH."
            else
                say "Run: fornax chat qwen3-4b \"Write a haiku about a kiln\""
            fi
            ;;
        *)
            case "$bin" in "$HOME"/*) shown=\$HOME${bin#"$HOME"} ;; *) shown=$bin ;; esac
            say "$bin isn't on your PATH. Add this line to your shell's startup file (~/.zshrc, ~/.bashrc, ...):"
            say
            say "  export PATH=\"$shown:\$PATH\""
            say
            say "Then: fornax chat qwen3-4b \"Write a haiku about a kiln\""
            ;;
    esac
}

# Everything runs from here, so a download cut short runs nothing.
main "$@"
