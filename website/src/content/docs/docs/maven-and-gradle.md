---
title: Maven and Gradle
description: Add Fornax to a Maven or Gradle build from Maven Central - the org.fornax4j engine modules, the natives jar for each platform, and how a GPU backend is chosen.
---

From the first release, Fornax is on Maven Central under the group **`org.fornax4j`**. A build needs two things: an
engine module, and the **natives jar** for your platform, which holds the engines' native libraries. Your app then
needs no native build of its own.

For a complete app to copy, see [`samples/hello-fornax`](https://github.com/liberation-data/fornax/tree/main/samples/hello-fornax):
a standalone Gradle and Maven project that streams a chat reply from Qwen3 4B. Fetch the model, then run
`./gradlew run` or `mvn -q compile exec:java`. To try it against a Fornax you built, `./gradlew publishToMavenLocal`
in the Fornax repository and run the sample with `-PfornaxLocal=true -PfornaxVersion=0.2.0-SNAPSHOT` (Gradle) or
`-Dfornax.version=0.2.0-SNAPSHOT` (Maven).

## The artifacts

The Java modules, each with sources and javadoc jars:

| Artifact | |
|---|---|
| `fornax-llama` | language and vision language models, embeddings and reranking (llama.cpp, with `mtmd` for images) |
| `fornax-tts` | text to speech (llama.cpp's audio generation) |
| `fornax-sd` | images and video (stable-diffusion.cpp) |
| `fornax-acestep` | ACE-Step songs (acestep.cpp) |
| `fornax-yue2` | YuE2 songs (yue2.cpp) |
| `fornax-whisper` | speech to text with word timings (whisper.cpp) |
| `fornax-demucs` | stem separation: vocals and backing, drums, bass, ... (stems.cpp) |
| `fornax-core` | what they share; each engine module brings it in |

The natives, one jar per platform and GPU backend, each with all seven engines' libraries:

| Natives artifact | Platform | GPU backend |
|---|---|---|
| `fornax-natives-linux-x86_64-vulkan` | Linux x86-64 (glibc 2.39 or newer: Ubuntu 24.04, Fedora 40 or newer) | Vulkan (AMD, Intel, NVIDIA) |
| `fornax-natives-macos-aarch64-metal` | macOS 13.3 or newer on Apple silicon | Metal |
| `fornax-natives-macos-x86_64-metal` | macOS 13.3 or newer on Intel | Metal |

On Linux, the GPU's Vulkan driver must be installed; it usually is. The Linux natives jar is about 130 MB.

The Java packages keep their names: `org.liberationdata.fornax`, `org.liberationdata.fornax.llama` and so on.

## Gradle

In `build.gradle.kts`, pick the natives jar for the machine that builds:

```kotlin
val fornaxVersion = "0.2.0"
val fornaxNatives = when {
    !System.getProperty("os.name").startsWith("Mac") -> "linux-x86_64-vulkan"
    System.getProperty("os.arch") == "aarch64" -> "macos-aarch64-metal"
    else -> "macos-x86_64-metal"
}

dependencies {
    implementation("org.fornax4j:fornax-llama:$fornaxVersion")
    runtimeOnly("org.fornax4j:fornax-natives-$fornaxNatives:$fornaxVersion")
}
```

Add more engine modules (`fornax-sd`, `fornax-tts`, ...) the same way; one natives jar covers all of them.

## Maven

A profile per platform, activated by the OS, names the natives jar:

```xml
<properties>
    <fornax.version>0.2.0</fornax.version>
</properties>

<dependencies>
    <dependency>
        <groupId>org.fornax4j</groupId>
        <artifactId>fornax-llama</artifactId>
        <version>${fornax.version}</version>
    </dependency>
    <dependency>
        <groupId>org.fornax4j</groupId>
        <artifactId>fornax-natives-${fornax.natives}</artifactId>
        <version>${fornax.version}</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>

<profiles>
    <profile>
        <id>fornax-linux-x86_64</id>
        <activation><os><name>Linux</name><arch>amd64</arch></os></activation>
        <properties><fornax.natives>linux-x86_64-vulkan</fornax.natives></properties>
    </profile>
    <profile>
        <id>fornax-macos-aarch64</id>
        <activation><os><family>mac</family><arch>aarch64</arch></os></activation>
        <properties><fornax.natives>macos-aarch64-metal</fornax.natives></properties>
    </profile>
    <profile>
        <id>fornax-macos-x86_64</id>
        <activation><os><family>mac</family><arch>x86_64</arch></os></activation>
        <properties><fornax.natives>macos-x86_64-metal</fornax.natives></properties>
    </profile>
</profiles>
```

## Running

Run with `--enable-native-access=ALL-UNNAMED`, as for any Fornax app.

The first time an engine loads, Fornax extracts its library from the natives jar into a per-version cache and loads it
from there:

- Linux: `~/.cache/fornax/<version>/<platform>/` (`$XDG_CACHE_HOME/fornax/...` when that is set)
- macOS: `~/Library/Caches/fornax/<version>/<platform>/`

Move the cache with `-Dfornax.cache.dir` or `FORNAX_CACHE_DIR`. Each copy is written to a temporary file and moved into
place, so several JVMs starting at once never load a half-written library, and a copy is reused for as long as it
matches the jar's.

## Choosing a backend

Each natives jar is one platform and one GPU backend, so **the jar you add is the choice**.

- Natives jars for different platforms can sit side by side. To ship one build for every platform, add all three;
  Fornax loads the one for the machine it runs on.
- If jars for several backends of the same platform are on the classpath, `-Dfornax.backend=vulkan` (or the
  `FORNAX_BACKEND` environment variable) picks one. Otherwise Fornax prefers Metal, then CUDA, HIP, Vulkan and the CPU,
  in that order.
- For HIP (ROCm), CUDA or anything else without a natives jar, build the libraries yourself
  ([Choosing a GPU backend](/docs/guides/gpu-backends/)) and point `-Dfornax.library.path` at them. An explicit path
  always wins over a natives jar.

The full search order is then:

1. `-Dfornax.<name>.lib`, an exact file;
2. `-Dfornax.library.path`;
3. `FORNAX_LIBRARY_PATH`;
4. a natives jar on the classpath;
5. the platform loader's path.
