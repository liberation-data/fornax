---
title: Maven and Gradle
description: Add Kiln to a Maven or Gradle build from Maven Central - the org.kiln4j engine modules, the natives jar for each platform, and how a GPU backend is chosen.
---

From the first release, Kiln is on Maven Central under the group **`org.kiln4j`**. A build needs two things: an
engine module, and the **natives jar** for your platform, which holds the engines' native libraries. Your app then
needs no native build of its own.

For a complete app to copy, see [`samples/hello-kiln`](https://github.com/liberation-data/kiln/tree/main/samples/hello-kiln):
a standalone Gradle and Maven project that streams a chat reply from Qwen3 4B. Fetch the model, then run
`./gradlew run` or `mvn -q compile exec:java`. To try it against a Kiln you built, `./gradlew publishToMavenLocal`
in the Kiln repository and run the sample with `-PkilnLocal=true -PkilnVersion=0.1.0-SNAPSHOT` (Gradle) or
`-Dkiln.version=0.1.0-SNAPSHOT` (Maven).

## The artifacts

The Java modules, each with sources and javadoc jars:

| Artifact | |
|---|---|
| `kiln-llama` | language and vision language models, embeddings and reranking (llama.cpp, with `mtmd` for images) |
| `kiln-tts` | text to speech (llama.cpp's audio generation) |
| `kiln-sd` | images and video (stable-diffusion.cpp) |
| `kiln-acestep` | ACE-Step songs (acestep.cpp) |
| `kiln-yue2` | YuE2 songs (yue2.cpp) |
| `kiln-whisper` | speech to text with word timings (whisper.cpp) |
| `kiln-demucs` | stem separation: vocals and backing, drums, bass, ... (stems.cpp) |
| `kiln-core` | what they share; each engine module brings it in |

The natives, one jar per platform and GPU backend, each with all seven engines' libraries:

| Natives artifact | Platform | GPU backend |
|---|---|---|
| `kiln-natives-linux-x86_64-vulkan` | Linux x86-64 (glibc 2.39 or newer: Ubuntu 24.04, Fedora 40 or newer) | Vulkan (AMD, Intel, NVIDIA) |
| `kiln-natives-macos-aarch64-metal` | macOS 13.3 or newer on Apple silicon | Metal |
| `kiln-natives-macos-x86_64-metal` | macOS 13.3 or newer on Intel | Metal |

On Linux, the GPU's Vulkan driver must be installed; it usually is. The Linux natives jar is about 130 MB.

The Java packages keep their names: `org.liberationdata.kiln`, `org.liberationdata.kiln.llama` and so on.

## Gradle

In `build.gradle.kts`, pick the natives jar for the machine that builds:

```kotlin
val kilnVersion = "0.1.0"
val kilnNatives = when {
    !System.getProperty("os.name").startsWith("Mac") -> "linux-x86_64-vulkan"
    System.getProperty("os.arch") == "aarch64" -> "macos-aarch64-metal"
    else -> "macos-x86_64-metal"
}

dependencies {
    implementation("org.kiln4j:kiln-llama:$kilnVersion")
    runtimeOnly("org.kiln4j:kiln-natives-$kilnNatives:$kilnVersion")
}
```

Add more engine modules (`kiln-sd`, `kiln-tts`, ...) the same way; one natives jar covers all of them.

## Maven

A profile per platform, activated by the OS, names the natives jar:

```xml
<properties>
    <kiln.version>0.1.0</kiln.version>
</properties>

<dependencies>
    <dependency>
        <groupId>org.kiln4j</groupId>
        <artifactId>kiln-llama</artifactId>
        <version>${kiln.version}</version>
    </dependency>
    <dependency>
        <groupId>org.kiln4j</groupId>
        <artifactId>kiln-natives-${kiln.natives}</artifactId>
        <version>${kiln.version}</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>

<profiles>
    <profile>
        <id>kiln-linux-x86_64</id>
        <activation><os><name>Linux</name><arch>amd64</arch></os></activation>
        <properties><kiln.natives>linux-x86_64-vulkan</kiln.natives></properties>
    </profile>
    <profile>
        <id>kiln-macos-aarch64</id>
        <activation><os><family>mac</family><arch>aarch64</arch></os></activation>
        <properties><kiln.natives>macos-aarch64-metal</kiln.natives></properties>
    </profile>
    <profile>
        <id>kiln-macos-x86_64</id>
        <activation><os><family>mac</family><arch>x86_64</arch></os></activation>
        <properties><kiln.natives>macos-x86_64-metal</kiln.natives></properties>
    </profile>
</profiles>
```

## Running

Run with `--enable-native-access=ALL-UNNAMED`, as for any Kiln app.

The first time an engine loads, Kiln extracts its library from the natives jar into a per-version cache and loads it
from there:

- Linux: `~/.cache/kiln/<version>/<platform>/` (`$XDG_CACHE_HOME/kiln/...` when that is set)
- macOS: `~/Library/Caches/kiln/<version>/<platform>/`

Move the cache with `-Dkiln.cache.dir` or `KILN_CACHE_DIR`. Each copy is written to a temporary file and moved into
place, so several JVMs starting at once never load a half-written library, and a copy is reused for as long as it
matches the jar's.

## Choosing a backend

Each natives jar is one platform and one GPU backend, so **the jar you add is the choice**.

- Natives jars for different platforms can sit side by side. To ship one build for every platform, add all three;
  Kiln loads the one for the machine it runs on.
- If jars for several backends of the same platform are on the classpath, `-Dkiln.backend=vulkan` (or the
  `KILN_BACKEND` environment variable) picks one. Otherwise Kiln prefers Metal, then CUDA, HIP, Vulkan and the CPU,
  in that order.
- For HIP (ROCm), CUDA or anything else without a natives jar, build the libraries yourself
  ([Choosing a GPU backend](/docs/guides/gpu-backends/)) and point `-Dkiln.library.path` at them. An explicit path
  always wins over a natives jar.

The full search order is then:

1. `-Dkiln.<name>.lib`, an exact file;
2. `-Dkiln.library.path`;
3. `KILN_LIBRARY_PATH`;
4. a natives jar on the classpath;
5. the platform loader's path.
