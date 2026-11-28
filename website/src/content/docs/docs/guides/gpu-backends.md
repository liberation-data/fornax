---
title: Choosing a GPU backend
description: Build Fornax's engines for Vulkan, HIP (ROCm), CUDA, Metal or the CPU, and the settings that matter on each.
---

Fornax's Java code is the same for every GPU. The backend is chosen when you build the native libraries: each build
goes to its own folder, `build/native/<os>-<arch>-<backend>/`, and your app loads whichever folder you point it at.

```sh
native/build.sh vulkan     # or hip | cuda | metal | cpu
native/build.sh hip sd     # one backend, some engines: llama tts sd acestep yue2 whisper demucs
```

Every backend needs JDK 25, CMake, Ninja and a C++17 compiler. The build script reads a few environment variables:

| Variable | Default | |
|---|---|---|
| `JOBS` | the number of CPUs | parallel build jobs |
| `HIP_ARCH` | `gfx1151` | the AMD GPU target for HIP |
| `ROCM_PATH` | `/opt/rocm` | where ROCm is; falls back to `/usr/lib64` (Fedora's own ROCm lives in `/usr`) |
| `CUDA_PATH` | `/usr/local/cuda` | where the CUDA toolkit is |
| `CMAKE_EXTRA` | | extra CMake arguments for every engine |

The Gradle build runs the examples and tests against `build/native/<os>-<arch>-vulkan/` unless you say otherwise:
add `-Pfornax.backend=hip` (or `cuda`, `metal`, `cpu`) to a Gradle command, or change `fornax.backend` in
`gradle.properties`.

## Which one?

| Backend | GPUs | State |
|---|---|---|
| [Vulkan](#vulkan) | AMD, Intel, NVIDIA | the backend Fornax is developed and tested on |
| [HIP (ROCm)](#hip-rocm) | AMD | passes the test suite nightly on the Radeon 8060S; runs LTX video faster than Vulkan |
| [CUDA](#cuda) | NVIDIA | builds from the same script; not tested yet |
| [Metal](#metal) | Macs | passes the test suite nightly on an M2 Max; songs and video not run yet |
| [CPU](#cpu) | none | for checking a setup; slow |

Start with Vulkan unless you have a reason not to. Reports and fixes for the untested backends are welcome.

## Vulkan

Vulkan runs on AMD, Intel and NVIDIA GPUs through their ordinary graphics drivers. Building needs the Vulkan headers,
`glslc` and SPIRV-Headers; on Fedora:

```sh
sudo dnf install vulkan-headers vulkan-loader-devel glslc spirv-headers-devel
native/build.sh vulkan
```

Everything on the [Introduction](/docs/#status)'s results table ran on Vulkan, on an AMD Radeon 8060S.

**Running it** needs your GPU's Vulkan driver and the Vulkan loader. Desktop Linux usually has both; see
[Installing GPU drivers](/docs/gpu-drivers/) if not. Fornax won't quietly fall back to the CPU: with no GPU it stops and
says what to install.

**The driver's compute timeout.** A GPU driver resets a GPU that runs one piece of work for too long. ggml's Vulkan
backend puts up to 100 graph nodes in each submission to the GPU, and a big decode can then run past the timeout and
lose the device. Lower it:

```sh
java -Dfornax.vulkan.maxNodesPerSubmit=8 ...        # sets GGML_VK_MAX_NODES_PER_SUBMIT for the engines
```

This is most likely on a power-limited GPU, such as a laptop on battery, and for video. A single very long dispatch
(attention over a long video clip) can't be split at all; use [HIP](#hip-rocm) for those.

**bf16 on RADV.** The Mesa driver for the Radeon 8060S reports no bf16 support, so bf16 work falls back to the CPU. A
checkpoint that mixes bf16 and f32 can then abort. See [Troubleshooting](/docs/troubleshooting/#bf16-checkpoints-on-amd-vulkan-radv).

## HIP (ROCm)

For AMD GPUs, through ROCm. Building needs ROCm's HIP compiler, hipBLAS and rocBLAS. Set `HIP_ARCH` to your GPU's
target (`gfx1151` is the Radeon 8060S in Strix Halo; `rocminfo` names yours).
[Set up on your platform](/docs/platforms/#linux-amd-with-rocm) has the whole setup, from installing ROCm to pointing
your app at the libraries.

```sh
HIP_ARCH=gfx1100 native/build.sh hip
./gradlew video -Pfornax.backend=hip --args='clip.mp4 "a potter at a wheel"'
```

The test suite passes on HIP on the 8060S (ROCm 7, `gfx1151`) every night, with the models downloaded for
embeddings, reranking and Demucs. A borderline rerank score (near 0.5) moves by up to about 0.06 between scoring a
document alone and in a batch, on HIP as in llama.cpp's own `llama-embedding`; decided scores and the ranking hold
([#80](https://github.com/liberation-data/fornax/issues/80)).

What Fornax has learned on HIP so far, with LTX-2.5 video on the 8060S:

- **No compute timeout.** ROCm work goes through user queues that the driver's compute timeout doesn't cover, so the
  long clips that lose the device on Vulkan finish on HIP.
- **Faster, here.** About 2.7× faster per LTX step than Vulkan (44 s against 119 s at 672×448×313, after a slower
  first step).
- **GPU graphs are off.** ggml can capture its work as a HIP graph, and ROCm schedules one recursively, a stack frame
  per node. On a JVM thread, whose stack is smaller than a command-line tool's 8 MB main thread, that overflowed and
  crashed during LTX's text encoding. So Fornax sets `GGML_CUDA_DISABLE_GRAPHS=1` before the first engine loads. Turn
  graphs back on with `-Dfornax.gpu.graphs=true`.

## CUDA

For NVIDIA GPUs. Building needs the CUDA toolkit (with cuBLAS) at `CUDA_PATH`:

```sh
native/build.sh cuda
```

`-Dfornax.gpu.graphs` applies to CUDA too: Fornax turns ggml's CUDA graphs off by default, as on HIP. Whether CUDA
needs that hasn't been tested; `-Dfornax.gpu.graphs=true` turns them back on.

## Metal

For Macs. The build embeds the Metal shader library in each engine, so there are no separate shader files to ship.
Building needs Apple's command-line developer tools, for the compiler and the Metal frameworks.

```sh
native/build.sh metal
```

The libraries are `.dylib`s, in `build/native/macos-arm64-metal/` on Apple silicon.

Fornax's test suite passes on Metal on an Apple M2 Max with 32 GB (macOS 27.2), with every layer on the GPU: chat,
vision, embeddings and reranking, speech, transcription, stem separation, FLUX.2 [klein] images and upscaling, isolated
workers and Dispatch. A borderline reranker score differs from Vulkan's by about 0.05, as kernels do; the ranking is
the same. Songs (ACE-Step, YuE2) and video haven't been run on a Mac yet.
Intel Macs aren't supported. To run the tests on a Mac from a Linux checkout, see `scripts/test-on-mac.sh` (or
`./gradlew testOnMac -Pfornax.macHost=<host>`).

## CPU

No GPU at all. Useful to check that a setup works, or on a machine without a supported GPU, but much slower.

```sh
native/build.sh cpu
```

## Settings, at a glance

Fornax passes these to the engines as environment variables before the first engine loads. A variable you set
yourself always wins.

| Setting | Default | Effect |
|---|---|---|
| `-Dfornax.gpu.graphs` | `false` | ggml's CUDA and HIP graph capture; `false` sets `GGML_CUDA_DISABLE_GRAPHS=1` |
| `-Dfornax.vulkan.maxNodesPerSubmit` | ggml's (100) | graph nodes per Vulkan submission; sets `GGML_VK_MAX_NODES_PER_SUBMIT` |

And these the engines read themselves:

| Variable | Default | Effect |
|---|---|---|
| `FORNAX_VERBOSE` | unset | `1` shows the engines' own logs; `debug` (for stable-diffusion.cpp) shows everything |
| `FORNAX_ACESTEP_VAE_CHUNK` | `512` | frames per ACE-Step VAE decode tile; smaller means shorter GPU dispatches |
| `GGML_VK_MAX_NODES_PER_SUBMIT` | 100 | as `-Dfornax.vulkan.maxNodesPerSubmit` |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` (set by Fornax) | as `-Dfornax.gpu.graphs` |

Where your app finds the libraries (`-Dfornax.library.path` and the rest) is on
[Getting started](/docs/getting-started/#in-your-own-application).
