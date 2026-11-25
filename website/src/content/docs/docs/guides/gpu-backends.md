---
title: Choosing a GPU backend
description: Build Kiln's engines for Vulkan, HIP (ROCm), CUDA, Metal or the CPU, and the settings that matter on each.
---

Kiln's Java code is the same for every GPU. The backend is chosen when you build the native libraries: each build
goes to its own folder, `build/native/<os>-<arch>-<backend>/`, and your app loads whichever folder you point it at.

```sh
native/build.sh vulkan     # or hip | cuda | metal | cpu
native/build.sh hip sd     # one backend, some engines: llama tts sd acestep yue2
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
add `-Pkiln.backend=hip` (or `cuda`, `metal`, `cpu`) to a Gradle command, or change `kiln.backend` in
`gradle.properties`.

## Which one?

| Backend | GPUs | State |
|---|---|---|
| [Vulkan](#vulkan) | AMD, Intel, NVIDIA | the backend Kiln is developed and tested on |
| [HIP (ROCm)](#hip-rocm) | AMD | builds; has run LTX video on the Radeon 8060S, faster than Vulkan |
| [CUDA](#cuda) | NVIDIA | builds from the same script; not tested yet |
| [Metal](#metal) | Macs | builds from the same script; not tested yet |
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

**The driver's compute timeout.** A GPU driver resets a GPU that runs one piece of work for too long. ggml's Vulkan
backend puts up to 100 graph nodes in each submission to the GPU, and a big decode can then run past the timeout and
lose the device. Lower it:

```sh
java -Dkiln.vulkan.maxNodesPerSubmit=8 ...        # sets GGML_VK_MAX_NODES_PER_SUBMIT for the engines
```

This is most likely on a power-limited GPU, such as a laptop on battery, and for video. A single very long dispatch
(attention over a long video clip) can't be split at all; use [HIP](#hip-rocm) for those.

**bf16 on RADV.** The Mesa driver for the Radeon 8060S reports no bf16 support, so bf16 work falls back to the CPU. A
checkpoint that mixes bf16 and f32 can then abort. See [Troubleshooting](/docs/troubleshooting/#bf16-checkpoints-on-amd-vulkan-radv).

## HIP (ROCm)

For AMD GPUs, through ROCm. Building needs ROCm's HIP compiler, hipBLAS and rocBLAS. Set `HIP_ARCH` to your GPU's
target (`gfx1151` is the Radeon 8060S in Strix Halo; `rocminfo` names yours):

```sh
HIP_ARCH=gfx1100 native/build.sh hip
./gradlew video -Pkiln.backend=hip --args='clip.mp4 "a potter at a wheel"'
```

What Kiln has learned on HIP so far, with LTX-2.5 video on the 8060S:

- **No compute timeout.** ROCm work goes through user queues that the driver's compute timeout doesn't cover, so the
  long clips that lose the device on Vulkan finish on HIP.
- **Faster, here.** About 2.7× faster per LTX step than Vulkan (44 s against 119 s at 672×448×313, after a slower
  first step).
- **GPU graphs are off.** ggml can capture its work as a HIP graph, and ROCm schedules one recursively, a stack frame
  per node. On a JVM thread, whose stack is smaller than a command-line tool's 8 MB main thread, that overflowed and
  crashed during LTX's text encoding. So Kiln sets `GGML_CUDA_DISABLE_GRAPHS=1` before the first engine loads. Turn
  graphs back on with `-Dkiln.gpu.graphs=true`.

## CUDA

For NVIDIA GPUs. Building needs the CUDA toolkit (with cuBLAS) at `CUDA_PATH`:

```sh
native/build.sh cuda
```

`-Dkiln.gpu.graphs` applies to CUDA too: Kiln turns ggml's CUDA graphs off by default, as on HIP. Whether CUDA
needs that hasn't been tested; `-Dkiln.gpu.graphs=true` turns them back on.

## Metal

For Macs. The build embeds the Metal shader library in each engine, so there are no separate shader files to ship.
Building needs Apple's command-line developer tools, for the compiler and the Metal frameworks.

```sh
native/build.sh metal
```

The libraries are `.dylib`s, in `build/native/macos-arm64-metal/` on Apple silicon.

## CPU

No GPU at all. Useful to check that a setup works, or on a machine without a supported GPU, but much slower.

```sh
native/build.sh cpu
```

## Settings, at a glance

Kiln passes these to the engines as environment variables before the first engine loads. A variable you set
yourself always wins.

| Setting | Default | Effect |
|---|---|---|
| `-Dkiln.gpu.graphs` | `false` | ggml's CUDA and HIP graph capture; `false` sets `GGML_CUDA_DISABLE_GRAPHS=1` |
| `-Dkiln.vulkan.maxNodesPerSubmit` | ggml's (100) | graph nodes per Vulkan submission; sets `GGML_VK_MAX_NODES_PER_SUBMIT` |

And these the engines read themselves:

| Variable | Default | Effect |
|---|---|---|
| `KILN_VERBOSE` | unset | `1` shows the engines' own logs; `debug` (for stable-diffusion.cpp) shows everything |
| `KILN_ACESTEP_VAE_CHUNK` | `512` | frames per ACE-Step VAE decode tile; smaller means shorter GPU dispatches |
| `GGML_VK_MAX_NODES_PER_SUBMIT` | 100 | as `-Dkiln.vulkan.maxNodesPerSubmit` |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` (set by Kiln) | as `-Dkiln.gpu.graphs` |

Where your app finds the libraries (`-Dkiln.library.path` and the rest) is on
[Getting started](/docs/getting-started/#in-your-own-application).
