---
title: Engine notes and troubleshooting
description: How to debug a model whose output looks wrong, and what the engines under Fornax do that surprises people - with the fix for each.
---

Fornax runs the engines' own code, so most surprises are the engines'. This page collects what was learned getting
them to behave, so you don't have to learn it again. The notes these come from are in the repository's
[`docs/engine-notes.md`](https://github.com/liberation-data/fornax/blob/main/docs/engine-notes.md); add to them when an
engine surprises you.

## First steps

- **Turn the logs on.** The engines are quiet apart from errors. `FORNAX_VERBOSE=1` shows their own logs, and
  `FORNAX_VERBOSE=debug` shows everything stable-diffusion.cpp does, including the program it compiles for each
  generation.
- **Read the exception.** Each model class throws an `IllegalStateException` (or, in `fornax-sd`, an `Sd.SdException`)
  with the engine's own error message when a native call fails.
- **"Fornax could not find libfornax_….so".** The libraries aren't where Fornax looks. Add the natives jar for your
  platform ([Maven and Gradle](/docs/maven-and-gradle/)), or build them with `native/build.sh` and point
  `-Dfornax.library.path` at the folder; [Getting started](/docs/getting-started/#in-your-own-application)
  has the search order.
- **"Not in the Hugging Face cache".** `HfCache.require` didn't find the file; the message ends with the
  `hf download` command that fetches it.

## "Fornax found no GPU"

Fornax stops rather than run a GPU build on the CPU. Install your GPU's driver: [Installing GPU drivers](/docs/gpu-drivers/)
has the steps for each distribution, and how to check with `vulkaninfo --summary`. To run on the CPU anyway, set
`-Dfornax.allowCpu=true` (or `FORNAX_ALLOW_CPU=true`).

## When a model's output looks wrong

1. **Compare against the engine's own tool first.** Before guessing at causes, run the same settings through the
   engine's command-line tool (`sd-cli` for stable-diffusion.cpp, `llama-cli` or `llama-tts` for llama.cpp). If the
   tool is right and Fornax is wrong, the problem is in Fornax. For stable-diffusion.cpp,
   `scripts/compare-with-sdcli.sh` diffs the program the engine compiles for each side, in seconds and without
   rendering anything.
2. **Render small first.** For LTX video, 336×224 and 49 frames took about 3 minutes and showed every artifact the
   1344×896 renders did (those took 10 to 30 minutes).
3. **Split a pipeline into its stages.** For LTX's two-stage recipe, rendering the draft alone
   (`"hires.enabled", false`, or `-Phires=false` in the example) showed that the draft was clean and the refine
   wasn't.

## Long GPU work loses the device

**What you see:** the GPU resets mid-run, and the engine reports a lost device. It happens more on battery or on a
power-limited GPU.

**Why:** the GPU driver gives one piece of work a time limit. ggml's Vulkan backend puts up to 100 graph nodes in each
submission, and one big decode (LTX's audio VAE at full size, for example) can run past the amdgpu driver's timeout.
ComfyUI never hits this, because PyTorch on ROCm launches many small kernels instead.

**Fixes:**

- Put fewer nodes in each submission: `-Dfornax.vulkan.maxNodesPerSubmit=8` (or `GGML_VK_MAX_NODES_PER_SUBMIT=8`).
  That fixed the LTX audio decode.
- Decode in tiles: `vae_tiling.enabled` in a `fornax-sd` request.
- For ACE-Step, see [below](#ace-step-on-a-power-limited-gpu).

**When that isn't enough:** a 13 s LTX clip (313 frames) refined at 1344×896 still lost the device on Vulkan, on its
first refine step. One attention dispatch over about 47,000 tokens outlasts the timeout, and no submission size can
split a single node. Use the HIP backend for long clips: ROCm work goes through the kernel driver's user queues, which
have no such timeout. HIP was also about 2.7× faster per LTX step than Vulkan on the 8060S (44 s against 119 s at
672×448×313, after a slower first step).

## ACE-Step on a power-limited GPU

acestep.cpp's default VAE decode tile of 1024 frames hung the GPU on battery: the same compute timeout. Fornax's layer
over it decodes in 512-frame tiles instead. If that still hangs, go lower:

```sh
FORNAX_ACESTEP_VAE_CHUNK=256 ./gradlew song --args='song.wav'
```

## bf16 checkpoints on AMD Vulkan (RADV)

**What you see:** a run aborts on the CPU backend, unable to add f32 and bf16.

**Why:** the Radeon 8060S's Vulkan driver reports no bf16 support (`bf16: 0`), so bf16 operations fall back to the
CPU, which has no f32 + bf16 add. stable-diffusion.cpp never converts `.bias` tensors, so its `tensor_type_rules`
alone couldn't fix it.

**Fix:** from 0.3.0 nothing: stable-diffusion.cpp loads LTX-2.5's bf16 tensors as f32 itself
(leejet/stable-diffusion.cpp#2432, which Fornax's pin includes). On 0.2.0, widen them with
`.set("tensor_type_rules", DiffusionModel.Config.LTX_BF16_TO_F32)`; 0.2.0 carries the patch that lets the rules
reach `.bias` tensors.

## A hires refine that ghosts

**What you see:** a two-stage video (or image) whose refined output has ghosting and ghost structures.

**Why:** stable-diffusion.cpp only runs a learned hires upscaler when `hires.upscaler` is `"model"`. With any other
value, or none, it interpolates the latent (bilinearly, by default) and silently ignores `hires.model_path`.
Refining an interpolated video latent ghosts.

**Fix:** set `hires.upscaler` to `"model"`. Fornax now does this for you when you give `hires.model_path` without an
upscaler, and refuses a model path combined with any other upscaler.

## A crash in libamdhip64 on HIP

**What you see:** on the HIP backend, the JVM segfaults inside `libamdhip64` (seen with LTX, during text encoding).

**Why:** ggml's HIP backend captures its work as a HIP graph, and ROCm's `hip::Graph::ScheduleOneNode` recurses once
per node. A JVM thread's stack is smaller than a command-line tool's 8 MB main thread, and it overflows.

**Fix:** none needed: Fornax sets `GGML_CUDA_DISABLE_GRAPHS=1` by default. If you turned graphs on with
`-Dfornax.gpu.graphs=true`, turn them off again.

## ROCm installed, but `native/build.sh hip` can't find it

Fedora's own ROCm packages live in `/usr`, not `/opt/rocm`. The build script falls back to `/usr/lib64` when
`/opt/rocm/lib` isn't there; for anywhere else, set `ROCM_PATH`.

## LTX-2.5 without sound

LTX-2.5 generates sound and picture together. Leaving out `audio_vae_path` removes the audio stream. That didn't
cause any of the artifacts above, but it isn't how the model was trained to run, so keep it.

## Matching ComfyUI's LTX-2.5 workflow

Fornax's two-stage recipe (`./gradlew video -Precipe=two-stage`) reproduces ComfyUI's "Image to Video (LTX-2.5)"
workflow node by node. The mapping is on the [Video](/docs/guides/video/#the-two-stage-recipe) page.

## Text to speech: a 2^64-byte allocation

If you build on llama.cpp's TTS code yourself: `common_init_from_params()` expects `common_params_parse()` to have
run first, because the parser fills in the thread counts afterwards (`postprocess_cpu_params`). Without that, the
thread pool is sized from -1 and ggml tries to allocate 2^64 bytes. Fornax's layer calls `postprocess_cpu_params`
itself, so `fornax-tts` doesn't hit this. The research behind `fornax-tts` (which models run, their voices and licences)
is in [`docs/tts-notes.md`](https://github.com/liberation-data/fornax/blob/main/docs/tts-notes.md).
