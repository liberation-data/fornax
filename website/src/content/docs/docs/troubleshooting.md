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

**When that isn't enough:** a long LTX video can still lose the device; see the next section.

## Device lost or ring timeout on long video

**What you see:** a long LTX clip on Vulkan (a 313-frame clip refined at 1344×896, for example) loses the device on
its first refine step, and `journalctl -k` shows `amdgpu ... ring comp_... timeout` followed by a GPU reset. A second
reset in a row can take the display down with it.

**Why:** at that size the refine works on 47,040 video tokens, and some single GPU dispatches run for seconds: the
video self-attention (one flash-attention node, about 3.6e13 FLOP, 4.5 to 7.6 s on the Radeon 8060S) and each of
LTX's int8 feed-forward matmuls (about 2.7 s). amdgpu kills any job that runs longer than its `lockup_timeout` (2 s for
compute on this kernel). Fewer nodes per submission can't help: one node is already the smallest unit.

**The fix:** Fornax's stable-diffusion.cpp carries a patch (`native/patches/stable-diffusion.cpp/0003`) that splits
long work into chunks of rows, each its own node: attention by query rows, linear layers and feed-forwards by token
rows. Rows are independent, so the output is the same (test clips came out bit-identical with and without it). Two
environment variables set the size of one chunk in FLOPs, and `0` turns that split off:

- `SD_FA_CHUNK_FLOPS` for attention: the default, 1e12, is 0.12 to 0.21 s on the 8060S.
- `SD_FF_CHUNK_FLOPS` for each matmul of a linear layer: the default, 4e11, is about 0.2 s for LTX's int8
  feed-forward on the 8060S (0.27 s at worst).

Both leave several times the margin to 2 s; lower them for a slower GPU. Keep `-Dfornax.vulkan.maxNodesPerSubmit` low
too (1 or 2 for long video), since a submission of several chunks is still one job. HIP has no such timeout and is
also faster (see [the video guide](/docs/guides/video/#long-clips)).

**Fallback: raise the timeout.** amdgpu takes the timeout as a kernel parameter, in milliseconds, for the graphics,
compute, SDMA and video queues in that order. On Fedora:

```sh
sudo grubby --update-kernel=ALL --args="amdgpu.lockup_timeout=10000,60000,10000,10000"
```

Reboot for it to take effect. This gives compute work 60 s. The cost: the watchdog is how the kernel recovers from a
GPU that really has hung, so a hung job now freezes the GPU (and, on an APU like the 8060S, the display) for up to a
minute before the reset, and a desktop on the same GPU can stall while a long dispatch runs. Remove it with
`sudo grubby --update-kernel=ALL --remove-args="amdgpu.lockup_timeout"`.

## ACE-Step on a power-limited GPU

acestep.cpp's default VAE decode tile of 1024 frames hung the GPU on battery: the same compute timeout. Fornax's layer
over it decodes in 512-frame tiles instead. If that still hangs, go lower:

```sh
FORNAX_ACESTEP_VAE_CHUNK=256 ./gradlew song --args='song.wav'
```

## bf16 checkpoints on AMD Vulkan (RADV)

**What you see:** a run aborts on the CPU backend, unable to add f32 and bf16.

**Why:** the Radeon 8060S's Vulkan driver reports no bf16 support (`bf16: 0`), so bf16 operations fall back to the
CPU, which has no f32 + bf16 add. In LTX-2.5 the tensor at fault was `keyframes_abs_pos_embedding`, which
stable-diffusion.cpp's `tensor_type_rules` skip (they never convert names containing `embedding`), so rules alone
couldn't fix it.

**Fix:** from 0.3.0 nothing: stable-diffusion.cpp loads that tensor as f32 itself
(leejet/stable-diffusion.cpp#2432, which Fornax's pin includes). On 0.2.0, `Config.builder().model("ltx-…")` sets
`DiffusionModel.Config.LTX_BF16_TO_F32`; loading file by file, set it with `.set("tensor_type_rules", …)`. 0.2.0
carries the patch that lets those rules reach any tensor.

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
