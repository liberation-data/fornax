# Engine notes

What we learned getting the engines to behave, so the next person doesn't relearn it. Add to it when an engine
surprises you.

## How to debug a model that looks wrong

1. **Compare against the engine's own tool first.** Before you guess at causes, run the same settings through the
   engine's CLI (`sd-cli` for stable-diffusion.cpp). If the CLI is right and Kiln is wrong, it's Kiln.
   `scripts/compare-with-sdcli.sh` diffs the program stable-diffusion.cpp compiles for each side, in seconds and
   without rendering.
2. **Render small first.** 336×224 and 49 frames took about 3 minutes and showed every artifact the
   1344×896 renders did (those took 10 to 30 minutes).
3. **Split a pipeline into its stages.** For LTX, a draft-only render (`-Dhires=false`) showed that the draft was
   clean and the refine wasn't.

## stable-diffusion.cpp

- **The hires upscaler needs `hires.upscaler = "model"`.** Without it, the latent is interpolated (bilinear by
  default) and `hires.model_path` is silently ignored. Refining an interpolated video latent produces ghosting and
  ghost structures. `DiffusionModel` now selects the model when a model path is given, and refuses contradictory
  settings.
- **bf16 on RADV.** The Radeon 8060S (Strix Halo) Vulkan driver reports `bf16: 0`, so bf16 ops fall back to the
  CPU, which has no f32 + bf16 add, and the run aborts. stable-diffusion.cpp never converts `.bias` tensors, so
  `tensor_type_rules` alone couldn't fix it. `native/patches/stable-diffusion.cpp/0001` lets an explicit bf16→f32
  rule widen any tensor.
- **Long GPU submissions hit the amdgpu compute timeout.** ggml-vulkan puts up to 100 graph nodes in each
  submission. One big decode (LTX's audio VAE at full size) can exceed the kernel's timeout and lose the device.
  `GGML_VK_MAX_NODES_PER_SUBMIT=8` fixed it. ComfyUI never hit this, because PyTorch on ROCm launches many small
  kernels.
- **Long video on Vulkan hits the timeout even at 8 nodes per submission.** A 13 s LTX clip (313 frames) refined at
  1344×896 lost the device on the first refine step: a single attention dispatch over ~47k tokens outlasts the
  amdgpu compute ring timeout, and no submission size can split one node. Use the HIP backend for long clips: ROCm
  work goes through KFD user queues, which have no such timeout (that is why ComfyUI never hit it). HIP was also
  about 2.7× faster per LTX step than Vulkan on the 8060S (44 s against 119 s at 672×448×313, after a slower first step).
- **HIP graphs overflow a JVM thread's stack.** ggml's HIP backend captures the graph as a HIP graph, and ROCm's
  `hip::Graph::ScheduleOneNode` recurses once per node. On a JVM thread (smaller stack than a CLI's 8 MB main
  thread) LTX segfaults in libamdhip64 during text encoding. `GGML_CUDA_DISABLE_GRAPHS=1` fixes it, and Kiln sets it by default (`-Dkiln.gpu.graphs=true` turns graphs back on).
- **Fedora's ROCm lives in `/usr`**, not `/opt/rocm`; `native/build.sh hip` falls back to `/usr/lib64`.
- **LTX audio is part of the model.** LTX-2.5 generates sound and picture together. Leaving out
  `audio_vae_path` removes the audio stream; that didn't cause the ghosting, but it isn't how the model was trained.

## acestep.cpp

- **Decode in 512-frame VAE tiles.** The default 1024-frame tiles hung the GPU on battery (the same compute
  timeout). The shim defaults to 512 and reads `KILN_ACESTEP_VAE_CHUNK`.

## ComfyUI parity for LTX-2.5 (Endless Country episode 2)

| ComfyUI | Kiln (`VideoClip -Drecipe=two-stage`) |
|---|---|
| Stage 1: half size, `ManualSigmas` 1.0 … 0.421875, 0, euler_ancestral, CFG 1 | `sample.sigmas`, `sample.method=euler_a` |
| `LTXVImgToVideoInplace` strength 0.7 | `strength=0.7` |
| `LTXVPreprocess` 18 | ffmpeg H.264 round trip at CRF 18 (no visible effect in tests) |
| `LTXVLatentUpsampler` with the x2 model | `hires.upscaler=model` plus `hires.model_path` |
| Stage 2: sigmas 0.85, 0.725, 0.4219, 0, first frame at 1.0 | `hires.sigmas`; sd.cpp holds the first frame fully in the refine |
| Tiled VAE decode | `vae_tiling.enabled` |
