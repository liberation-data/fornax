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
  `GGML_VK_MAX_NODES_PER_SUBMIT=8` (`-Dkiln.vulkan.maxNodesPerSubmit=8`) fixed it. ComfyUI never hit this, because PyTorch on ROCm launches many small
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
- **Cancel only reaches a run that has started.** `sd_cancel_generation` is thread-safe, but an idle context ignores
  it, so a cancel that lands just before `sd_generate_image` begins is lost. Kiln asks again from the progress callback.
  A cancelled run returns `SD_STATUS_CANCELLED` (10) and no images.
- **The progress callback also reports loading.** sd.cpp loads weights on the first run after `sd_ctx_create`, and the
  process-wide progress callback counts the tensors it loads (hundreds of "steps") before the sampling steps. Loading
  doesn't check for a cancel; sampling does, at every step.
- **LTX audio is part of the model.** LTX-2.5 generates sound and picture together. Leaving out
  `audio_vae_path` removes the audio stream; that didn't cause the ghosting, but it isn't how the model was trained.

## acestep.cpp

- **It can stop mid-run.** At the pinned commit, `ace_lm_generate`, `ace_synth_job_run_dit` and
  `ace_synth_job_run_vae` take a `bool (*cancel)(void *)` polled between LM tokens, DiT steps and VAE tiles. The shim
  passes one that reads a flag `kiln_acestep_set_cancel` sets. yue2.cpp's `pipeline_generate` takes the same kind of
  callback (AR tokens, NAR steps, VAE tiles), and its shim does the same.
- **Decode in 512-frame VAE tiles.** The default 1024-frame tiles hung the GPU on battery (the same compute
  timeout). The shim defaults to 512 and reads `KILN_ACESTEP_VAE_CHUNK`.

## whisper.cpp (kiln-whisper)

- **DTW needs flash attention off.** whisper.cpp quietly drops DTW token timestamps when flash attention is on (its
  default), so the shim turns flash attention off whenever an alignment preset is given. On 55 s of narration with
  large-v3-turbo on Vulkan, DTW made a transcription about 50% slower (1.8 s against 1.2 s, beam 5).
- **The new-segment callback is unreliable with DTW on.** Its DTW branch passes a segment index as `n_new`, and its
  loop bound skips segments after the first window. The shim counts the segments it has reported itself and hands
  over the rest when `whisper_full` returns.
- **Word times drift at pauses.** In running speech, DTW puts a word within a frame or two (20 ms), but a word after
  a pause started 0.06-0.27 s late, and the word before a pause ran on to the next word's start. `Words.snapToVoice`
  moves word edges onto the voice where there is silence to go by: sentence starts went from 0.16 s to 0.03 s off on
  average, ends from 0.76 s to 0.08 s (synthesized narration with known gaps). Whisper's timestamp tokens (no DTW)
  were worse: starts up to 0.5 s early or late, and ends that ran into the next sentence.
- **large-v3-turbo's segment times are coarse**, often whole seconds. Kiln fits each segment to its words.
- **SPIRV-Headers.** whisper.cpp v1.9.6's ggml finds SPIRV-Headers but doesn't link the target into ggml-vulkan, so the
  headers must be on the compiler's include path. Fedora's `spirv-headers-devel` puts them in `/usr/include`; for a
  private install, pass `CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-I<prefix>/include"`.

## ComfyUI parity for LTX-2.5 (Endless Country episode 2)

| ComfyUI | Kiln (`VideoClip -Drecipe=two-stage`) |
|---|---|
| Stage 1: half size, `ManualSigmas` 1.0 … 0.421875, 0, euler_ancestral, CFG 1 | `sample.sigmas`, `sample.method=euler_a` |
| `LTXVImgToVideoInplace` strength 0.7 | `strength=0.7` |
| `LTXVPreprocess` 18 | ffmpeg H.264 round trip at CRF 18 (no visible effect in tests) |
| `LTXVLatentUpsampler` with the x2 model | `hires.upscaler=model` plus `hires.model_path` |
| Stage 2: sigmas 0.85, 0.725, 0.4219, 0, first frame at 1.0 | `hires.sigmas`; sd.cpp holds the first frame fully in the refine |
| Tiled VAE decode | `vae_tiling.enabled` |

## llama.cpp text to speech (kiln-tts)

- **Kokoro streams one chunk.** It isn't autoregressive: in stream mode the whole text decodes in one step and arrives
  as a single chunk, so cancel and backpressure have nothing to act on. The autoregressive models (ZONOS3, LFM3-Audio,
  ...) stream frame by frame.

- **Call `postprocess_cpu_params` before `common_init_from_params`.** The parser fills in the thread counts after
  the arguments; without it the threadpool is sized from -1 and ggml tries to allocate 2^64 bytes. See
  [tts-notes.md](tts-notes.md) for the rest (models, voices, licences).

## llama.cpp language models (kiln-llama)

- **`chatJson`'s stream callback only fires on content.** Reasoning and tool-call tokens add no content, so a callback
  that only sees content can't stop them. The shim calls it with an empty piece for those tokens, which LlamaModel
  doesn't pass on but uses to check for a cancel.
- **Upcalls must not throw.** An exception that escapes an FFM upcall can't unwind through C++ frames: the JVM prints
  "Unrecoverable uncaught exception encountered" and exits. Every Kiln upcall catches it and stops the engine instead.

## llama.cpp vision (kiln-llama with an mmproj)

- **One path for every chat with images.** The shim converts image parts to media markers, renders the chat with the
  model's own Jinja template (`common_chat_session`, as llama-server does), splits the rendered prompt on the
  markers, and tokenizes text and images together with `mtmd_tokenize_from_parts`. Template text is parsed for
  special tokens and user text isn't, as in the text-only path. `mtmd_helper_eval_chunks` then runs the vision encoder
  and decodes everything; after that, generation is the usual loop. So `json_schema` grammars and tool calls work with
  images for free.
- **The marker is private.** The shim uses a random marker per process rather than `mtmd_default_marker()`
  (`<__media__>`), so user text that happens to contain `<__media__>` can't claim an image. Raw `complete` with images
  does use the default marker, since the caller writes the whole prompt. The shim counts those markers itself:
  `mtmd_tokenize` returns 2 for a marker/image mismatch (its header says 1), the same code as a failed image.
- **Text tokens after an image need only one position.** Qwen3-VL uses M-RoPE; `mtmd_helper_eval_chunks` returns the
  next position (fewer positions than tokens for an image), and text tokens after it take a single position each
  through `llama_batch_ext_set_pos`. So the shim counts context cells apart from positions: a ~1200-token image
  takes ~40 positions, and checking the position against `n_ctx` would overrun the KV cache.
- **Non-causal image attention needs the whole image in one ubatch.** Gemma 3 (MedGemma) attends over an image
  non-causally. Like `llama-mtmd-cli`, `kiln_llama_create_mm` caps the image's tokens at `n_ubatch` (512 by default)
  for such models, using `mtmd_get_memory_usage` (an unstable mtmd API that may move).
- **PaddleOCR-VL takes task prompts, not instructions.** It answers "OCR:", "Table Recognition:", "Chart
  Recognition:" and so on, and ignores a request to fill a schema: a grammar-constrained reply comes back as
  well-formed JSON with the wrong values. Read the text with PaddleOCR-VL, then extract with a text model
  (`./gradlew invoice -Pmodel=paddleocr-vl`). Its OCR of a 1000×1300 invoice took ~1.5 s at ~200 tokens/s on the
  8060S.
- **Qwen3-VL 4B grounds what is there, not what is missing.** Asked for defects on a tray of parts, it boxed a dented
  part accurately (with a schema that asks for a short description before each box) but never found an empty slot
  in the grid, with or without a schema. Its boxes are on a 0 to 1000 grid over the image.
- **Images are cheap on the GPU.** A 1000×1300 invoice is ~1200 Qwen3-VL tokens and is read (encoded and prefilled)
  in ~1.6 s; a 720×480 picture in ~0.4 s.
