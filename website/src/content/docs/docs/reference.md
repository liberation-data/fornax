---
title: API overview
description: The public classes of each Kiln module, what they're for, and how to build the Javadoc.
---

Kiln's public API is small: one model class per engine, plus a few shared helpers. This page lists it by module.
For every method and parameter, build the Javadoc from a clone:

```sh
./gradlew javadoc        # one site per module, in <module>/build/docs/javadoc/index.html
```

All packages are under `org.liberationdata.kiln`. Every model class is `AutoCloseable` (its weights are freed on
`close()`, which is idempotent). A model can be called from any thread, one call at a time; the `...Async` methods run
on its own thread. See [Threading](/docs/guides/threading/).

## kiln-core

Package `org.liberationdata.kiln`. The engine modules depend on it.

| Type | |
|---|---|
| `HfCache` | finds files in the Hugging Face cache: `root()`, `find(repo, path)`, `require(repo, path)`, `requireAny(repo, path, ...)` |
| `ModelPool` | keeps models loaded within a memory budget: `acquire(key, sizeBytes, loader)` returns a `Lease`; `unload`, `unloadAll`, `entries`, `loadedBytes`, `sizeOf(Path...)`. See [ModelPool and memory](/docs/guides/model-pool/) |
| `ModelPool.Lease<M>` | a borrowed model: `model()`; `close()` gives it back |
| `ModelPool.Entry` | a snapshot of one pooled model: `key`, `sizeBytes`, `loaded`, `inUse`, `lastUsed` |
| `Audio` | stereo float PCM in [-1, 1]: `left`, `right`, `sampleRate`, `seconds()`, `writeWav(File)` (16-bit), `readWav(File)`, `ofMono`, `mono()`, `resample(rate)` |
| `NativeLibrary` | finds and binds a native library, and sets the engines' tunables before the first one loads. You won't usually call it |
| `ModelThread` | what each model uses to run its calls one at a time, on the caller's thread or its own, and to cancel and close them. You won't usually call it |
| `Call` | one call into an engine: its stop flag and the first exception a callback threw. For engine bindings |
| `StreamOverflowException` | what a streaming subscriber gets when it falls a whole buffer behind, and the generation stops |

## kiln-llama

Package `org.liberationdata.kiln.llama`. See [Language models](/docs/guides/language-models/).

| Type | |
|---|---|
| `LlamaModel` | a GGUF language model: `new LlamaModel(gguf)` or `(gguf, gpuLayers, contextSize)` |
| `chat(messages, options[, onPiece])` | a chat reply, optionally streamed; returns a `Result` |
| `chatJson(requestJson, onContent)` | an OpenAI-format request and reply, with tool calls and `json_schema` |
| `complete(prompt, options, onPiece)` | raw completion of a prompt you've formatted |
| `cancel()` | stops the call in progress, from any thread |
| `chatAsync(messages, options[, subscriber])`, `chatStream(messages, options)` | `chat` on the model's own thread: a `CompletableFuture<Result>`, optionally streaming to a `Flow.Subscriber`; or a cold `Flow.Publisher<String>` |
| `chatJsonAsync(requestJson[, images[, subscriber]])`, `extractAsync(...)`, `completeAsync(prompt, images, options)` | the same for `chatJson`, `extract` and `complete` |
| `streamOn(executor, bufferSize)` | where streamed pieces are delivered and how far a subscriber may fall behind (default: virtual threads, 1024) |
| `new LlamaModel(Weights)`, `(gguf, mmproj)` | a vision language model with its mmproj; see [Vision](/docs/guides/vision/). `supportsVision()` |
| `extract(messages, jsonSchema, options)` | a reply constrained to a JSON schema, as JSON text; messages may carry images |
| `chatJson(requestJson, images, onContent)`, `complete(prompt, images, options, onPiece)` | `chatJson` and `complete` with images; `mediaMarker()` |
| `LlamaModel.Message` | `role`, `content` and `images`; `system(...)`, `user(...)`, `user(text, images...)`, `assistant(...)` |
| `LlamaModel.Image` | packed RGB: `of(BufferedImage)`, `read(Path)`, `decode(bytes)` |
| `LlamaModel.Weights` | the models Kiln knows by name: `QWEN3_4B`, `QWEN3_VL_4B`, `PADDLEOCR_VL`, `MEDGEMMA_4B`; `named(...)`, `vision()`, `model()`, `mmproj()`, `downloaded()`, `downloadCommand()` |
| `Json` | just enough JSON to read `chatJson` replies: `parse`, `get(doc, path...)`, `quote` |
| `LlamaModel.Options` | `maxTokens`, `temperature`, `topP`, `minP`, `seed`; `DEFAULT`, `withSeed`, `withMaxTokens`, `withTemperature` |
| `LlamaModel.Result` | `text`, `promptTokens`, `generatedTokens` |

## kiln-tts

Package `org.liberationdata.kiln.tts`. See [Speech](/docs/guides/speech/).

| Type | |
|---|---|
| `TtsModel` | a TTS model: `new TtsModel(Weights)`, `(model, mmproj)` or `(model, mmproj, gpuLayers, contextSize)` |
| `speak(text, voice)`, `(text, voice, onChunk)`, `(text, voice, options, onChunk)` | speech as `Audio`, optionally streamed in chunks; a voice of `@clip.wav` speaks like that clip |
| `voices()`, `sampleRate()`, `cancel()` | the built-in voices (the default first), the output rate, and stopping a streaming call |
| `speakAsync(text, voice, options[, subscriber])`, `speakStream(text, voice, options)` | `speak` on the model's own thread, optionally streaming chunks to a `Flow.Subscriber`; `streamOn(executor, bufferSize)` |
| `TtsModel.Weights` | the models Kiln knows by name: `KOKORO`, `ZONOS3`, `LFM3_AUDIO`, `TINY_VOCA`, `CHATTERBOX_TURBO`, `SOPRANO`, `VOXCPM2`, `QWEN3_TTS`; `named(...)`, `vision()`, `model()`, `mmproj()`, `downloaded()`, `downloadCommand()` |
| `TtsModel.Options` | `lang`, `temperature`, `seed`, `speed`, `maxFrames`; `DEFAULT` and a `with...` for each |

## kiln-sd

Package `org.liberationdata.kiln.sd`. See [Images](/docs/guides/images/) and [Video](/docs/guides/video/).

| Type | |
|---|---|
| `DiffusionModel` | a stable-diffusion.cpp model: `new DiffusionModel(Config)` |
| `generate(Request)` | images, as `List<BufferedImage>` |
| `generateVideo(VideoRequest)` | a `Video` |
| `generateAsync(Request)`, `generateVideoAsync(VideoRequest)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops a run at the next step (it throws `CancellationException`) |
| `modelVersion()` | the model stable-diffusion.cpp detected |
| `onProgress(sink)`, `onLog(sink)` | process-wide progress (step, steps, seconds per step) and log callbacks |
| `DiffusionModel.Config` | `builder()`: `diffusionModel`, `vae`, `llm`, and `set(key, value)` for any context option |
| `DiffusionModel.Request` | `prompt`, `width`, `height`, `steps`, `cfg`, `seed`, plus `refImages` and an `extra` options map |
| `DiffusionModel.VideoRequest` | the same, plus `frames` and `fps`; `withInitImage`, `withEndImage`, `withRefAudio`, `withExtra` |
| `DiffusionModel.Video` | `frames`, `fps`, `audio` (or `null`), `seconds()` |
| `Sd` | the raw bindings to stable-diffusion.cpp's C API; `Sd.SdException` is what a failed call throws |

Option keys are stable-diffusion.cpp's, from its
[C API guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/c_api.md).

## kiln-acestep

Package `org.liberationdata.kiln.acestep`. See [Songs](/docs/guides/songs/).

| Type | |
|---|---|
| `AceStepModel` | ACE-Step 1.5: `new AceStepModel(lm, textEncoder, dit, vae, keepLoaded)`; `lm` may be `null` |
| `generate(requestJson)` | a song from acestep.cpp's JSON request; returns a `Result` |
| `generateAsync(requestJson)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between LM tokens, DiT steps and VAE tiles (it throws `CancellationException`) |
| `AceStepModel.Result` | `audio` (48 kHz stereo) and `resolvedRequestJson` |

## kiln-yue2

Package `org.liberationdata.kiln.yue2`. See [Songs](/docs/guides/songs/#yue2).

| Type | |
|---|---|
| `Yue2Model` | YuE2: `new Yue2Model(model, vae, keepLoaded, maxSeq)`; `maxSeq` 0 is the full context |
| `generate(requestJson)` | a song from yue2.cpp's JSON request; returns a `Result` |
| `generateAsync(requestJson)`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between LM tokens, diffusion steps and VAE tiles (it throws `CancellationException`) |
| `Yue2Model.Result` | `audio` (48 kHz stereo), `resolvedRequestJson` and `abcScore` |

## kiln-whisper

Package `org.liberationdata.kiln.whisper`. See [Speech to text](/docs/guides/speech-to-text/).

| Type | |
|---|---|
| `WhisperModel` | a Whisper model: `new WhisperModel(Weights)`, `(path)`, `(path, alignment)` or `(path, alignment, gpu)` |
| `transcribe(audio[, options][, listener])` | a `Transcript` from `Audio` at any rate, or from 16 kHz mono `float[]` |
| `multilingual()`, `type()`, `cancel()` | English-only or not, the model size, and stopping a call from another thread |
| `WhisperModel.Weights` | the models Kiln knows by name: `LARGE_V3_TURBO`, `LARGE_V3_TURBO_Q8_0`, `BASE_EN`; `named(...)`, `path()`, `downloaded()`, `alignment` |
| `WhisperModel.Alignment` | the DTW alignment heads for a model size (`LARGE_V3_TURBO`, `BASE_EN`, ...), or `NONE` |
| `WhisperModel.Options` | `language`, `translate`, `initialPrompt`, `threads`, `beamSize`, `temperature`; `DEFAULT` and a `with...` for each |
| `WhisperModel.Listener` | `progress(percent)` and `segment(segment)` as the transcription runs |
| `Transcript` | `text`, `language`, `segments`; `words()`, `wordsTsv()`, `srt()`, `vtt()` |
| `Transcript.Segment`, `Transcript.Word` | `start`, `end` (seconds), `text`; a segment's `words`, a word's `probability` |

## kiln-demucs

Package `org.liberationdata.kiln.demucs`. See [Stem separation](/docs/guides/stem-separation/).

| Type | |
|---|---|
| `DemucsModel` | an HTDemucs model: `new DemucsModel(Weights)`, `(path)` or `(path, gpu, cpuThreads)` |
| `separate(audio[, options][, progress])` | `Stems` from `Audio` at any rate, at its rate and length |
| `separateAsync(audio[, options, progress])`, `cancel()` | the same on the model's own thread, as a `CompletableFuture`; `cancel()` stops it between segments (it throws `CancellationException`) |
| `name()`, `backend()`, `sources()`, `sampleRate()` | the checkpoint, the ggml backend, the stem names in order, and the model's own rate (44100) |
| `DemucsModel.Weights` | the models Kiln knows by name: `HTDEMUCS`, `HTDEMUCS_F16`, `HTDEMUCS_6S`, `HTDEMUCS_FT`; `named(...)`, `path()`, `downloaded()` |
| `DemucsModel.Options` | `shifts`, `overlap`, `seed`; `DEFAULT` and a `with...` for each |
| `DemucsModel.Progress` | `update(done, total)`, segment passes, as the separation runs |
| `Stems` | `names()`, `get(name)`, `vocals()`, `backing()` (every stem but the vocals), `mix()`, `asMap()`, `sampleRate()` |

## kiln-examples

Package `org.liberationdata.kiln.examples`: runnable programs, each a Gradle task, and the best place to see the API
used end to end.

| Task | Class | |
|---|---|---|
| `single` | `Single` | an LLM writes a song, ACE-Step sings it, FLUX.2 [klein] paints the cover |
| `chat` | `Chat` | streams a chat reply |
| `speak`, `speakBatch` | `Speak`, `SpeakBatch` | text to speech, one line or many at one seed |
| `image` | `Image` | text to image or image editing: `klein`, `qwen-image`, `qwen-edit` |
| `video` | `VideoClip` | an LTX-2.5 clip with sound, one- or two-stage |
| `song` | `Song` | an ACE-Step song |
| `yue2` | `Yue2Song` | a YuE2 song and its score |
| `transcribe` | `Transcribe` | speech to text: prints the text, writes `<name>.words.tsv` and `<name>.srt` |
| `separate` | `Separate` | stem separation: writes `<name>-vocals.wav` and `<name>-backing.wav` (`--all` for every stem) |

## System properties and environment

| Name | |
|---|---|
| `-Dkiln.library.path`, `KILN_LIBRARY_PATH` | where to look for the native libraries |
| `-Dkiln.<name>.lib` | one library's exact file; the names are `kiln_llama`, `kiln_tts`, `stable-diffusion`, `kiln_acestep`, `kiln_yue2`, `kiln_whisper` and `kiln_demucs` |
| `-Dkiln.backend`, `KILN_BACKEND` | which natives jar's backend to load when several for this platform are on the classpath |
| `-Dkiln.cache.dir`, `KILN_CACHE_DIR` | where natives jars' libraries are extracted to |
| `-Dkiln.gpu.graphs` | ggml's CUDA and HIP graphs; `false` by default |
| `-Dkiln.vulkan.maxNodesPerSubmit` | graph nodes per Vulkan submission |
| `KILN_VERBOSE` | the engines' own logs; `debug` for everything from stable-diffusion.cpp |
| `KILN_ACESTEP_VAE_CHUNK` | ACE-Step's VAE decode tile, in frames (512) |
| `HF_HUB_CACHE`, `HF_HOME` | where `HfCache` looks |
