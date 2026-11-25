# Kiln

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Call open-weight models (language, vision, speech, images, songs, video) as plain Java methods, running on your GPU, inside your JVM.**

```java
try (var llm = new LlamaModel("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")) {   // weights load onto the GPU once
    llm.chat(List.of(Message.user("Name a country song about a robot.")), Options.DEFAULT, System.out::print);
}
```

**Why:** no overhead between your code and the model.

- **The native runners themselves.** Kiln calls llama.cpp, stable-diffusion.cpp, acestep.cpp and yue2.cpp
  directly. These are the same C/C++ engines the CLIs and servers use, so you get their speed and their model
  support.
- **On the GPU.** Vulkan today (AMD, Intel, NVIDIA). ROCm, CUDA and Metal builds come from the same script.
- **Nothing in between.** A Kiln call is a native function call through Java's Foreign Function & Memory API.
  There is no Python, no model server, no HTTP, no JSON round trip and no second process. Weights stay resident
  between calls, and tokens stream back to a Java callback as they are produced.

Kiln binds native inference engines with Java's Foreign Function & Memory API (Java 22+, built and tested on 25):

| Module | Engine | Models |
|---|---|---|
| `kiln-llama` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | any GGUF language model (Qwen, Llama, Gemma, Mistral, ...) |
| `kiln-tts` | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`mtmd`) | text to speech: Kokoro, ZONOS3, LFM3-Audio, Chatterbox, VoxCPM2, ... |
| `kiln-sd` | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | FLUX.1/FLUX.2, Qwen-Image and Qwen-Image-Edit, SD 1.x/2.x/XL/3, Wan, LTX |
| `kiln-acestep` | [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) | ACE-Step 1.5 songs (vocals, lyrics, up to several minutes) |
| `kiln-yue2` | [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) | YuE2 songs |

Images and songs work the same way:

```java
try (var songs = new AceStepModel(lm, textEncoder, dit, vae, true)) {
    songs.generate(requestJson).audio().writeWav(new File("song.wav"));
}

try (var images = new DiffusionModel(DiffusionModel.Config.builder()
        .diffusionModel(flux).vae(vae).llm(textEncoder).build())) {
    BufferedImage img = images.generate(new DiffusionModel.Request("a neon sign", 1024, 1024, 20, 4.0, 7)).get(0);
}
```

## Quick start

The fastest way to see Kiln work: build the engines, fetch three models, and run the demo. One Java program
has Qwen3 4B write a country song, ACE-Step 1.5 sing it and FLUX.2 [klein] paint its cover, all in one JVM.

```sh
native/build.sh vulkan            # or hip | cuda | metal | cpu (see Building)
hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
hf download Serveurperso/ACE-Step-1.5-GGUF acestep-5Hz-lm-4B-Q8_0.gguf Qwen3-Embedding-0.6B-Q8_0.gguf \
    acestep-v15-turbo-Q8_0.gguf vae-BF16.gguf
hf download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors
hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors
hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors
./gradlew single                  # writes single.wav, cover.png and lyrics.txt
```

To try models without writing code, use [Kiln Studio](https://github.com/liberation-data/kiln-studio).

## Status

Early. Everything below runs from Java on Linux with an AMD Radeon 8060S (Ryzen AI Max+ 395 "Strix Halo",
128 GB unified memory) using the **Vulkan** backend:

| Model | Result |
|---|---|
| Qwen3 4B Instruct (Q4_K_M) | ~60 tokens/s, streamed through a Java callback |
| ACE-Step 1.5 XL turbo | a 3 min 20 s song in 57 s |
| ACE-Step 1.5 turbo | a 60 s song in ~15 s |
| YuE2 3B (Q8_0) | a 60 s song in ~60 s |
| FLUX.2 [klein] base 4B (fp8) | 768×768, 12 steps in ~130 s |
| Qwen-Image 2512 (fp8) | 1024×1024, 20 steps in ~13 min |

The build script also knows the **ROCm (HIP)**, **CUDA** and **Metal** backends of the underlying engines, but
those Kiln builds haven't been tested yet. The Java code is the same for every backend: you pick one when you
build the native libraries. Reports and fixes welcome.

## How it works

Each engine is built into **one self-contained shared library** with its ggml linked in statically and kept
private. The engines pin different ggml versions, so this is what lets several of them share one process.

- `stable-diffusion.cpp` has a key/value C API, so `kiln-sd` binds it directly.
- `llama.cpp` has a C API too, but it passes structs by value and is mid-migration to a new batch API, so
  `kiln-llama` binds a small stable C layer (`kiln-llama/native`) instead.
- `acestep.cpp` and `yue2.cpp` are C++ only, so each gets a ~100-line C layer: JSON request in, PCM audio out.

Engine quirks and how to debug them: [docs/engine-notes.md](docs/engine-notes.md).

Every model class is `AutoCloseable`; weights are freed on `close()`. Instances aren't thread-safe.

`ModelPool` keeps models resident between uses, the way ComfyUI does: it unloads idle models least recently used
first to stay within a memory budget, optionally unloads after an idle timeout, and gives each model one user at a
time.

`LlamaModel.chatJson` does OpenAI-style chat with **tool calling**, using the model's own chat template and
tool-call format (the same templates and parsers as llama-server): messages and tools in OpenAI's format in, a
message with `content`, `reasoning_content` and `tool_calls` out.

## Building

Needs JDK 25, CMake, Ninja, a C++17 compiler, and the SDK for your GPU backend
(Vulkan: the Vulkan headers, `glslc` and SPIRV-Headers; on Fedora `dnf install vulkan-headers vulkan-loader-devel glslc spirv-headers-devel`).

```sh
native/build.sh vulkan            # or hip | cuda | metal | cpu; optionally list engines: llama tts sd acestep yue2
./gradlew build
```

`native/build.sh` clones the engines at the commits pinned in `native/engines.properties` into `engines-src/`
and writes the libraries to `build/native/<os>-<arch>-<backend>/`. The Gradle build points the examples and tests
at that folder; pick another backend with `-Pkiln.backend=hip`.

In your own application, put the libraries on `-Dkiln.library.path=<dir>` (or `KILN_LIBRARY_PATH`, or the
platform loader path) and run with `--enable-native-access=ALL-UNNAMED`. Set `KILN_VERBOSE=1` to see the
engines' own logs.

On a power-limited GPU (for example a laptop on battery), a single long GPU dispatch can exceed the driver's
compute timeout and lose the device. Kiln decodes ACE-Step audio in 512-frame tiles to avoid that; tune it with
`KILN_ACESTEP_VAE_CHUNK`.

## Models

Kiln doesn't download anything. Fetch weights with the Hugging Face CLI, and `HfCache` finds them in the cache
(`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`), or pass file paths directly.

```sh
hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
hf download Serveurperso/ACE-Step-1.5-GGUF acestep-5Hz-lm-4B-Q8_0.gguf Qwen3-Embedding-0.6B-Q8_0.gguf \
    acestep-v15-turbo-Q8_0.gguf vae-BF16.gguf
hf download Serveurperso/YuE2-GGUF YuE2-3B-Q8_0.gguf YuE2-Vae-F32.gguf
hf download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors
hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors
hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors
```

Check each model's license before you use what it makes. Some are non-commercial: YuE2's weights are
CC BY-NC 4.0, for example.

## Examples

```sh
./gradlew single    # three engines, one JVM: an LLM writes a song, ACE-Step sings it, FLUX.2 klein paints the cover
./gradlew chat --args='"Write a haiku about a kiln"'
./gradlew song --args='song.wav'                         # -Pdit=xl-turbo for the larger model
./gradlew yue2 --args='yue2-song.wav'
./gradlew speak --args='out.wav "Hello from the JVM" bm_george'   # -Pmodel=zonos3|lfm3-audio|chatterbox-turbo|...
./gradlew image --args='klein out.png "a neon sign that says KILN"' -Psteps=12 -Pwidth=768 -Pheight=768
```

The song examples use the requests in `kiln-examples/src/main/resources/requests/`; pass your own JSON as a
second argument. The request formats are the engines' own (see
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) and [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)).

## Roadmap

- Test and document the HIP, CUDA and Metal builds; prebuilt native libraries per platform.
- Video through stable-diffusion.cpp (`sd_generate_video`: Wan, LTX).
- `kiln-embabel`: an Embabel Agent `LlmService` and `LocalModelSource` (needs tool-call parsing in `kiln-llama`).
- An optional OpenAI-compatible server.
- Vision-language models through llama.cpp's `mtmd` (document parsing, image Q&A).
- An MLX module for Apple silicon, through `mlx-c`.
- Speech: whisper.cpp.
- A Kotlin layer with coroutine `Flow` streaming.

## License

Apache 2.0. The engines Kiln builds against are MIT-licensed; see [NOTICE](NOTICE).
