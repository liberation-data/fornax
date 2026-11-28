---
title: "jitllm: language models in Java"
description: Run GGUF language and embedding models on the GPU through jitllm, a Java inference engine that TornadoVM compiles for CUDA, OpenCL or Metal, as a second engine next to llama.cpp.
---

[jitllm](https://github.com/beehive-lab/jitllm) is an inference engine written in Java. Its GPU kernels are Java code
that [TornadoVM](https://github.com/beehive-lab/TornadoVM) compiles at run time for CUDA, OpenCL or Metal, so the same
jar runs on NVIDIA, AMD, Intel and Apple GPUs with no native build. Fornax offers it as a second engine for language
and embedding models, next to llama.cpp: `fornax-jitllm`'s `JitLlmModel` takes the same `Message`s and `Options` as
`LlamaModel` and gives the same `Result`, and `fornax chat` and `fornax server` take `--engine jitllm`.

## Install TornadoVM

jitllm needs a TornadoVM SDK, 8.2.0 or newer, built for JDK 22 or newer (a `-jdk22plus` SDK) and for your GPU's
backend: CUDA for NVIDIA, OpenCL for AMD, Intel or NVIDIA, Metal for Apple silicon. Take one from the
[TornadoVM downloads](https://www.tornadovm.org/downloads) or from SDKMAN!:

```sh
sdk install tornadovm
export TORNADOVM_HOME=~/.sdkman/candidates/tornadovm/current
$TORNADOVM_HOME/bin/tornado --devices      # your GPU should be listed
```

Fornax finds the SDK through `TORNADOVM_HOME` (or `-Dfornax.tornadovm.home`). Without one, a jitllm command says so
before it downloads anything.

## From Java

<!-- snippet: Readme#jitllm -->
```java
try (var llm = new JitLlmModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")) {   // same messages, options and result as LlamaModel
    llm.chat(List.of(Message.user("Name a country song about a robot.")), Options.DEFAULT, System.out::print);
}
```

`new JitLlmModel(model, contextSize, JitLlmModel.Mode.EMBED)` loads an embedding model instead, with `embed` and
`embeddings` as on `LlamaModel`.

TornadoVM needs JVM options of its own. A JVM started with the SDK's `tornado-argfile`,
`--add-modules jdk.incubator.vector`, `-Duse.tornadovm=true` and `-Dtornado.tvm.maxbytecodesize=65536` runs the model
in-process. Any other JVM, such as a plain `java` or `./gradlew run`, runs it in a
[worker process](/docs/guides/isolation/) that Fornax starts with those options. A worker also keeps a crash in the
GPU driver from taking your app down with it.

TornadoVM's device memory budget is 14 GB, as for the `jitllm` command. Set `-Dtornado.device.memory=8GB` (or another
size) on the JVM that runs Fornax to change it; a worker takes it from there.

## From the command line

```sh
fornax chat qwen3-4b "Write a haiku about a kiln" --engine jitllm
fornax server -m qwen3-4b -m qwen3-embedding-0.6b --engine jitllm
```

`fornax server --engine jitllm` serves chat completions (streaming too) and embeddings over the
[OpenAI-compatible API](/docs/guides/openai-server/). `FornaxServer.builder().languageEngine(LanguageEngine.JITLLM)`
does the same from Java. Every other kind of model the server runs (speech, images and so on) stays on its own engine.

A reasoning model's thinking comes back as `reasoning_content`, as from llama.cpp, and `enable_thinking: false` turns
it off for a model whose template allows that (Qwen3).

## What it doesn't cover

jitllm runs text in and text out. These stay with llama.cpp, and the server answers a request that needs them with a
400:

- tool calls (`tools`, unless `tool_choice` is `"none"`)
- JSON schemas, grammars and `response_format` other than text
- images and vision models
- reranking
- `--parallel` above 1: a jitllm model answers one request at a time
- `--gpu-layers`: jitllm runs every layer on the GPU

It samples with temperature and top-p: `min_p` and `top_k` are ignored.

An embedding model's context is at most 2048 tokens (less if `--ctx` is smaller). jitllm reserves its memory for the
whole context when the model loads, so it can't grow to fit a long text as llama.cpp does; a text longer than the
context is refused with a 400.

## Speed

Decode speed on an NVIDIA A10 with CUDA, tokens per second (higher is better), against llama.cpp on the same GPU:

| Model | llama.cpp | jitllm, worker | jitllm, in-process |
|---|---|---|---|
| Llama-3.2-1B Q8_0 | 297.6 | 286.5 | 289.6 |
| Qwen3-0.6B Q8_0 | 402.5 | 370.8 | 356.1 |

On the same GPU through OpenCL, jitllm decodes 2.8 to 4.6 times slower than through CUDA.
