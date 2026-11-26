---
title: ModelPool and memory
description: Keep models loaded between uses within a memory budget, share them safely between threads, and unload them when they sit idle.
---

Loading a model means reading gigabytes of weights onto the GPU, so Kiln's models keep their weights loaded until
you `close()` them. Load each model once and reuse it.

An app with several models (a chat model, an image model, a song model) can't always keep all of them loaded at once.
`ModelPool`, in `kiln-core`, does what ComfyUI does: it keeps models loaded between uses, within a memory budget,
and unloads the ones used least recently when it needs room.

## Borrowing a model

```java
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.ModelPool;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

Path gguf = HfCache.require("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");

try (var pool = new ModelPool(96L << 30, Duration.ofMinutes(10))) {   // a 96 GB budget; unload after 10 idle minutes
    try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf), () -> new LlamaModel(gguf.toString()))) {
        lease.model().chat(List.of(Message.user("Hello!")), Options.DEFAULT, System.out::print);
    }
    // The model is still loaded: the next acquire("qwen3-4b", ...) gets it at once.
}
```

`acquire(key, sizeBytes, loader)` returns a `Lease`. The first time a key is asked for, the loader runs; after that,
the loaded model is handed out again. Closing the lease gives the model back to the pool, which keeps it loaded.
Closing the pool unloads everything.

## The rules

- **A model loads on first `acquire` and stays loaded** after its lease closes.
- **Room is made least recently used first.** Loading a model that would take the pool over its budget first unloads
  idle models, the one used longest ago first.
- **Models in use are never unloaded.** If the models in use alone are over the budget, the load goes ahead anyway:
  the budget is a target, not a hard limit.
- **Idle models are unloaded in the background** once they have been idle longer than the timeout. Pass `null` for
  no timeout.
- **One user at a time.** A model has one lease at a time; `acquire` waits until the model is free (on a fair
  semaphore, which a virtual thread waits on without pinning its carrier). Different models can be used from different
  threads at once. From virtual threads, call the model's `...Async` methods and await them before closing the lease:
  see [Threading](/docs/guides/threading/#with-a-modelpool).

## Sizes

The size you pass is your estimate of the memory a model uses. `ModelPool.sizeOf(Path...)` adds up the sizes of its
weight files, which is a reasonable estimate of what loading them takes. Add a margin for large contexts or large images if you need
to.

```java
import org.liberationdata.kiln.sd.DiffusionModel;

Path flux = HfCache.require("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors");
Path vae = HfCache.require("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors");
Path qwen3 = HfCache.require("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors");

try (var lease = pool.acquire("flux2-klein", ModelPool.sizeOf(flux, vae, qwen3), () -> new DiffusionModel(
        DiffusionModel.Config.builder().diffusionModel(flux.toString()).vae(vae.toString()).llm(qwen3.toString()).build()))) {
    var img = lease.model().generate(new DiffusionModel.Request("a kiln at dusk", 768, 768, 12, 4.0, 7)).get(0);
}
```

Any `AutoCloseable` can go in a pool, so the same pool can hold your own resources too.

## Looking inside, and letting go

```java
for (ModelPool.Entry e : pool.entries()) {
    System.out.printf("%s: %d MB, loaded=%s, inUse=%s, last used %s%n",
            e.key(), e.sizeBytes() >> 20, e.loaded(), e.inUse(), e.lastUsed());
}
System.out.println((pool.loadedBytes() >> 20) + " MB loaded");

pool.unload("flux2-klein");   // waits until it is free, then unloads it; false if it wasn't loaded
pool.unloadAll();             // unloads every model that isn't in use
```

## Isolated models

A loader can return an isolated model (`() -> LlamaModel.isolated(path)`), which runs in a worker process of its own.
Add about 100 MB to its size for the worker's JVM; unloading it stops the worker. See
[Isolation](/docs/guides/isolation/#with-a-modelpool).

A server with many callers and several models can use [Dispatch](/docs/guides/dispatch/) instead: a pool of worker
processes with the same LRU rules across all of them, plus model affinity and priority queues.

## Without a pool

For one or two models, plain try-with-resources is enough, and keeps memory use obvious. `./gradlew single` loads
the language model, the song model and the image model one after the other, closing each before the next, so the
three never hold memory at the same time.

Two of the engines also have a setting of their own: ACE-Step and YuE2 take `keepLoaded`. With `false`, they keep at
most one of their parts on the GPU at a time. See [Songs](/docs/guides/songs/).
