---
title: Isolation
description: Run a model in a worker process so that a native crash fails the call instead of killing your JVM, what it costs, and how it works with ModelPool.
---

Fornax calls its engines in-process, through FFM. That is the default and the fast path: no copies, direct GPU access,
the engines' own builds. The catch is that a native crash, such as a failed `GGML_ASSERT`, a segfault, or a malformed
or hostile GGUF, kills the whole JVM, and Java code can't catch it.

Isolation runs a model in a **worker**: a child JVM, launched from your class path, that loads the model through the
normal in-process API. Your code keeps the same `LlamaModel` and the same methods. If the worker crashes, only the
worker dies. The call in flight fails with a `WorkerCrashedException`, and the next call starts a new worker and
loads the model again.

## Turning it on

```java
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaModel;

try (var llm = LlamaModel.isolated(gguf)) {           // the model loads in a worker process of its own
    llm.chat(messages, Options.DEFAULT, System.out::print);
} catch (WorkerCrashedException e) {
    // the engine crashed: e.signal(), e.fatalSignal() ("SIGSEGV"), e.lastOutput() (its last log lines)
}
```

It returns a `LlamaModel`, so everything in [Threading](/docs/guides/threading/) applies unchanged: `chat`,
`chatJson`, `extract` and `complete`; images; the `...Async` methods; `Flow` streaming with its bounded buffer;
`cancel()`, which crosses to the worker and stops the generation there (a chat returns the text so far); cancelling a
future; and `close()`, which also stops the worker.

To isolate every `LlamaModel` without changing code, set `-Dfornax.isolation=process` (or `FORNAX_ISOLATION=process`).
`new LlamaModel(...)` then loads in a worker too. The default is `inprocess`.

`isolated(...)` also takes the constructor's other forms: `isolated(Weights)`, and
`isolated(gguf, mmproj, gpuLayers, contextSize)`. To put several models in one worker, so they share one JVM's
overhead (and one crash), pass a `Worker` you own:

```java
try (var worker = new Worker("chat-models", Worker.Options.DEFAULT)) {
    var small = LlamaModel.isolated(worker, smallGguf, null, -1, 8192);
    var large = LlamaModel.isolated(worker, largeGguf, null, -1, 8192);
    ...
}
```

Isolation covers `LlamaModel` today. The worker itself is generic: other engines need a `WorkerService` each, as
`fornax-llama`'s `LlamaService` is.
Within `LlamaModel` it covers every mode: chat and vision (`Mode.CHAT`), embeddings (`Mode.EMBED`) and reranking
(`Mode.RERANK`). Pass the mode as the constructor takes it: `LlamaModel.isolated(Weights.QWEN3_EMBEDDING_0_6B,
Mode.EMBED)`, `isolated(gguf, gpuLayers, contextSize, Mode.RERANK)`, or `isolated(worker, gguf, gpuLayers, contextSize,
mode)` for a shared worker. Embeddings come back through shared memory once a batch is large, so a big `embed` costs
little more than in-process.

## When to choose it

Choose isolation when a crash would cost more than the overhead:

- **Models you didn't pick.** User-supplied or downloaded GGUFs, where a malformed file can crash the loader.
- **Long-running servers.** One request that trips an engine bug shouldn't take down every other request, or the
  app around them.
- **New or experimental engine builds**, before you trust them.

Stay in-process for desktop apps, CLIs, tests and single-user tools, where a model is an object in your JVM and a
crash costs a restart anyway. Pinned, checksummed weights remove most of the risk from hostile files.

What isolation can't fix: a GPU driver hang or reset. That sits below both processes.

## What it costs

Measured on a Ryzen AI Max+ 395 (Radeon 8060S, Vulkan), with the GPU shared with other work. The numbers come from
`./gradlew :fornax-core:workerBench` and
`./gradlew :fornax-llama:isolationBench --args='Qwen3-0.6B-Q8_0.gguf 256 3'`.

| | In-process | Isolated |
|---|---|---|
| A call that does nothing (an echo service), blocking | 0.1 µs | 12 to 16 µs |
| The same call, async | 3 to 7 µs | 17 to 19 µs |
| Starting a worker (JVM up and connected) | | 70 ms |
| Each streamed item (a token) | 0.08 µs | 0.9 µs |
| An 8 MB payload, there and back | none (no copy) | 3.5 ms (5.9 ms through the socket) |
| Qwen3 0.6B tokens/s, 256 tokens (three rounds) | 248, 224, 226 | 227, 154, 225 |
| Qwen3 0.6B, a 1-token chat | 12 to 37 ms | 13 to 16 ms |
| Loading Qwen3 0.6B | 180 to 340 ms | 390 to 910 ms |

- **Latency per call** is about 15 µs: a round trip over a Unix domain socket. That is noise next to any real
  generation, which takes milliseconds per token.
- **Streaming** adds about a microsecond per token. At 250 tokens a second, a token takes 4 ms.
- **Throughput** is the same within the noise of a shared GPU. The engine runs the same native code on the same GPU.
  Only the process around it changes.
- **Memory per worker** is one more JVM: about 60 MB resident when idle, on top of the model. The model's own
  memory (weights, KV cache, GPU buffers) moves to the worker; it isn't doubled. The worker's heap grows with the
  payloads it handles (to 700 MB after the benchmark's hundred 8 MB round trips), up to the JVM's default maximum of a
  quarter of RAM. Cap it with `Worker.Options.DEFAULT.withJvmArgs("-Xmx1g")` if that matters.
- **A crash costs a reload.** The worker's models are gone with it, so the next call starts a JVM (about 70 ms to
  connect, more to load the engine library) and loads the weights again: well under a second for a small model, but seconds to minutes for tens of gigabytes. A
  model that crashes every time it loads won't make the worker spin: restarts back off, from 250 ms and doubling up to
  30 s, and the backoff resets after a call succeeds.

## How it works

- **The worker** is `java` from your JVM's `java.home`, with the same class path, every `-Dfornax.*` property, the same
  environment (so every `FORNAX_*` variable) and `--enable-native-access=ALL-UNNAMED`. It finds the same natives your JVM
  would. Your JVM doesn't load the engine library at all for an isolated model. Workers start when the model loads.
- **Calls** go over a Unix domain socket in a private temporary directory: the call, each streamed token, the result
  or the exception, and cancels. The worker's exceptions arrive as the same class where they can be rebuilt
  (`IllegalStateException`, `CancellationException`, ...), with the worker's stack trace.
- **Large payloads** (over 64 KB: images, audio) go through shared memory: files on `/dev/shm`, each mapped once by
  both sides and reused, so a payload is copied in and out once and never pushed through the socket. Where there is
  no `/dev/shm` (macOS), the files go in the worker's private temporary directory instead.
- **Health.** The worker is pinged every 2 seconds. One that answers nothing for 30 seconds is killed and treated as
  crashed. The worker's own reader answers pings while a model loads or generates, so only a wedged process times out.
- **Crashes.** When the worker dies, every call in flight fails with a `WorkerCrashedException`. It names the exit code
  or signal (`SIGABRT` for an abort), the fatal signal the JVM's crash handler reported (`SIGSEGV` for a segfault in
  native code), and the worker's last lines of output, which include the path of its `hs_err` crash report. Workers run
  with core dumps off, because the core of a process holding a model's weights runs to gigabytes.
- **Reaping.** The worker exits as soon as its standard input or its socket closes. That happens when you close the
  model or the worker, and when your JVM dies, however it dies: the OS closes a dead process's pipes. A worker never
  outlives its parent.

`Worker.Options` sets the JVM options, class path, heartbeat, start timeout, shared-memory threshold and backoff, and
whether the worker's output (the engines' logs) is copied to your standard error. It is on by default.

The class path is your JVM's `java.class.path`. An app that loads its classes some other way (a Spring Boot fat jar,
an IDE plugin's class loader) passes the jars the worker needs with `withClassPath`.

## With a ModelPool

`ModelPool` takes any loader, so a pooled model is isolated when its loader says so:

```java
try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf) + (100L << 20),   // weights + the worker JVM
                              () -> LlamaModel.isolated(gguf.toString()))) {
    lease.model().chat(...);
}
```

- **Budgets.** A model's memory is in its worker, so the size you pass still estimates it. Add about 100 MB for each
  worker JVM. The pool's budget then counts workers too.
- **Unloading** closes the model, which stops its worker and returns all its memory to the OS. Nothing is left
  behind in your JVM.
- **After a crash**, the pooled model stays in the pool and reloads on its next call. The pool doesn't see the crash:
  the lease's call fails with `WorkerCrashedException`, and the next caller pays for the reload.

To serve many models from a few workers, with affinity, one budget and priorities, see
[Dispatch](/docs/guides/dispatch/).
