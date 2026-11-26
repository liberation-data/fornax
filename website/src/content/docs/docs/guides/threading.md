---
title: Threading
description: What blocks, where callbacks run, how to call Kiln from virtual threads with the async API, how streaming handles a slow subscriber, and how cancel and close behave for each engine.
---

A Kiln call is a native function call: a generation runs inside the engine, on some Java thread, for seconds or
minutes. This page covers which thread that is, what it means for virtual threads, and how to stop a call or close a
model while one is running.

## The short version

- **Every model can be called from any thread.** Calls to one model run one at a time, in the order they arrive.
  Different models run in parallel.
- **The blocking methods** (`chat`, `chatJson`, `extract`, `speak`, `generate`, ...) run on the calling thread, and so
  do their callbacks. They behave as they always have.
- **The `...Async` methods** (`chatAsync`, `chatJsonAsync`, `extractAsync`, `speakAsync`, `generateAsync`, ...) run the
  native call on the model's own platform thread and return a `CompletableFuture`. Use them from virtual threads.
- **Streaming** goes to a `java.util.concurrent.Flow.Subscriber` through a bounded buffer, delivered on virtual
  threads. A slow subscriber doesn't slow the GPU.
- **`cancel()`** stops the call in progress, from any thread. So does cancelling its future or its subscription.
- **`close()`** can be called twice, and from any thread while a call is running: it cancels the call, waits for it to
  return, then frees the weights.
- **A callback that throws** no longer takes the JVM down. The call stops and rethrows that exception.

## Virtual threads

A virtual thread that makes a native call is pinned to its carrier thread for the whole call. The JVM has about one
carrier per core, so a few virtual threads each generating text can occupy all of them, and unrelated virtual-thread
work (HTTP handlers, other agents) stalls until a generation finishes.

The async methods avoid that. Each model has one platform thread, named `kiln-<model file>`, started on first use. It
owns the engine instance and makes every async call's downcall. A virtual thread that waits on the future unmounts, and
its carrier goes on running other work:

```java
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;

try (var llm = new LlamaModel(gguf);
     var executor = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
    for (String question : questions) {
        executor.submit(() -> {
            var reply = llm.chatAsync(List.of(Message.user(question)), Options.DEFAULT).join();  // unmounts while waiting
            System.out.println(reply.text());
        });
    }
}
```

The calls still run one at a time on the model's thread, in submission order: one model generates one reply at a time.
What changes is that the waiting threads cost nothing.

Kiln's tests show the difference with the virtual-thread scheduler cut to one carrier
(`-Djdk.virtualThreadScheduler.parallelism=1`) and a heartbeat virtual thread that wakes every 5 ms. With Qwen3 4B,
two blocking `chat` calls from virtual threads stopped the heartbeat for the whole time they ran. With eight `chatAsync`
calls from virtual threads, it kept its 5 ms beat throughout.

The model's thread has an 8 MB stack, the size of a command-line tool's main thread, because some GPU backends recurse
deeply in native code (see [GPU backends](/docs/guides/gpu-backends/)).

Futures complete on a virtual thread, not on the model's thread, so a `thenApply` or `thenAccept` you chain on can
take its time without holding up the next generation.

### With a ModelPool

`ModelPool.acquire` waits on a fair semaphore, which unmounts a virtual thread, so a pool works from virtual threads as
it is. Borrow the model, await the async call, then give the model back:

```java
try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf), () -> new LlamaModel(gguf.toString()))) {
    String reply = lease.model().chatJsonAsync(requestJson).join();
}
```

Await the future before the lease closes. The pool only unloads a model that isn't leased, so once the lease is
closed it may unload this one while your call is still queued or running, and `close()` then cancels the call.

## Streaming

Each streaming async method takes a `Flow.Subscriber`, and each has a cold `Flow.Publisher` form that runs the call
once for each subscription:

| Model | Subscriber | Publisher |
|---|---|---|
| `LlamaModel` | `chatAsync(messages, options, subscriber)`, `chatJsonAsync(request, images, subscriber)` | `chatStream(messages, options)` |
| `TtsModel` | `speakAsync(text, voice, options, subscriber)` | `speakStream(text, voice, options)` |

```java
import java.util.concurrent.Flow;

var done = llm.chatAsync(List.of(Message.user("Write a haiku about a kiln.")), Options.DEFAULT,
        new Flow.Subscriber<String>() {
            public void onSubscribe(Flow.Subscription s) { s.request(Long.MAX_VALUE); }
            public void onNext(String piece) { System.out.print(piece); }
            public void onError(Throwable t) { t.printStackTrace(); }
            public void onComplete() { System.out.println(); }
        });
Result r = done.join();   // the whole reply and its token counts, as chat returns them
```

Pieces go from the model's thread into a bounded buffer (a JDK `SubmissionPublisher`) and reach the subscriber on a
virtual thread. The generation never waits for the subscriber: a subscriber that sleeps 100 ms on each piece still
leaves the generation running at full speed, and catches up afterwards.

**Overflow.** A subscriber that falls a whole buffer behind (1024 pieces by default, or one that stops requesting)
stops the generation. Its `onError` and the future both get a `StreamOverflowException`. Kiln doesn't wait, which would
hold the GPU idle for the slowest consumer, and doesn't drop pieces, which would hand you a reply with holes in it.
Change the buffer, or the executor the subscriber runs on, with `streamOn`:

```java
llm.streamOn(Executors.newSingleThreadExecutor(), 4096);   // for calls made after this
```

**Errors.** A subscriber method that throws cancels its subscription. If the generation is still running, it stops,
and the future fails with that exception.

The future follows the generation, and the subscriber follows delivery: the future can complete while a slow subscriber
is still working through the buffer. `onComplete` arrives after the last piece.

## Where blocking callbacks run

The blocking methods' callbacks (`Consumer<String> onPiece`, `Consumer<Audio> onChunk`, and stable-diffusion.cpp's log
and progress sinks) run on the thread that made the call, between tokens or steps. Keep them quick: append to a buffer,
push to a queue, write to a socket.

A callback may call `cancel()`, but not the same model's other blocking methods, and not `close()`. Both throw an
`IllegalStateException`, because the engine is in the middle of the call that's running the callback. An `...Async`
method called there is queued behind the running call, so don't wait for its future inside the callback: that waits
for itself.

`DiffusionModel.onProgress` and `onLog` are process-wide, so they see every `DiffusionModel`'s runs. stable-diffusion.cpp
also reports loading the weights through the progress sink, on the first run after a model loads, with a different
`steps` count from the sampling steps.

## When a callback throws

Before, an exception thrown from a token or chunk consumer escaped into the engine's native frames, and the JVM printed
it and exited. Now every Kiln upcall catches it, tells the engine to stop, and returns normally. When the native call
returns, the Java method throws that same exception:

```java
try {
    llm.chat(messages, Options.DEFAULT, piece -> { throw new IllegalStateException("socket closed"); });
} catch (IllegalStateException e) {
    // "socket closed": the generation stopped at that token, and the model is ready for the next call
}
```

Checked exceptions (thrown sneakily) are wrapped in a `RuntimeException`. stable-diffusion.cpp's log and progress sinks
work the same way during a `generate` call. Outside one, for example a log line while a model loads, there's no call to
hand the exception to, so Kiln prints it to standard error.

## Cancel

`cancel()` stops the call that is running on that model, from any thread. Cancelling a future, or its `orTimeout`
firing, stops that call only: if it is still queued, it never starts. Cancelling a subscription stops its call too.

| Engine | Stops | What the call returns |
|---|---|---|
| `LlamaModel` | after the current token, in every call (with or without a callback), and during reasoning and tool calls in `chatJson` | the text so far (`extract` throws `CancellationException`: cut-off JSON is no use) |
| `TtsModel` | between chunks, in a streaming call (with `onChunk` or a subscriber) | the audio so far |
| `DiffusionModel` (images and video) | at the next step, through `sd_cancel_generation` | throws `CancellationException` |
| `AceStepModel` | between LM tokens, DiT steps and VAE tiles | throws `CancellationException` |
| `Yue2Model` | between LM tokens, diffusion steps and VAE tiles | throws `CancellationException` |

Limits:

- **TTS without streaming can't stop early.** Without a callback, llama.cpp's audio generation decodes in one go, as
  `llama-tts` does, and Kiln leaves that path as it is.
- **Kokoro speaks in one chunk.** It isn't autoregressive: it decodes the whole text in one step and streams it as a
  single chunk, so there's nothing left to stop by the time a subscriber sees it.
- **A model doesn't stop while it loads its weights.** stable-diffusion.cpp loads lazily on the first run, and only
  checks for a cancel when sampling starts.

An async call whose future is cancelled (or times out) completes it with that `CancellationException` (or
`TimeoutException`). A call stopped by `cancel()`, blocking or async, returns what the engine returns, as in the table
above: an async `chatAsync` stopped that way completes normally, with the text so far.

## Close

`close()` is idempotent, and safe from any thread at any time except from inside the model's own callback:

1. Calls made after it throw an `IllegalStateException` (async ones fail their future with it).
2. Queued async calls fail with a `CancellationException`.
3. The call in progress is cancelled, and `close()` waits for its native call to return. For a call that can't stop
   early (TTS without streaming), that means waiting for it to finish.
4. The weights are freed, once. A second `close()`, even from another thread at the same moment, waits for that and
   returns.

`ModelPool` closes models only when they aren't leased, so it never hits step 3 as long as you await async calls before
closing the lease.
