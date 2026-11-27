---
title: Dispatch
description: Serve many models from a pool of worker processes, routed by model affinity, loaded on demand within one memory budget, and queued by priority with aging so batch work can't starve.
---

`Dispatch`, in `fornax-core`, is a scheduler for servers. It owns a few [isolated](/docs/guides/isolation/) worker
processes and serves requests for any number of models through them:

- **Model affinity.** A request goes to the worker that already has its model loaded.
- **Loading on demand, within one budget.** A request for a model that isn't loaded takes an idle worker, and the model
  loads there. To make room, the least recently used idle models are unloaded, from any worker.
- **Priority with aging.** Requests are `INTERACTIVE` or `BATCH`. Interactive ones go first, and a batch request that
  has waited long enough is ranked with them, so it can't starve.
- **Crash recovery.** A worker that crashes takes only its own models with it. The next request for one of them loads
  it again, in another worker if one is free.

Use it when one JVM serves many callers (agents, HTTP requests) across several models on one GPU. For an app that
holds a model or two, a plain model or a [ModelPool](/docs/guides/model-pool/) is simpler, and stays the default.

## Serving a model

<!-- snippet: DispatchGuide#dispatch -->
```java
import org.liberationdata.fornax.Dispatch;
import org.liberationdata.fornax.Dispatch.QoS;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.llama.LlamaModel;

try (var dispatch = new Dispatch(Dispatch.Options.of(2, 48L << 30))) {   // two worker processes, a 48 GB budget
    var qwen = dispatch.register("qwen3-4b", ModelPool.sizeOf(gguf),
            worker -> LlamaModel.isolated(worker, gguf.toString(), null, -1, 8192));

    LlamaModel.Result r = qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, LlamaModel.Options.DEFAULT)).join();
}
```

`register(key, size, loader)` names a model. Nothing loads yet. When a request for it is granted, the loader loads it
into the worker the dispatch picked for it. `LlamaModel.isolated(worker, ...)` is the form that does this. The loader
must load into the worker it is given: the dispatch can't tell if it doesn't, and an in-process model runs in your JVM,
where a crash takes the JVM with it and its memory is still counted against the budget.

A request gets the model to itself. The worker runs nothing else meanwhile.

- **`submit(qos, body)`** runs `body` with the model once the request is granted. The dispatch holds the model until
  the future that `body` returns completes, and then gives the model back. Use the model's async methods in the body,
  so its streaming and cancel work as they always do:

  <!-- snippet: DispatchGuide#submit -->
  ```java
  qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, options, subscriber));   // Flow streaming
  qwen.submit(QoS.BATCH, llm -> llm.extractAsync(messages, schema, options));
  qwen.submit(QoS.INTERACTIVE, llm -> llm.chatJsonAsync(requestJson));
  ```

- **`call(qos, body)`** waits for the model, runs `body` on your thread, and gives the model back. Use it for the
  blocking methods: `qwen.call(QoS.BATCH, llm -> llm.chat(messages, options))`. A virtual thread unmounts while it
  waits in the queue.
- **`acquire(qos)` and `acquireAsync(qos)`** return a `ModelPool.Lease`, as `ModelPool.acquire` does. Close it to give
  the model back.

The dispatch takes a body, not a copy of every model method, so one `Dispatch` serves any engine. Your code calls the
model's own API, and the dispatch decides only when and where it runs.

## How requests are scheduled

- **A worker runs one request at a time.** The GPU is the scarce resource, so the number of workers is the number of
  requests that run at once. A model is loaded in one worker at a time, and a worker can hold several models.
- **Affinity.** A request for a loaded model waits for that model's worker, even if another worker is idle. Moving the
  model would cost a reload.
- **Placement.** A request for a model that isn't loaded takes an idle worker. The dispatch skips one that crashed
  during its last request (it is still backing off), then picks the one with the fewest bytes loaded.
- **Order.** Each time a worker frees up, the dispatch grants requests in order of class, then arrival. Requests for
  other workers' models don't wait behind one that can't run yet.
- **Aging.** A waiting request's class rises one step for each `aging` period it has waited (10 seconds by default).
  A batch request that has waited that long is ranked with interactive requests by arrival, so it goes ahead of newer
  ones. `withAging(null)` turns aging off.

## The budget

`Options.of(workers, budgetBytes)` takes one budget for everything the dispatch runs: the models' sizes (the
estimates you pass to `register`), plus 100 MB per worker JVM (`withWorkerOverhead`). It follows ModelPool's rules,
across the whole pool:

- **Room is made least recently used first.** A load that would go over the budget first unloads idle models, the one
  used longest ago first, from whichever worker holds it.
- **A model in use is never unloaded.** If a load won't fit until a request in progress finishes, it waits. Loads for
  requests ranked below it wait too, so a small batch model can't take the room a large interactive one is waiting for.
- **A model too big for the budget** still loads once nothing else is loaded.
- **Idle models unload** after `withIdleTimeout(duration)`, if you set one.

The dispatch owns the models it serves, and their memory. Don't also put them in a `ModelPool`: give the dispatch
the GPU's budget, or the part of it that its models may use.

## Cancel, timeouts and back-pressure

- **Cancelling a queued request** (`future.cancel(true)`, or `orTimeout` or `completeOnTimeout` running out) takes it out of the queue. It
  never runs, and its model never loads.
- **Cancelling a running request** cancels the future that its body returned. For a Fornax model, that stops the
  generation in the worker, and the worker takes the next request.
- **Return the model's own future.** A cancel reaches the generation only through the future the model returned.
  Cancelling one derived from it (`chatAsync(...).thenApply(...)`) doesn't stop the generation, yet the dispatch gives
  the worker back, so the next request there runs alongside it.
- **A request holds its worker until its body's future completes,** or its lease is closed. Nothing times it out for
  you: a body whose future never completes keeps the worker for good. Put `orTimeout` on requests that can hang.
- **Don't wait on the same dispatch from inside a body.** A body that waits for another request to the dispatch
  holds its worker while that request waits for one. With every worker busy (with one worker, always), neither runs.
- **Back-pressure.** At most `maxQueued` requests (10,000 by default) wait in each class. One more fails at once with
  a `RejectedExecutionException`, so a flood of batch work doesn't stop interactive requests from queueing.

## Crashes

A worker that crashes during a request (an abort, a segfault, or a kill) fails that request with a
`WorkerCrashedException`, as an isolated model does. The dispatch then drops every model that worker held. The worker
restarts on its own, with the backoff described in [Isolation](/docs/guides/isolation/). The next request for one of
the dropped models loads it again, and it prefers another idle worker over the one backing off. The dispatch doesn't
retry the failed request, because its stream may already have gone to the caller.

## Looking inside

<!-- snippet: DispatchGuide#stats -->
```java
for (Dispatch.Entry e : dispatch.entries())          // least recently used first
    System.out.printf("%s in worker %d (pid %d), inUse=%s%n", e.key(), e.worker(), dispatch.worker(e.worker()).pid(), e.inUse());

Dispatch.Stats s = dispatch.stats();   // requests, hits, loads, evictions, crashes, rejected, queue depths, bytes loaded
```

## What it costs

With Qwen3 0.6B on a Ryzen AI Max+ 395 (Vulkan), two workers, and a budget that fits one copy of the model (from
`DispatchIT`, on a GPU shared with other work):

| | |
|---|---|
| First request: start a worker, load the model, chat | 590 ms |
| A request to the loaded model (affinity hit), short chat | 39 ms |
| Switch to another model: unload, load, chat | 440 ms |
| After a worker is killed: reload in the other worker, chat | 220 ms |
| Cancel mid-generation to the next reply | 42 ms |
| An interactive request queued behind a 256-token batch generation | done 40 ms before the batch request queued ahead of it |

What you pay for is a worker per concurrent request (about 60 MB idle, plus its own GPU buffers) and a reload on a
miss. Each call also pays isolation's overhead, about 15 µs.

## Not yet

- **Continuous batching.** llama.cpp can decode several sequences in one batch, sharing one context and its KV cache,
  as `llama-server`'s parallel slots do. Many chats on one loaded model would then share its worker, instead of
  queueing for it. That is the next step. stable-diffusion.cpp, ACE-Step and YuE2 run one job at a time, so for them
  the worker stays the unit.
- **Replicas.** A model is loaded in one worker at a time. A hot model doesn't get a second copy in an idle worker.
- **Preemption.** An interactive request waits for a running batch request to finish. It doesn't pause the batch
  request, or cancel it.
- **In-process workers,** without the process isolation, and a GPU compute budget as well as a memory one.
- **Other engines.** Only `LlamaModel` runs in a worker today (see [Isolation](/docs/guides/isolation/)).
