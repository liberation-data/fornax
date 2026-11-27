package org.liberationdata.fornax.snippets;

// --8<-- [start:isolated]
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaModel;
// --8<-- [end:isolated]

import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.Worker;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.util.List;

/** The snippets of website/src/content/docs/docs/guides/isolation.md: compiled, never run. */
class Isolation {
    void isolated(String gguf, List<Message> messages) {
        // --8<-- [start:isolated]
        try (var llm = LlamaModel.isolated(gguf)) {           // the model loads in a worker process of its own
            llm.chat(messages, Options.DEFAULT, System.out::print);
        } catch (WorkerCrashedException e) {
            // the engine crashed: e.signal(), e.fatalSignal() ("SIGSEGV"), e.lastOutput() (its last log lines)
        }
        // --8<-- [end:isolated]
    }

    void sharedWorker(String smallGguf, String largeGguf) {
        // --8<-- [start:shared-worker]
        try (var worker = new Worker("chat-models", Worker.Options.DEFAULT)) {
            var small = LlamaModel.isolated(worker, smallGguf, null, -1, 8192);
            var large = LlamaModel.isolated(worker, largeGguf, null, -1, 8192);
            // ...
        }
        // --8<-- [end:shared-worker]
    }

    void pool(ModelPool pool, Path gguf, List<Message> messages) {
        // --8<-- [start:pool]
        try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf) + (100L << 20),   // weights + the worker JVM
                                      () -> LlamaModel.isolated(gguf.toString()))) {
            lease.model().chat(messages, Options.DEFAULT);
        }
        // --8<-- [end:pool]
    }
}
