package org.liberationdata.fornax.snippets;

// --8<-- [start:dispatch]
import org.liberationdata.fornax.Dispatch;
import org.liberationdata.fornax.Dispatch.QoS;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.llama.LlamaModel;
// --8<-- [end:dispatch]

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Flow;

/**
 * The snippets of website/src/content/docs/docs/guides/dispatch.md: compiled, never run. Not named Dispatch, which
 * would clash with the {@link Dispatch} it imports.
 */
class DispatchGuide {
    void dispatch(Path gguf, List<LlamaModel.Message> messages) {
        // --8<-- [start:dispatch]
        try (var dispatch = new Dispatch(Dispatch.Options.of(2, 48L << 30))) {   // two worker processes, a 48 GB budget
            var qwen = dispatch.register("qwen3-4b", ModelPool.sizeOf(gguf),
                    worker -> LlamaModel.isolated(worker, gguf.toString(), null, -1, 8192));

            LlamaModel.Result r = qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, LlamaModel.Options.DEFAULT)).join();
        }
        // --8<-- [end:dispatch]
    }

    void slots(Dispatch dispatch, Path gguf, List<LlamaModel.Message> messages) {
        // --8<-- [start:slots]
        int slots = 4, context = 4096;
        long size = ModelPool.sizeOf(gguf) + LlamaModel.kvCacheBytes(gguf.toString(), slots * context);
        var qwen = dispatch.register("qwen3-4b", size, slots,
                worker -> LlamaModel.isolated(worker, gguf.toString(), -1, context, slots));
        // up to four of these decode together in qwen's worker; more wait for a free slot
        qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, LlamaModel.Options.DEFAULT));
        // --8<-- [end:slots]
    }

    void submit(Dispatch.Model<LlamaModel> qwen, List<LlamaModel.Message> messages, LlamaModel.Options options,
                Flow.Subscriber<String> subscriber, String schema, String requestJson) {
        // --8<-- [start:submit]
        qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, options, subscriber));   // Flow streaming
        qwen.submit(QoS.BATCH, llm -> llm.extractAsync(messages, schema, options));
        qwen.submit(QoS.INTERACTIVE, llm -> llm.chatJsonAsync(requestJson));
        // --8<-- [end:submit]
    }

    void stats(Dispatch dispatch) {
        // --8<-- [start:stats]
        for (Dispatch.Entry e : dispatch.entries())          // least recently used first
            System.out.printf("%s in worker %d (pid %d), inUse=%s%n", e.key(), e.worker(), dispatch.worker(e.worker()).pid(), e.inUse());

        Dispatch.Stats s = dispatch.stats();   // requests, hits, loads, evictions, crashes, rejected, queue depths, bytes loaded
        // --8<-- [end:stats]
    }
}
