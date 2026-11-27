package org.liberationdata.fornax.snippets;

// --8<-- [start:virtual-threads]
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
// --8<-- [end:virtual-threads]
// --8<-- [start:flow]
import java.util.concurrent.Flow;
// --8<-- [end:flow]

import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;

/** The snippets of website/src/content/docs/docs/guides/threading.md: compiled, never run. */
class Threading {
    void virtualThreads(String gguf, List<String> questions) {
        // --8<-- [start:virtual-threads]
        try (var llm = new LlamaModel(gguf);
             var executor = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
            for (String question : questions) {
                executor.submit(() -> {
                    var reply = llm.chatAsync(List.of(Message.user(question)), Options.DEFAULT).join();  // unmounts while waiting
                    System.out.println(reply.text());
                });
            }
        }
        // --8<-- [end:virtual-threads]
    }

    void pool(ModelPool pool, Path gguf, String requestJson) {
        // --8<-- [start:pool]
        try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf), () -> new LlamaModel(gguf.toString()))) {
            String reply = lease.model().chatJsonAsync(requestJson).join();
        }
        // --8<-- [end:pool]
    }

    void flow(LlamaModel llm) {
        // --8<-- [start:flow]
        var done = llm.chatAsync(List.of(Message.user("Write a haiku about a fornax.")), Options.DEFAULT,
                new Flow.Subscriber<String>() {
                    public void onSubscribe(Flow.Subscription s) { s.request(Long.MAX_VALUE); }
                    public void onNext(String piece) { System.out.print(piece); }
                    public void onError(Throwable t) { t.printStackTrace(); }
                    public void onComplete() { System.out.println(); }
                });
        Result r = done.join();   // the whole reply and its token counts, as chat returns them
        // --8<-- [end:flow]
    }

    void streamOn(LlamaModel llm) {
        // --8<-- [start:stream-on]
        llm.streamOn(Executors.newSingleThreadExecutor(), 4096);   // for calls made after this
        // --8<-- [end:stream-on]
    }

    void throwingConsumer(LlamaModel llm, List<Message> messages) {
        // --8<-- [start:throwing-consumer]
        try {
            llm.chat(messages, Options.DEFAULT, piece -> { throw new IllegalStateException("socket closed"); });
        } catch (IllegalStateException e) {
            // "socket closed": the generation stopped at that token, and the model is ready for the next call
        }
        // --8<-- [end:throwing-consumer]
    }
}
