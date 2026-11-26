package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** close() against Qwen3 4B: each test loads its own instance (one at a time) and closes it under load. */
class LlamaCloseIT {
    static final String REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF", FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    static final List<Message> COUNT = List.of(Message.user("Count from 1 to 300, one number per line, nothing else."));
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1);
    static String gguf;

    @BeforeAll
    static void find() {
        var path = HfCache.find(REPO, FILE);
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        gguf = path.get().toString();
    }

    @Test
    void closeTwiceIsSafe() {
        var m = new LlamaModel(gguf, -1, 2048);
        m.close();
        m.close();
        assertThrows(IllegalStateException.class, () -> m.chat(COUNT, GREEDY.withMaxTokens(4)));
        var f = m.chatAsync(COUNT, GREEDY.withMaxTokens(4));
        assertInstanceOf(IllegalStateException.class, assertThrows(Exception.class, f::join).getCause());
    }

    @Test
    void closeDuringAnAsyncCallCancelsItAndQueuedCalls() throws Exception {
        var m = new LlamaModel(gguf, -1, 2048);
        var started = new CountDownLatch(1);
        var running = m.chatAsync(COUNT, GREEDY.withMaxTokens(2000), new LlamaAsyncIT.Pieces() {
            @Override public void onNext(String piece) { started.countDown(); }
        });
        var queued = m.chatAsync(COUNT, GREEDY.withMaxTokens(8));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        long t0 = System.nanoTime();
        m.close();
        long closeMs = (System.nanoTime() - t0) / 1_000_000;
        Result r = running.get(60, TimeUnit.SECONDS);
        assertTrue(r.generatedTokens() < 2000);
        assertThrows(CancellationException.class, queued::join);
        assertTrue(closeMs < 3000, "close waited " + closeMs + " ms: the generation wasn't cancelled");
        m.close();
    }

    @Test
    void closeDuringABlockingCallOnAnotherThreadWaitsForItToStop() throws Exception {
        var m = new LlamaModel(gguf, -1, 2048);
        var started = new CountDownLatch(1);
        AtomicReference<Object> outcome = new AtomicReference<>();
        Thread caller = Thread.ofPlatform().start(() -> {
            try { outcome.set(m.chat(COUNT, GREEDY.withMaxTokens(2000), piece -> started.countDown())); }
            catch (Throwable t) { outcome.set(t); }
        });
        assertTrue(started.await(60, TimeUnit.SECONDS));
        m.close();
        caller.join(10_000);
        assertInstanceOf(Result.class, outcome.get(), String.valueOf(outcome.get()));
        assertTrue(((Result) outcome.get()).generatedTokens() < 2000);
    }

    @Test
    void closeFromInsideTheCallbackIsRefused() {
        var m = new LlamaModel(gguf, -1, 2048);
        try {
            var e = assertThrows(IllegalStateException.class, () -> m.chat(COUNT, GREEDY.withMaxTokens(16), piece -> m.close()));
            assertTrue(e.getMessage().contains("close()"), e.getMessage());
        } finally {
            m.close();
        }
    }
}
