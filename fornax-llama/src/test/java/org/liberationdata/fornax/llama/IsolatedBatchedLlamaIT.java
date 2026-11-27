package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * An isolated model with slots: the worker process batches, and its requests reach it together. Against Qwen3 0.6B
 * when it's downloaded (or at -Dfornax.test.smallGguf) and the native library is built; skipped otherwise.
 */
class IsolatedBatchedLlamaIT {
    static final List<Message> COUNT = List.of(Message.user("Count from 1 to 300, one number per line, nothing else. /no_think"));
    static final List<Message> FORNAX = List.of(Message.user("Reply with exactly the word: fornax /no_think"));
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1);
    static LlamaModel model;

    @BeforeAll
    static void load() {
        var path = LlamaIsolatedIT.gguf();
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + LlamaIsolatedIT.REPO + " " + LlamaIsolatedIT.FILE);
        model = LlamaModel.isolated(path.get().toString(), -1, 2048, 2);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    /** First and last piece times of a streamed chat, in nanoseconds. */
    static long[] streamed(int maxTokens) {
        AtomicLong first = new AtomicLong(), last = new AtomicLong();
        Result r = model.chat(COUNT, GREEDY.withMaxTokens(maxTokens), p -> {
            long now = System.nanoTime();
            first.compareAndSet(0, now);
            last.set(now);
        });
        // batched greedy decoding can drift from the serial reply, and end before maxTokens
        assertTrue(r.generatedTokens() > 50, r.generatedTokens() + " tokens: " + r.text());
        return new long[]{first.get(), last.get()};
    }

    @Test void twoRequestsDecodeTogetherInTheWorker() {
        assertTrue(model.isIsolated());
        assertEquals(2, model.slots());
        var a = CompletableFuture.supplyAsync(() -> streamed(150));
        var b = CompletableFuture.supplyAsync(() -> streamed(150));
        long[] ta = a.join(), tb = b.join();
        // one after the other, one stream would end before the other began
        assertTrue(tb[0] < ta[1] && ta[0] < tb[1], "the streams overlap");
    }

    @Test void cancellingOneStopsOnlyThatOne() throws Exception {
        CountDownLatch started = new CountDownLatch(5);
        CompletableFuture<Result> cancelled = model.chatAsync(COUNT, GREEDY.withMaxTokens(200), new LlamaAsyncIT.Pieces());
        CompletableFuture<Result> other = CompletableFuture.supplyAsync(() ->
                model.chat(COUNT, GREEDY.withMaxTokens(200), p -> started.countDown()));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        cancelled.cancel(true);
        assertEquals(200, other.join().generatedTokens(), "the other request ran to max_tokens");
        assertTrue(cancelled.isCancelled());
    }

    @Test void aWorkerKilledMidBatchFailsEveryRequestInFlightAndTheNextReloads() throws Exception {
        long pid = model.worker().pid();
        CountDownLatch both = new CountDownLatch(2);
        CompletableFuture<Result> a = CompletableFuture.supplyAsync(() -> model.chat(COUNT, GREEDY.withMaxTokens(1000), p -> both.countDown()));
        CompletableFuture<Result> b = CompletableFuture.supplyAsync(() -> model.chat(COUNT, GREEDY.withMaxTokens(1000), p -> both.countDown()));
        assertTrue(both.await(60, TimeUnit.SECONDS));
        ProcessHandle.of(pid).ifPresent(ProcessHandle::destroyForcibly);
        for (var f : List.of(a, b)) {
            var e = assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS));
            assertInstanceOf(WorkerCrashedException.class, e.getCause(), String.valueOf(e.getCause()));
        }
        Result r = model.chat(FORNAX, GREEDY.withMaxTokens(32));   // a new worker, loaded with slots again
        assertTrue(r.text().toLowerCase().contains("fornax"), r.text());
        assertNotEquals(pid, model.worker().pid());
    }
}
