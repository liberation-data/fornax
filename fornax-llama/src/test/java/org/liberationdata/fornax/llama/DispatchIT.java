package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.liberationdata.fornax.Dispatch;
import org.liberationdata.fornax.Dispatch.QoS;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaAsyncIT.Pieces;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A Dispatch of two worker processes serving Qwen3 0.6B under two keys, with a budget that fits one: so at most two
 * worker processes and one loaded model at a time. Affinity, chat / chatJson / streaming through the dispatch, an
 * interactive request overtaking a queued batch one, cancel, eviction between the keys, and a worker killed mid-generation
 * with the next request rerouted to the other worker. Prints its timings. Skipped without the model.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DispatchIT {
    static final List<Message> FORNAX = LlamaIsolatedIT.FORNAX;
    static final List<Message> COUNT = LlamaIsolatedIT.COUNT;
    static final Options GREEDY = LlamaIsolatedIT.GREEDY;
    static Dispatch dispatch;
    static Dispatch.Model<LlamaModel> a, b;

    @BeforeAll
    static void start() {
        var path = LlamaIsolatedIT.gguf();
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + LlamaIsolatedIT.REPO + " " + LlamaIsolatedIT.FILE);
        Path gguf = path.get();
        long size = ModelPool.sizeOf(gguf);
        var o = Dispatch.Options.of(2, size + size / 2 + 2 * (100L << 20));   // room for one copy, and two workers
        dispatch = new Dispatch(o);
        a = dispatch.register("qwen-a", size, w -> LlamaModel.isolated(w, gguf.toString(), null, -1, 4096));
        b = dispatch.register("qwen-b", size, w -> LlamaModel.isolated(w, gguf.toString(), null, -1, 2048));
    }

    @AfterAll
    static void stop() { if (dispatch != null) dispatch.close(); }

    static long ms(long startNanos) { return (System.nanoTime() - startNanos) / 1_000_000; }

    static Result chat(Dispatch.Model<LlamaModel> m, QoS qos) throws Exception {
        return m.submit(qos, llm -> llm.chatAsync(FORNAX, GREEDY.withMaxTokens(32))).get(120, TimeUnit.SECONDS);
    }

    static long pidOf(String key) {
        var e = dispatch.entries().stream().filter(x -> x.key().equals(key)).findFirst().orElseThrow();
        return dispatch.worker(e.worker()).pid();
    }

    @Test @Order(1)
    void servesChatChatJsonAndStreamingFromTheWorkerThatHasTheModel() throws Exception {
        long t = System.nanoTime();
        Result cold = chat(a, QoS.INTERACTIVE);
        long coldMs = ms(t);
        assertTrue(cold.text().toLowerCase().contains("fornax"), cold.text());
        long pid = pidOf("qwen-a");

        t = System.nanoTime();
        Result warm = chat(a, QoS.INTERACTIVE);
        long warmMs = ms(t);
        assertEquals(cold.text(), warm.text(), "greedy");
        assertEquals(pid, pidOf("qwen-a"), "an affinity hit");

        String json = a.submit(QoS.INTERACTIVE, llm -> llm.chatJsonAsync("""
                {"messages": [{"role": "user", "content": "Say hello. /no_think"}], "temperature": 0, "max_tokens": 32}"""))
                .get(60, TimeUnit.SECONDS);
        assertTrue(json.contains("\"role\""), json);

        var pieces = new Pieces();
        Result streamed = a.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(FORNAX, GREEDY.withMaxTokens(32), pieces)).get(60, TimeUnit.SECONDS);
        pieces.await();
        assertEquals(streamed.text(), pieces.text());

        String blocking = a.call(QoS.BATCH, llm -> llm.chat(FORNAX, GREEDY.withMaxTokens(32)).text());
        assertEquals(cold.text(), blocking);

        var s = dispatch.stats();
        assertEquals(1, s.loads());
        assertEquals(4, s.hits());
        System.out.printf("dispatch IT: first chat (load + chat) %d ms, warm chat %d ms%n", coldMs, warmMs);
    }

    @Test @Order(2)
    void anInteractiveRequestOvertakesAQueuedBatchOne() throws Exception {
        var started = new CountDownLatch(1);
        var pieces = new Pieces();
        pieces.onCount = n -> started.countDown();
        var running = a.submit(QoS.BATCH, llm -> llm.chatAsync(COUNT, GREEDY.withMaxTokens(256), pieces));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        long t = System.nanoTime();
        var batch = a.submit(QoS.BATCH, llm -> llm.chatAsync(FORNAX, GREEDY.withMaxTokens(32)).thenApply(r -> System.nanoTime()));
        var interactive = a.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(FORNAX, GREEDY.withMaxTokens(32)).thenApply(r -> System.nanoTime()));
        long iDone = interactive.get(120, TimeUnit.SECONDS), bDone = batch.get(120, TimeUnit.SECONDS);
        Result r = running.get(120, TimeUnit.SECONDS);
        assertTrue(iDone < bDone, "interactive first, though queued second");
        System.out.printf("dispatch IT: behind a %d-token batch generation, interactive done after %d ms, the earlier batch request after %d ms%n",
                r.generatedTokens(), (iDone - t) / 1_000_000, (bDone - t) / 1_000_000);
    }

    @Test @Order(3)
    void cancelStopsTheGenerationAndTheWorkerTakesTheNextRequest() throws Exception {
        var five = new CountDownLatch(1);
        var pieces = new Pieces();
        pieces.onCount = n -> { if (n == 5) five.countDown(); };
        var f = a.submit(QoS.BATCH, llm -> llm.chatAsync(COUNT, GREEDY.withMaxTokens(1000), pieces));
        assertTrue(five.await(60, TimeUnit.SECONDS));
        long t = System.nanoTime();
        f.cancel(true);
        assertThrows(CancellationException.class, f::join);
        Result next = chat(a, QoS.INTERACTIVE);
        assertTrue(next.text().toLowerCase().contains("fornax"));
        System.out.printf("dispatch IT: cancel mid-generation to the next reply %d ms (%d pieces streamed)%n", ms(t), pieces.items.size());
    }

    @Test @Order(4)
    void theSecondKeyEvictsTheFirstSoOnlyOneModelIsLoaded() throws Exception {
        long t = System.nanoTime();
        Result r = chat(b, QoS.INTERACTIVE);
        long loadMs = ms(t);
        assertTrue(r.text().toLowerCase().contains("fornax"));
        var entries = dispatch.entries();
        assertEquals(List.of("qwen-b"), entries.stream().map(Dispatch.Entry::key).toList(), "a was unloaded for b");
        assertEquals(1, entries.get(0).worker(), "b loaded into the idle, empty worker");
        assertTrue(dispatch.worker(0).isRunning() && dispatch.worker(1).isRunning());
        assertEquals(1, dispatch.stats().evictions());
        System.out.printf("dispatch IT: switch to the second key (unload + load + chat) %d ms%n", loadMs);
    }

    @Test @Order(5)
    void aWorkerKilledMidGenerationFailsTheRequestAndTheNextOneGoesToTheOtherWorker() throws Exception {
        long victim = pidOf("qwen-b");
        var started = new CountDownLatch(1);
        var pieces = new Pieces();
        pieces.onCount = n -> started.countDown();
        CompletableFuture<Result> f = b.submit(QoS.BATCH, llm -> llm.chatAsync(COUNT, GREEDY.withMaxTokens(1000), pieces));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        ProcessHandle.of(victim).orElseThrow().destroyForcibly();
        var e = assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS));
        assertInstanceOf(WorkerCrashedException.class, e.getCause());
        assertTrue(dispatch.entries().isEmpty(), "b went with its worker");

        long t = System.nanoTime();
        Result r = chat(b, QoS.INTERACTIVE);
        long rerouteMs = ms(t);
        assertTrue(r.text().toLowerCase().contains("fornax"));
        assertEquals(dispatch.worker(0).pid(), pidOf("qwen-b"), "rerouted to the worker that didn't crash");
        assertNotEquals(victim, pidOf("qwen-b"));
        assertEquals(1, dispatch.stats().crashes());
        System.out.printf("dispatch IT: after the crash, reload in the other worker + chat %d ms%n", rerouteMs);
    }
}
