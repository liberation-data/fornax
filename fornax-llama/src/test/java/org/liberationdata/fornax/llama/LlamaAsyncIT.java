package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.StreamOverflowException;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The async API, streaming, cancel and safe callbacks against Qwen3 4B, when it's in the Hugging Face cache and the
 * native library is built; skipped otherwise.
 */
class LlamaAsyncIT {
    static final String REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF", FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    static final List<Message> FORNAX = List.of(Message.user("Reply with exactly the word: fornax"));
    static final List<Message> COUNT = List.of(Message.user("Count from 1 to 300, one number per line, nothing else."));
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1);
    static LlamaModel model;

    @BeforeAll
    static void load() {
        var path = HfCache.find(REPO, FILE);
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        model = new LlamaModel(path.get().toString(), -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    /** A subscriber that records pieces; it can sleep on each, cancel after some, or hand each to a hook. */
    static class Pieces implements Flow.Subscriber<String> {
        final List<String> items = Collections.synchronizedList(new ArrayList<>());
        final CountDownLatch done = new CountDownLatch(1);
        final long request, sleepMs;
        final int cancelAfter;
        volatile Flow.Subscription subscription;
        volatile Throwable error;
        volatile boolean completed;
        volatile Consumer<Integer> onCount = n -> {};

        Pieces() { this(Long.MAX_VALUE, 0, -1); }

        Pieces(long request, long sleepMs, int cancelAfter) { this.request = request; this.sleepMs = sleepMs; this.cancelAfter = cancelAfter; }

        @Override public void onSubscribe(Flow.Subscription s) { subscription = s; if (request > 0) s.request(request); }

        @Override public void onNext(String piece) {
            items.add(piece);
            onCount.accept(items.size());
            if (sleepMs > 0) {
                try { Thread.sleep(sleepMs); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            }
            if (cancelAfter > 0 && items.size() == cancelAfter) subscription.cancel();
        }

        @Override public void onError(Throwable t) { error = t; done.countDown(); }

        @Override public void onComplete() { completed = true; done.countDown(); }

        String text() { synchronized (items) { return String.join("", items); } }

        void await() throws InterruptedException { assertTrue(done.await(60, TimeUnit.SECONDS), "never completed"); }
    }

    static Throwable cause(CompletableFuture<?> f) {
        return assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS)).getCause();
    }

    @Test
    void aThrowingConsumerSurfacesInsteadOfCrashing() {
        var boom = new IllegalStateException("the consumer broke");
        var thrown = assertThrows(IllegalStateException.class,
                () -> model.chat(COUNT, GREEDY.withMaxTokens(64), piece -> { throw boom; }));
        assertSame(boom, thrown);
        var thrownJson = assertThrows(IllegalStateException.class, () -> model.chatJson("""
                {"messages": [{"role": "user", "content": "Say hello."}], "temperature": 0, "max_tokens": 16}""",
                piece -> { throw boom; }));
        assertSame(boom, thrownJson);
        assertTrue(model.chat(FORNAX, GREEDY.withMaxTokens(8)).text().toLowerCase().contains("fornax"), "the model still works");
    }

    @Test
    void cancelStopsACallWithoutACallback() throws Exception {
        var timer = Thread.ofVirtual().start(() -> {
            try { Thread.sleep(300); } catch (InterruptedException e) { return; }
            model.cancel();
        });
        Result r = model.chat(COUNT, GREEDY.withMaxTokens(1000));
        timer.join();
        assertTrue(r.generatedTokens() < 1000, "generated " + r.generatedTokens());
    }

    @Test
    void manyVirtualThreadsChatAsyncAndAllFinish() throws Exception {
        int n = 16;
        List<String> replies = Collections.synchronizedList(new ArrayList<>());
        List<Thread> callers = new ArrayList<>();
        for (int i = 0; i < n; i++)
            callers.add(Thread.ofVirtual().start(() -> replies.add(model.chatAsync(FORNAX, GREEDY.withMaxTokens(8)).join().text())));
        for (Thread t : callers) t.join();
        assertEquals(n, replies.size());
        assertEquals(1, replies.stream().distinct().count(), "greedy replies differ: " + replies);
        assertTrue(replies.getFirst().toLowerCase().contains("fornax"), replies.getFirst());
    }

    @Test
    void streamsTheReplyToASubscriber() throws Exception {
        var p = new Pieces();
        Result r = model.chatAsync(FORNAX, GREEDY.withMaxTokens(8), p).get(60, TimeUnit.SECONDS);
        p.await();
        assertTrue(p.completed);
        assertEquals(r.text(), p.text());
    }

    @Test
    void aColdPublisherRunsOncePerSubscription() throws Exception {
        Flow.Publisher<String> reply = model.chatStream(FORNAX, GREEDY.withMaxTokens(8));
        var a = new Pieces();
        var b = new Pieces();
        reply.subscribe(a);
        reply.subscribe(b);
        a.await();
        b.await();
        assertEquals(a.text(), b.text());
        assertTrue(a.text().toLowerCase().contains("fornax"), a.text());
    }

    @Test
    void aSlowSubscriberDoesNotSlowTheGeneration() throws Exception {
        long t0 = System.nanoTime();
        Result plain = model.chatAsync(COUNT, GREEDY.withMaxTokens(96)).get(60, TimeUnit.SECONDS);
        long plainMs = (System.nanoTime() - t0) / 1_000_000;

        var slow = new Pieces(Long.MAX_VALUE, 100, -1);  // 100 ms a piece: ~10 s to consume what takes ~2 s to generate
        t0 = System.nanoTime();
        Result r = model.chatAsync(COUNT, GREEDY.withMaxTokens(96), slow).get(60, TimeUnit.SECONDS);
        long streamedMs = (System.nanoTime() - t0) / 1_000_000;
        int consumedAtEnd = slow.items.size();
        System.out.printf("plain %d ms, with a slow subscriber %d ms; %d of %d pieces consumed when it finished%n",
                plainMs, streamedMs, consumedAtEnd, r.generatedTokens());
        assertEquals(plain.text(), r.text());
        assertTrue(streamedMs < plainMs * 1.5 + 500, "the slow subscriber slowed it: " + streamedMs + " ms vs " + plainMs);
        assertTrue(consumedAtEnd < r.generatedTokens() / 2, "the subscriber kept up, so this proves nothing");
        slow.await();
        assertEquals(r.text(), slow.text(), "every piece arrived in the end");
    }

    @Test
    void overflowStopsTheGenerationAndTellsTheSubscriber() throws Exception {
        model.streamOn(r -> Thread.ofVirtual().start(r), 4);
        try {
            var stalled = new Pieces(1, 0, -1);  // takes one piece and never asks for more
            var f = model.chatAsync(COUNT, GREEDY.withMaxTokens(400), stalled);
            assertInstanceOf(StreamOverflowException.class, cause(f));
            stalled.await();
            assertInstanceOf(StreamOverflowException.class, stalled.error);
            assertTrue(stalled.items.size() <= 1);
        } finally {
            model.streamOn(r -> Thread.ofVirtual().start(r), ModelThread.DEFAULT_BUFFER);
        }
    }

    @Test
    void cancellingTheSubscriptionStopsTheGeneration() throws Exception {
        var p = new Pieces(Long.MAX_VALUE, 0, 3);
        Result r = model.chatAsync(COUNT, GREEDY.withMaxTokens(400), p).get(60, TimeUnit.SECONDS);
        assertTrue(r.generatedTokens() < 40, "generated " + r.generatedTokens());
        assertEquals(3, p.items.size());
    }

    @Test
    void cancelMidStreamReturnsTheTextSoFar() throws Exception {
        var p = new Pieces();
        p.onCount = n -> { if (n == 3) model.cancel(); };
        Result r = model.chatAsync(COUNT, GREEDY.withMaxTokens(400), p).get(60, TimeUnit.SECONDS);
        p.await();
        assertTrue(p.completed);
        assertTrue(r.generatedTokens() < 40, "generated " + r.generatedTokens());
        assertTrue(r.text().startsWith(p.text()) || p.text().startsWith(r.text()));
    }

    @Test
    void cancellingTheFutureStopsItsGenerationAndTheNextCallRunsAtOnce() throws Exception {
        var started = new CountDownLatch(1);
        var p = new Pieces();
        p.onCount = n -> started.countDown();
        var f = model.chatAsync(COUNT, GREEDY.withMaxTokens(2000), p);
        var next = model.chatAsync(FORNAX, GREEDY.withMaxTokens(8));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        long t0 = System.nanoTime();
        assertTrue(f.cancel(true));
        assertTrue(next.get(60, TimeUnit.SECONDS).text().toLowerCase().contains("fornax"));
        long waitedMs = (System.nanoTime() - t0) / 1_000_000;
        assertTrue(waitedMs < 3000, "the cancelled generation kept running for " + waitedMs + " ms");
        assertThrows(CancellationException.class, f::join);
        p.await();
        assertInstanceOf(CancellationException.class, p.error);
    }

    @Test
    void aThrowingSubscriberFailsTheFuture() {
        var boom = new IllegalArgumentException("bad piece");
        var p = new Pieces() {
            @Override public void onNext(String piece) { throw boom; }
        };
        assertSame(boom, cause(model.chatAsync(COUNT, GREEDY.withMaxTokens(400), p)));
    }

    @Test
    void chatJsonAsyncCallsTools() throws Exception {
        String req = """
            {"messages": [{"role": "user", "content": "What's the weather in Nashville right now?"}],
             "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
             "temperature": 0, "max_tokens": 128, "seed": 1}""";
        String res = model.chatJsonAsync(req).get(60, TimeUnit.SECONDS);
        assertTrue(res.contains("\"finish_reason\":\"tool_calls\"") && res.contains("get_weather"), res);
    }

    @Test
    void chatJsonAsyncStreamsContent() throws Exception {
        var p = new Pieces();
        String res = model.chatJsonAsync("""
                {"messages": [{"role": "user", "content": "Reply with exactly the word: fornax"}], "temperature": 0, "max_tokens": 16}""",
                List.of(), p).get(60, TimeUnit.SECONDS);
        p.await();
        assertEquals(Json.get(Json.parse(res), "message", "content"), p.text());
    }

    /** 5000 optional properties overflowed the native schema converter's stack and crashed the JVM (#74). */
    @Test
    void aSchemaTooBigForTheGrammarIsRefusedBeforeTheEngine() {
        var ask = List.of(Message.user("The invoice from Acme is for 42 dollars."));
        long t0 = System.nanoTime();
        var e = assertThrows(IllegalArgumentException.class, () -> model.extract(ask, SchemasTest.object(5000, 0), GREEDY));
        assertTrue(e.getMessage().contains("5000 optional properties"), e.getMessage());
        assertTrue(System.nanoTime() - t0 < 5_000_000_000L, "refused at once, not after the converter ran");
        assertTrue(model.chat(ask, GREEDY.withMaxTokens(5)).generatedTokens() > 0, "the model goes on");
    }

    @Test
    void extractAsyncFollowsTheSchema() throws Exception {
        String json = model.extractAsync(List.of(Message.user("The invoice from Acme is for 42 dollars.")),
                """
                {"type": "object", "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
                 "required": ["vendor", "total"]}""", GREEDY.withMaxTokens(512)).get(60, TimeUnit.SECONDS);
        assertEquals(42.0, ((Number) Json.get(Json.parse(json), "total")).doubleValue(), json);
    }

    @Test
    void aCancelledExtractThrowsRatherThanReturningCutOffJson() throws Exception {
        var f = model.extractAsync(COUNT, """
                {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "integer"}, "minItems": 300}},
                 "required": ["numbers"]}""", GREEDY.withMaxTokens(2000));
        Thread.sleep(500);
        model.cancel();
        assertThrows(CancellationException.class, () -> f.get(60, TimeUnit.SECONDS));  // get() throws it unwrapped
    }

    @Test
    void anAsyncCallerCanWaitForTheModelFromAVirtualThreadWhileBlockingCallersQueue() throws Exception {
        AtomicReference<Result> blocking = new AtomicReference<>();
        Thread b = Thread.ofVirtual().start(() -> blocking.set(model.chat(FORNAX, GREEDY.withMaxTokens(8))));
        Result async = model.chatAsync(FORNAX, GREEDY.withMaxTokens(8)).get(60, TimeUnit.SECONDS);
        b.join();
        assertEquals(async.text(), blocking.get().text());
    }
}
