package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaAsyncIT.Pieces;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;
import java.util.Optional;
import java.nio.file.Path;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * An isolated LlamaModel, in a worker process, against Qwen3 0.6B (small enough to load quickly, and to reload after
 * the crash test): chat, chatJson, extract, streaming and cancel across the boundary, and recovery from the worker
 * being killed mid-generation. Skipped unless the model is in the Hugging Face cache (or at -Dfornax.test.smallGguf)
 * and the native library is built.
 */
class LlamaIsolatedIT {
    static final String REPO = "Qwen/Qwen3-0.6B-GGUF", FILE = "Qwen3-0.6B-Q8_0.gguf";
    static final List<Message> FORNAX = List.of(Message.user("Reply with exactly the word: fornax /no_think"));
    static final List<Message> COUNT = List.of(Message.user("Count from 1 to 300, one number per line, nothing else. /no_think"));
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1);
    static LlamaModel model;

    static Optional<Path> gguf() {
        String p = System.getProperty("fornax.test.smallGguf");
        return p != null ? Optional.of(Path.of(p)) : HfCache.find(REPO, FILE);
    }

    @BeforeAll
    static void load() {
        var path = gguf();
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        model = LlamaModel.isolated(path.get().toString(), null, -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    @Test
    void chatsInAWorkerProcess() {
        assertTrue(model.isIsolated());
        assertNotEquals(ProcessHandle.current().pid(), model.worker().pid());
        Result r = model.chat(FORNAX, GREEDY.withMaxTokens(32));
        assertTrue(r.text().toLowerCase().contains("fornax"), r.text());
        assertTrue(r.promptTokens() > 0 && r.generatedTokens() > 0);
    }

    @Test
    void streamsPiecesToACallbackAndASubscriber() throws Exception {
        StringBuilder b = new StringBuilder();
        Result r = model.chat(FORNAX, GREEDY.withMaxTokens(32), b::append);
        assertEquals(r.text(), b.toString());

        var p = new Pieces();
        Result s = model.chatAsync(FORNAX, GREEDY.withMaxTokens(32), p).get(60, TimeUnit.SECONDS);
        p.await();
        assertTrue(p.completed);
        assertEquals(s.text(), p.text());
        assertEquals(r.text(), s.text(), "greedy, so the same reply");
    }

    @Test
    void chatJsonAndExtract() {
        StringBuilder content = new StringBuilder();
        String response = model.chatJson("""
                {"messages": [{"role": "user", "content": "Say hello. /no_think"}], "temperature": 0, "max_tokens": 32}""",
                content::append);
        Object message = Json.get(Json.parse(response), "message", "content");
        assertInstanceOf(String.class, message);
        assertFalse(content.isEmpty());

        String json = model.extract(List.of(Message.user("The answer is 42. /no_think")),
                """
                {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}""",
                GREEDY.withMaxTokens(64));
        assertEquals(42L, ((Number) Json.get(Json.parse(json), "answer")).longValue(), json);
    }

    @Test
    void cancelMidStreamReturnsTheTextSoFar() throws Exception {
        var p = new Pieces();
        p.onCount = n -> { if (n == 5) model.cancel(); };
        Result r = model.chatAsync(COUNT, GREEDY.withMaxTokens(600), p).get(60, TimeUnit.SECONDS);
        p.await();
        assertTrue(r.generatedTokens() < 100, "generated " + r.generatedTokens());
        assertTrue(r.text().startsWith(p.text()) || p.text().startsWith(r.text()));
    }

    @Test
    void cancellingTheFutureStopsTheGenerationInTheWorker() throws Exception {
        var p = new Pieces();
        var started = new CompletableFuture<Void>();
        p.onCount = n -> started.complete(null);
        CompletableFuture<Result> f = model.chatAsync(COUNT, GREEDY.withMaxTokens(2000), p);
        started.get(60, TimeUnit.SECONDS);
        f.cancel(true);
        assertThrows(CancellationException.class, f::join);
        long t0 = System.nanoTime();
        assertTrue(model.chat(FORNAX, GREEDY.withMaxTokens(32)).text().toLowerCase().contains("fornax"));
        long ms = (System.nanoTime() - t0) / 1_000_000;
        assertTrue(ms < 20_000, "the next call waited " + ms + " ms: the cancelled one kept generating");
    }

    @Test
    void aThrowingConsumerSurfacesAndTheModelKeepsWorking() {
        var boom = new IllegalStateException("the consumer broke");
        var thrown = assertThrows(IllegalStateException.class, () -> model.chat(COUNT, GREEDY.withMaxTokens(64), piece -> { throw boom; }));
        assertSame(boom, thrown);
        assertTrue(model.chat(FORNAX, GREEDY.withMaxTokens(32)).text().toLowerCase().contains("fornax"));
    }

    @Test
    void aWorkerKilledMidGenerationFailsTheCallAndTheNextCallReloads() throws Exception {
        long pid = model.worker().pid();
        var p = new Pieces();
        p.onCount = n -> { if (n == 3) ProcessHandle.of(pid).ifPresent(ProcessHandle::destroyForcibly); };
        var f = model.chatAsync(COUNT, GREEDY.withMaxTokens(2000), p);
        var e = assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS));
        assertInstanceOf(WorkerCrashedException.class, e.getCause(), String.valueOf(e.getCause()));
        p.await();
        assertInstanceOf(WorkerCrashedException.class, p.error, "the subscriber hears of it too");

        Result r = model.chat(FORNAX, GREEDY.withMaxTokens(32));   // a new worker, the weights loaded again
        assertTrue(r.text().toLowerCase().contains("fornax"), r.text());
        assertNotEquals(pid, model.worker().pid());
    }
}
