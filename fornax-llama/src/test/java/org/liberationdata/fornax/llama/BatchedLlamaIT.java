package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A model loaded with slots (continuous batching) against Qwen3 0.6B, when it's in the Hugging Face cache and the
 * native library is built; skipped otherwise.
 */
class BatchedLlamaIT {
    static final String REPO = "Qwen/Qwen3-0.6B-GGUF", FILE = "Qwen3-0.6B-Q8_0.gguf";
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1).withMaxTokens(48);
    static final List<String> QUESTIONS = List.of(
            "Name three primary colours. /no_think",
            "What is the capital of France? Answer in one sentence. /no_think",
            "Write a haiku about a kiln. /no_think",
            "List the first five prime numbers. /no_think");
    static String path;
    static LlamaModel batched;

    @BeforeAll
    static void load() {
        var p = HfCache.find(REPO, FILE);
        assumeTrue(p.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        path = p.get().toString();
        batched = new LlamaModel(path, -1, 2048, 4);
    }

    @AfterAll
    static void close() { if (batched != null) batched.close(); }

    static List<Message> ask(String q) { return List.of(Message.user(q)); }

    /**
     * Batched decoding sums in a different order, so greedy replies can drift from the serial ones; this reports how
     * many characters each shares with its serial reply, and needs the start of each to agree.
     */
    @Test void concurrentGreedyChatsMatchTheSerialOnesAtTheStart() {
        List<String> serial = new ArrayList<>();
        try (LlamaModel one = new LlamaModel(path, -1, 2048)) {
            for (String q : QUESTIONS) serial.add(one.chat(ask(q), GREEDY).text());
        }
        List<CompletableFuture<Result>> fs = new ArrayList<>();
        for (String q : QUESTIONS) fs.add(batched.chatAsync(ask(q), GREEDY));
        for (int i = 0; i < QUESTIONS.size(); i++) {
            String b = fs.get(i).join().text(), s = serial.get(i);
            int common = 0;
            while (common < Math.min(b.length(), s.length()) && b.charAt(common) == s.charAt(common)) common++;
            System.out.printf("slot reply %d: %d of %d characters match the serial reply%n", i, common, s.length());
            assertTrue(common >= Math.min(12, s.length()), "reply " + i + " diverged at once:\n" + b + "\n---\n" + s);
        }
    }

    @Test void moreRequestsThanSlotsQueueAndAllFinish() {
        List<CompletableFuture<Result>> fs = new ArrayList<>();
        for (int i = 0; i < 10; i++) fs.add(batched.chatAsync(ask(QUESTIONS.get(i % 4)), GREEDY));
        for (var f : fs) assertFalse(f.join().text().isBlank());
    }

    @Test void aJsonSchemaInOneSlotAndFreeTextInAnother() {
        String schema = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}";
        CompletableFuture<String> json = CompletableFuture.supplyAsync(() ->
                batched.extract(ask("Which city is the capital of Italy? /no_think"), schema, GREEDY.withMaxTokens(64)));
        CompletableFuture<Result> text = batched.chatAsync(ask(QUESTIONS.get(2)), GREEDY);
        Object city = Json.get(Json.parse(json.join()), "city");
        assertInstanceOf(String.class, city);
        assertFalse(text.join().text().isBlank());
    }

    @Test void cancellingOneStopsOnlyThatOne() throws Exception {
        Options longer = GREEDY.withMaxTokens(200);
        List<Message> count = ask("Count from 1 to 300, one number per line, nothing else. /no_think");
        CountDownLatch started = new CountDownLatch(5);
        CompletableFuture<Result> cancelled = batched.chatAsync(count, longer, new LlamaAsyncIT.Pieces());
        CompletableFuture<Result> other = CompletableFuture.supplyAsync(() -> batched.chat(count, longer, p -> started.countDown()));
        assertTrue(started.await(60, TimeUnit.SECONDS));
        cancelled.cancel(true);
        Result r = other.join();
        assertEquals(200, r.generatedTokens(), "the other request ran to max_tokens");
        assertTrue(cancelled.isCancelled());
    }

    @Test void aLongPromptIsPrefilledWhileAnotherStreams() throws Exception {
        AtomicInteger pieces = new AtomicInteger();
        CountDownLatch streaming = new CountDownLatch(3);
        CompletableFuture<Result> stream = CompletableFuture.supplyAsync(() -> batched.chat(
                ask("Count from 1 to 300, one number per line, nothing else. /no_think"), GREEDY.withMaxTokens(400),
                p -> { pieces.incrementAndGet(); streaming.countDown(); }));
        assertTrue(streaming.await(60, TimeUnit.SECONDS));
        String filler = "The quick brown fox jumps over the lazy dog. ".repeat(150);   // ~1500 tokens: several batches
        Result longOne = batched.chat(ask(filler + "\nHow many times did the fox jump? /no_think"), GREEDY);
        int during = pieces.get();
        assertTrue(longOne.promptTokens() > 1200, "prompt " + longOne.promptTokens());
        assertTrue(during > 3, "the stream went on while the long prompt went in");
        assertEquals(400, stream.join().generatedTokens());
    }

    @Test void aSlotThatFillsItsContextEndsWithLengthWhileAnotherFinishes() throws Exception {
        try (LlamaModel small = new LlamaModel(path, -1, 256, 2)) {
            String request = "{\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 500, one number per line. /no_think\"}],"
                    + "\"temperature\":0,\"seed\":1}";
            CompletableFuture<String> full = small.chatJsonAsync(request);
            CompletableFuture<Result> brief = small.chatAsync(ask("Say hi. /no_think"), GREEDY.withMaxTokens(16));
            Object r = Json.parse(full.get(120, TimeUnit.SECONDS));
            assertEquals("length", Json.get(r, "finish_reason"));
            int used = ((Number) Json.get(r, "usage", "prompt_tokens")).intValue() + ((Number) Json.get(r, "usage", "completion_tokens")).intValue();
            // the last token is sampled from the full context and counted, but not decoded (as serial chat counts it)
            assertTrue(used <= 257 && used >= 250, "used " + used + " of the slot's 256");
            assertFalse(brief.get(120, TimeUnit.SECONDS).text().isBlank());
        }
    }

    @Test void rawCompletionAndAPromptTooLongForASlot() {
        Result r = batched.complete("The capital of France is", GREEDY.withMaxTokens(8), null);
        assertEquals(8, r.generatedTokens());
        String tooLong = "word ".repeat(3000);
        var e = assertThrows(IllegalStateException.class, () -> batched.complete(tooLong, GREEDY, null));
        assertTrue(e.getMessage().contains("exceeds a slot's context"), e.getMessage());
        assertFalse(batched.chat(ask(QUESTIONS.get(1)), GREEDY).text().isBlank(), "the model still serves after a refused request");
    }

    @Test void closeWithSlotsBusyEndsThemAndFrees() {
        LlamaModel m = new LlamaModel(path, -1, 1024, 2);
        List<Message> count = ask("Count from 1 to 300, one number per line, nothing else. /no_think");
        var a = m.chatAsync(count, GREEDY.withMaxTokens(400));
        var b = m.chatAsync(count, GREEDY.withMaxTokens(400));
        var c = m.chatAsync(count, GREEDY.withMaxTokens(400));   // queued behind the two slots
        m.close();
        // futures complete off the model's threads, just after close returns: with text so far, or cancelled
        for (var f : List.of(a, b, c)) assertDoesNotThrow(() -> { try { f.get(5, TimeUnit.SECONDS); } catch (ExecutionException | java.util.concurrent.CancellationException ok) {} });
        assertThrows(IllegalStateException.class, () -> m.chat(count, GREEDY));
    }

    @Test void slotsRefuseWhatIsntBatchedYet() {
        assertEquals(4, batched.slots());
        var img = new LlamaModel.Image(1, 1, new byte[3]);
        var e = assertThrows(IllegalStateException.class, () -> batched.chat(List.of(Message.user("what is this?", img)), GREEDY));
        assertTrue(e.getMessage().contains("images aren't batched"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> new LlamaModel(path, -1, 0, 2));
    }
}
