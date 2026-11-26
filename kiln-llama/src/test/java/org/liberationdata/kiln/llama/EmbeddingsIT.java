package org.liberationdata.kiln.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Scored;
import org.liberationdata.kiln.Vectors;
import org.liberationdata.kiln.llama.LlamaModel.EmbedOptions;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Mode;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.llama.LlamaModel.Weights;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs Qwen3-Embedding 0.6B when it's in the Hugging Face cache and the native library is built; skipped otherwise.
 * The reference embeddings are llama.cpp's own llama-embedding on the same model and texts
 * (scripts/embedding-reference.sh writes them).
 */
class EmbeddingsIT {
    static final Weights W = Weights.QWEN3_EMBEDDING_0_6B;
    static final String TASK = "Given a web search query, retrieve relevant passages that answer the query";
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, Mode.EMBED);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    record Reference(String text, float[] embedding) {}

    static List<Reference> reference() throws IOException {
        var in = EmbeddingsIT.class.getResourceAsStream("/embeddings/qwen3-embedding-0.6b.txt");
        assertNotNull(in, "missing fixture");
        List<Reference> out = new ArrayList<>();
        try (var r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            for (String line; (line = r.readLine()) != null; ) {
                String[] parts = line.split("\t");
                String[] xs = parts[1].trim().split(" ");
                float[] v = new float[xs.length];
                for (int i = 0; i < xs.length; i++) v[i] = Float.parseFloat(xs[i]);
                out.add(new Reference(parts[0].replace("\\n", "\n"), v));
            }
        }
        return out;
    }

    @Test
    void matchesLlamaEmbedding() throws IOException {
        List<Reference> ref = reference();
        float[][] v = model.embed(ref.stream().map(Reference::text).toList());
        assertEquals(ref.size(), v.length);
        for (int i = 0; i < v.length; i++) {
            double cos = Vectors.cosine(v[i], ref.get(i).embedding());
            System.out.printf("llama-embedding parity, text %d: cosine %.7f%n", i, cos);
            assertTrue(cos > 0.999, "text " + i + ": cosine " + cos);
        }
    }

    @Test
    void theQueryOptionIsQwensInstructionFormat() throws IOException {
        Reference query = reference().get(2);  // embedded by llama-embedding with the instruction written out
        String bare = query.text().substring(query.text().indexOf("Query:") + "Query:".length());
        float[] v = model.embed(List.of(bare), EmbedOptions.query(TASK))[0];
        assertTrue(Vectors.cosine(v, query.embedding()) > 0.999);
    }

    @Test
    void normalisedByDefault() {
        float[] v = model.embed("a glaze that crawls");
        assertEquals(1024, v.length);
        assertEquals(1024, model.embeddingSize());
        assertEquals(1.0, Math.sqrt(Vectors.dot(v, v)), 1e-4);
        float[] raw = model.embed(List.of("a glaze that crawls"), EmbedOptions.DEFAULT.withNormalize(false))[0];
        assertTrue(Math.abs(Math.sqrt(Vectors.dot(raw, raw)) - 1) > 0.01, "unnormalised length " + Math.sqrt(Vectors.dot(raw, raw)));
        assertTrue(Vectors.cosine(v, raw) > 0.9999);
    }

    /**
     * A text embedded alongside others comes out a little different from one embedded alone (cosine 0.9999 for "short"),
     * in llama-embedding too: the batch changes the GPU kernels' shapes, not what a sequence attends to.
     */
    @Test
    void batchedTextsDontAffectEachOther() {
        List<String> texts = List.of("short", "a much longer sentence about wood-fired kilns and ash glazes on stoneware",
                "Nashville", "the third text");
        float[][] together = model.embed(texts);
        for (int i = 0; i < texts.size(); i++) {
            double cos = Vectors.cosine(together[i], model.embed(texts.get(i)));
            assertTrue(cos > 0.999, texts.get(i) + ": cosine " + cos);
        }
    }

    @Test
    void moreTextsThanOneBatchTakes() {
        // llama.cpp runs at most 256 sequences at once; 600 texts take three batches
        List<String> texts = IntStream.range(0, 600).mapToObj(i -> "document number " + i + " about kilns").toList();
        float[][] v = model.embed(texts);
        assertEquals(600, v.length);
        assertTrue(Vectors.cosine(v[599], model.embed(texts.get(599))) > 0.999);
        assertTrue(Vectors.cosine(v[0], v[599]) < 0.9999);
    }

    @Test
    void aQueryFindsItsDocument() {
        List<String> docs = List.of(
                "Nashville is the capital of Tennessee and the home of country music.",
                "A pottery kiln fires stoneware at around 1200 to 1300 degrees Celsius.",
                "The Java Foreign Function and Memory API calls native code without JNI.");
        float[][] d = model.embed(docs);
        float[] q = model.embed(List.of("How hot does a kiln get?"), EmbedOptions.query(TASK))[0];
        List<Scored> top = Vectors.topK(q, d, 2);
        assertEquals(1, top.get(0).index(), top.toString());
        q = model.embed(List.of("calling C from Java"), EmbedOptions.query(TASK))[0];
        assertEquals(2, Vectors.topK(q, d, 1).get(0).index());
    }

    @Test
    void emptyListGivesNoEmbeddings() {
        assertEquals(0, model.embed(List.of()).length);
    }

    @Test
    void anEmptyTextEmbeds() {
        // Qwen3-Embedding's tokenizer appends an end-of-text, so even "" has a token to pool
        float[][] v = model.embed(List.of("", "kiln", ""));
        assertEquals(3, v.length);
        assertEquals(1.0, Math.sqrt(Vectors.dot(v[0], v[0])), 1e-4);
        assertTrue(Vectors.cosine(v[0], v[2]) > 0.999);
    }

    @Test
    void textsThatOverflowTheBatchTogetherGoInSeveral() {
        // a 64-token context is also a 64-token batch: these ~20-token texts take one batch each two or three
        List<String> texts = IntStream.range(0, 10)
                .mapToObj(i -> "passage " + i + ": a wood-fired kiln reaches stoneware temperatures over many hours").toList();
        try (var small = new LlamaModel(W, -1, 64, Mode.EMBED)) {
            float[][] v = small.embed(texts);
            assertEquals(texts.size(), v.length);
            for (int i = 0; i < texts.size(); i++) {
                double cos = Vectors.cosine(v[i], model.embed(texts.get(i)));
                assertTrue(cos > 0.999, "text " + i + ": cosine " + cos);
            }
        }
    }

    @Test
    void aTextLongerThanTheContextFails() {
        try (var small = new LlamaModel(W, -1, 64, Mode.EMBED)) {
            var e = assertThrows(IllegalStateException.class, () -> small.embed("kiln ".repeat(200)));
            assertTrue(e.getMessage().contains("tokens"), e.getMessage());
            assertEquals(1024, small.embed("still works").length);
        }
    }

    @Test
    void anEmbeddingModelDoesntChat() {
        var e = assertThrows(IllegalStateException.class,
                () -> model.chat(List.of(Message.user("hello")), Options.DEFAULT.withMaxTokens(4)));
        assertTrue(e.getMessage().contains("EMBED"), e.getMessage());
        assertThrows(IllegalStateException.class, () -> model.rerank("q", List.of("d")));
        assertThrows(IllegalStateException.class, () -> model.complete("hello", Options.DEFAULT, null));
    }

    @Test
    void aChatModelDoesntEmbed() {
        try (var chat = new LlamaModel(W, -1, 512, Mode.CHAT)) {
            assertEquals(Mode.CHAT, chat.mode());
            var e = assertThrows(IllegalStateException.class, () -> chat.embed("hello"));
            assertTrue(e.getMessage().contains("Mode.EMBED"), e.getMessage());
            assertThrows(IllegalStateException.class, chat::embeddingSize);
        }
    }

    @Test
    void anEmbeddingModelIsntAReranker() {
        var e = assertThrows(IllegalStateException.class, () -> new LlamaModel(W, -1, 512, Mode.RERANK));
        assertTrue(e.getMessage().contains("not a reranker"), e.getMessage());
    }

    @Test
    void embedAsyncAndCancel() throws Exception {
        assertTrue(Vectors.cosine(model.embedAsync(List.of("async"), EmbedOptions.DEFAULT).get()[0], model.embed("async")) > 0.999);

        List<String> many = IntStream.range(0, 20_000).mapToObj(i -> "text " + i + " about kilns, glazes and clay bodies").toList();
        var f = model.embedAsync(many, EmbedOptions.DEFAULT);
        Thread.sleep(300);
        long t0 = System.nanoTime();
        model.cancel();
        assertThrows(CancellationException.class, f::get);  // get() throws it unwrapped
        System.out.printf("embed stopped %.0f ms after cancel%n", (System.nanoTime() - t0) / 1e6);
        assertEquals(1024, model.embed("after a cancel").length);
    }

    @Test
    void closeDuringAnEmbedStopsItBetweenBatches() throws Exception {
        var m = new LlamaModel(W, Mode.EMBED);
        List<String> many = IntStream.range(0, 20_000).mapToObj(i -> "text " + i + " about kilns, glazes and clay bodies").toList();
        var running = m.embedAsync(many, EmbedOptions.DEFAULT);
        var queued = m.embedAsync(List.of("queued"), EmbedOptions.DEFAULT);
        Thread.sleep(300);
        long t0 = System.nanoTime();
        m.close();
        long closeMs = (System.nanoTime() - t0) / 1_000_000;
        assertThrows(CancellationException.class, running::get);
        assertThrows(CancellationException.class, queued::join);
        assertTrue(closeMs < 3000, "close waited " + closeMs + " ms: the embed wasn't cancelled");
        assertThrows(IllegalStateException.class, () -> m.embed("closed"));
        m.close();
    }

    @Test
    void aNullTextFailsBeforeTheEngine() {
        var e = assertThrows(NullPointerException.class, () -> model.embed(java.util.Arrays.asList("kiln", null)));
        assertTrue(e.getMessage().contains("text 1"), e.getMessage());
        assertEquals(1024, model.embed("still works").length);
    }

    @Test
    void throughput() {
        // ~20-token texts, as a search over short passages would embed (README Status figure)
        List<String> texts = IntStream.range(0, 2000)
                .mapToObj(i -> "Passage " + i + ": a wood-fired kiln reaches stoneware temperatures over many hours of stoking.").toList();
        model.embed(texts.subList(0, 256));  // warm up
        long t0 = System.nanoTime();
        model.embed(texts);
        double s = (System.nanoTime() - t0) / 1e9;
        System.out.printf("throughput: %d texts in %.2f s = %.0f embeddings/s%n", texts.size(), s, texts.size() / s);
    }
}
