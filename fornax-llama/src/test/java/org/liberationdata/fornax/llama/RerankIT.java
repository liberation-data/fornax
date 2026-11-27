package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Runs Qwen3-Reranker 0.6B when it's in the Hugging Face cache and the native library is built; skipped otherwise. */
class RerankIT {
    static final Weights W = Weights.QWEN3_RERANKER_0_6B;
    static final String QUERY = "How hot does a kiln get?";
    static final List<String> DOCS = List.of(
            "A pottery kiln fires stoneware at around 1200 to 1300 degrees Celsius.",
            "Nashville is the capital of Tennessee and the home of country music.",
            "Bisque firing in an electric kiln usually peaks near 1000 C, glaze firing hotter.",
            "A kiln is a thermally insulated chamber.");
    /**
     * llama.cpp's llama-embedding on the same model and pairs (the pinned llama.cpp, Vulkan): {@code llama-embedding -m
     * qwen3-reranker-0.6b-q8_0.gguf -ngl 99 -c 8192 -ub 8192 --embd-normalize -1 --embd-output-format raw
     * --embd-separator '<#kiln#>' -p "$QUERY<TAB>$DOC0<#kiln#>$QUERY<TAB>$DOC1..."}, first column ("yes").
     */
    static final float[] LLAMA_EMBEDDING = {0.9974396f, 0.0000205f, 0.9995825f, 0.6255551f};
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, Mode.RERANK);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    @Test
    void ordersTheFixture() {
        List<Scored> ranked = model.rerank(QUERY, DOCS);
        System.out.println(ranked);
        assertEquals(DOCS.size(), ranked.size());
        // the two about kiln temperatures first, then the vague one, then Nashville
        assertEquals(List.of(2, 0, 3, 1), ranked.stream().map(Scored::index).toList());
        for (Scored s : ranked) assertTrue(s.score() >= 0 && s.score() <= 1, s.toString());
    }

    @Test
    void scoresMatchLlamaEmbedding() {
        for (Scored s : model.rerank(QUERY, DOCS)) {
            System.out.printf("doc %d: fornax %.7f, llama-embedding %.7f%n", s.index(), s.score(), LLAMA_EMBEDDING[s.index()]);
            // The reference scores are llama-embedding's on Vulkan. A document near 0.5 moves with the backend's kernels
            // (the softmax magnifies small differences there: doc 3 is 0.626 on Vulkan, 0.576 on Metal), so it gets a
            // wider margin; the clear ones must match closely on any backend.
            double margin = Math.abs(LLAMA_EMBEDDING[s.index()] - 0.5) < 0.2 ? 0.06 : 0.01;
            assertEquals(LLAMA_EMBEDDING[s.index()], s.score(), margin, "doc " + s.index());
        }
    }

    /**
     * Nearly the same: a borderline document scores 0.626 in the batch and 0.598 alone, in llama-embedding too (the batch
     * changes the GPU kernels' shapes, and the softmax magnifies the difference near 0.5).
     */
    @Test
    void oneDocumentAtATimeScoresNearlyTheSame() {
        // A decided document (p near 0 or 1) scores the same alone as in a batch; a sequence-handling bug would move
        // it. A borderline one (doc 3, p about 0.5 to 0.6) sits where the sigmoid is steepest, and llama.cpp's own
        // llama-embedding moves it by more than 0.05 between a batch and a single sequence on HIP (0.589 vs 0.525),
        // and between backends (CPU 0.634): #80.
        List<Scored> all = model.rerank(QUERY, DOCS);
        for (Scored s : all) {
            double alone = model.rerank(QUERY, List.of(DOCS.get(s.index()))).get(0).score();
            boolean decided = s.score() < 0.1 || s.score() > 0.9;
            assertEquals(s.score(), alone, decided ? 0.01 : 0.1, "doc " + s.index());
        }
    }

    @Test
    void moreDocumentsThanOneBatchTakes() {
        // llama.cpp runs at most 256 sequences at once; the answer is last, in the second batch
        List<String> docs = new ArrayList<>(IntStream.range(0, 299).mapToObj(i -> DOCS.get(1) + " (" + i + ")").toList());
        docs.add(DOCS.get(0));
        List<Scored> ranked = model.rerank(QUERY, docs);
        assertEquals(300, ranked.size());
        assertEquals(299, ranked.get(0).index());
        assertEquals(LLAMA_EMBEDDING[0], ranked.get(0).score(), 0.01);
    }

    @Test
    void emptyQueryOrDocumentStillScores() {
        assertEquals(1, model.rerank(QUERY, List.of("")).size());
        assertEquals(1, model.rerank("", List.of(DOCS.get(0))).size());
    }

    @Test
    void aPairLongerThanTheContextFails() {
        try (var small = new LlamaModel(W, -1, 128, Mode.RERANK)) {
            var e = assertThrows(IllegalStateException.class, () -> small.rerank(QUERY, List.of("kiln ".repeat(200))));
            assertTrue(e.getMessage().contains("tokens"), e.getMessage());
        }
    }

    @Test
    void noDocumentsNoScores() { assertEquals(List.of(), model.rerank(QUERY, List.of())); }

    @Test
    void rerankAsyncAndCancel() throws Exception {
        assertEquals(2, model.rerankAsync(QUERY, DOCS).get().get(0).index());
        List<String> many = IntStream.range(0, 5000).mapToObj(i -> DOCS.get(i % DOCS.size()) + " (" + i + ")").toList();
        var f = model.rerankAsync(QUERY, many);
        Thread.sleep(300);
        model.cancel();
        assertThrows(CancellationException.class, f::get);  // get() throws it unwrapped
        assertEquals(DOCS.size(), model.rerank(QUERY, DOCS).size());
    }

    @Test
    void aRerankerOnlyReranks() {
        assertThrows(IllegalStateException.class, () -> model.embed("hello"));
        assertThrows(IllegalStateException.class, () -> model.chat(List.of(Message.user("hello")), Options.DEFAULT));
        assertThrows(IllegalStateException.class, model::embeddingSize);
    }

    @Test
    void aRerankerIsntAnEmbeddingModel() {
        var e = assertThrows(IllegalStateException.class, () -> new LlamaModel(W, -1, 512, Mode.EMBED));
        assertTrue(e.getMessage().contains("reranker"), e.getMessage());
    }
}
