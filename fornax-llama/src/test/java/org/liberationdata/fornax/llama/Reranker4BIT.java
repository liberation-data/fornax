package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs Qwen3-Reranker 4B (#153) on {@link RerankIT}'s fixture when it's in the Hugging Face cache and the native library
 * is built; skipped otherwise. The GGUF is a community conversion, so the reference is llama.cpp's own llama-embedding
 * on the same file and pairs, as RerankIT's is for the 0.6B.
 */
class Reranker4BIT {
    static final Weights W = Weights.QWEN3_RERANKER_4B;
    /** As RerankIT.LLAMA_EMBEDDING, with the 4B: the first column ("yes") of llama-embedding's raw output. */
    static final float[] LLAMA_EMBEDDING = {0.9884772f, 0.0000008f, 0.9886600f, 0.7005172f};
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, Mode.RERANK);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    @Test
    void loadsInRerankMode() {
        assertEquals(Mode.RERANK, model.mode());
    }

    @Test
    void ordersTheFixtureAsThe06BDoes() {
        List<Scored> ranked = model.rerank(RerankIT.QUERY, RerankIT.DOCS);
        System.out.println(ranked);
        // The two about kiln temperatures first (0.98866 and 0.98848 in llama-embedding: too close to fix their order
        // across backends), then the vague one, then Nashville.
        List<Integer> order = ranked.stream().map(Scored::index).toList();
        assertEquals(Set.of(0, 2), Set.copyOf(order.subList(0, 2)), order.toString());
        assertEquals(List.of(3, 1), order.subList(2, 4));
        for (Scored s : ranked) assertTrue(s.score() >= 0 && s.score() <= 1, s.toString());
    }

    @Test
    void scoresMatchLlamaEmbedding() { RerankIT.assertScoresMatch(model, LLAMA_EMBEDDING); }
}
