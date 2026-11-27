package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Worker;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Every RerankIT test again, with the model in a worker process: the same results through the isolated path. */
class RerankIsolatedIT extends RerankIT {
    /** Hides RerankIT's, so the model loads isolated instead (JUnit runs only this one). */
    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = LlamaModel.isolated(W, Mode.RERANK);
    }

    @Test
    void runsInAWorker() { assertTrue(model.isIsolated()); }

    @Test
    void anEmbedderAndARerankerShareOneWorker() {
        var e = Weights.QWEN3_EMBEDDING_0_6B;
        assumeTrue(e.downloaded(), "model not downloaded: " + e.downloadCommand());
        try (var worker = new Worker("embed-and-rerank", Worker.Options.DEFAULT)) {
            try (var embedder = LlamaModel.isolated(worker, e.model(), -1, 2048, Mode.EMBED);
                 var reranker = LlamaModel.isolated(worker, W.model(), -1, 2048, Mode.RERANK)) {
                assertSame(worker, embedder.worker());
                assertSame(worker, reranker.worker());
                assertEquals(1024, embedder.embed("a glaze that crawls").length);
                assertEquals(2, reranker.rerank(QUERY, DOCS).get(0).index());
            }
            assertFalse(worker.isClosed(), "closing the models leaves a shared worker running");
        }
    }
}
