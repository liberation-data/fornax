package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Worker;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * -Dfornax.isolation=process makes the plain constructor isolate, with no code change; closing the model stops its
 * worker. Qwen3 0.6B, as in LlamaIsolatedIT (a class of its own, so the two models never load at once).
 */
class LlamaIsolationSettingIT {
    @Test
    void theSettingIsolatesTheConstructor() {
        var path = LlamaIsolatedIT.gguf();
        assumeTrue(path.isPresent(), "model not downloaded");
        System.setProperty("fornax.isolation", "process");
        Worker worker;
        long pid;
        try (var model = new LlamaModel(path.get().toString(), -1, 2048)) {
            assertTrue(model.isIsolated());
            worker = model.worker();
            pid = worker.pid();
            String reply = model.chat(LlamaIsolatedIT.FORNAX, LlamaIsolatedIT.GREEDY.withMaxTokens(32)).text();
            assertTrue(reply.toLowerCase().contains("fornax"), reply);
        } finally {
            System.clearProperty("fornax.isolation");
        }
        assertTrue(worker.isClosed());
        assertFalse(ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false), "the worker stopped with the model");
    }

    @Test
    void theSettingIsolatesAnEmbeddingModelToo() {
        var w = LlamaModel.Weights.QWEN3_EMBEDDING_0_6B;
        assumeTrue(w.downloaded(), "model not downloaded: " + w.downloadCommand());
        System.setProperty("fornax.isolation", "process");
        try (var model = new LlamaModel(w, LlamaModel.Mode.EMBED)) {
            assertTrue(model.isIsolated());
            assertEquals(1024, model.embeddingSize());
            assertEquals(1024, model.embed("a glaze that crawls").length);
        } finally {
            System.clearProperty("fornax.isolation");
        }
    }
}
