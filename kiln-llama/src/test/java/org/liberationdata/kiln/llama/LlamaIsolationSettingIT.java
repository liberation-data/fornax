package org.liberationdata.kiln.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Worker;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * -Dkiln.isolation=process makes the plain constructor isolate, with no code change; closing the model stops its
 * worker. Qwen3 0.6B, as in LlamaIsolatedIT (a class of its own, so the two models never load at once).
 */
class LlamaIsolationSettingIT {
    @Test
    void theSettingIsolatesTheConstructor() {
        var path = LlamaIsolatedIT.gguf();
        assumeTrue(path.isPresent(), "model not downloaded");
        System.setProperty("kiln.isolation", "process");
        Worker worker;
        long pid;
        try (var model = new LlamaModel(path.get().toString(), -1, 2048)) {
            assertTrue(model.isIsolated());
            worker = model.worker();
            pid = worker.pid();
            String reply = model.chat(LlamaIsolatedIT.KILN, LlamaIsolatedIT.GREEDY.withMaxTokens(32)).text();
            assertTrue(reply.toLowerCase().contains("kiln"), reply);
        } finally {
            System.clearProperty("kiln.isolation");
        }
        assertTrue(worker.isClosed());
        assertFalse(ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false), "the worker stopped with the model");
    }
}
