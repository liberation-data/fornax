package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.WorkerCrashedException;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Every EmbeddingsIT test again, with the model in a worker process: the same results through the isolated path. */
class EmbeddingsIsolatedIT extends EmbeddingsIT {
    /** Hides EmbeddingsIT's, so the model loads isolated instead (JUnit runs only this one). */
    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = LlamaModel.isolated(W, Mode.EMBED);
    }

    @Test
    void runsInAWorker() { assertTrue(model.isIsolated()); }

    /** The worker's context grows past the 512 it starts at, as in process (#109). */
    @Test
    void aTextPastTheStartingContextEmbeds() {
        String longText = "The kiln fires stoneware to cone 10 over a long, slow climb. ".repeat(70);   // ~1330 tokens
        assertEquals(1024, model.embed(longText).length);
    }

    @Test
    void aWorkerKilledMidEmbedFailsTheCallAndTheNextCallReloadsInEmbedMode() throws Exception {
        long pid = model.worker().pid();
        List<String> many = IntStream.range(0, 20_000).mapToObj(i -> "text " + i + " about kilns, glazes and clay bodies").toList();
        var f = model.embedAsync(many, EmbedOptions.DEFAULT);
        Thread.sleep(300);
        ProcessHandle.of(pid).ifPresent(ProcessHandle::destroyForcibly);
        var e = assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS));
        assertInstanceOf(WorkerCrashedException.class, e.getCause(), String.valueOf(e.getCause()));

        assertEquals(1024, model.embed("reloaded").length);   // a new worker, loaded for EMBED again
        assertNotEquals(pid, model.worker().pid());
    }

    @Test
    void closeDuringAnIsolatedEmbedStopsItAndTheWorker() throws Exception {
        var m = LlamaModel.isolated(W.model(), -1, 8192, Mode.EMBED);
        assertTrue(m.isIsolated());
        assertEquals(1024, m.embeddingSize());
        var worker = m.worker();
        List<String> many = IntStream.range(0, 20_000).mapToObj(i -> "text " + i + " about kilns, glazes and clay bodies").toList();
        var running = m.embedAsync(many, EmbedOptions.DEFAULT);
        Thread.sleep(300);
        long t0 = System.nanoTime();
        m.close();
        long closeMs = (System.nanoTime() - t0) / 1_000_000;
        assertThrows(CancellationException.class, running::get);
        assertTrue(closeMs < 5000, "close waited " + closeMs + " ms: the embed wasn't cancelled");
        assertTrue(worker.isClosed());
    }

    @Test
    void aLoadThatFailsInTheWorkerSaysWhy() {
        var e = assertThrows(IllegalStateException.class, () -> LlamaModel.isolated(W, Mode.RERANK));
        assertTrue(e.getMessage().contains("not a reranker"), e.getMessage());
    }
}
