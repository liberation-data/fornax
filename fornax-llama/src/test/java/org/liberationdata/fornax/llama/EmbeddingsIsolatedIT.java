package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.llama.LlamaModel.Mode;

import static org.junit.jupiter.api.Assertions.assertTrue;
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
}
