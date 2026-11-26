package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A model by reference ({@code owner/repo:TAG}) instead of a path. Runs when Qwen3 0.6B (as in LlamaIsolatedIT) is
 * in the Hugging Face cache, so the reference resolves there and nothing is downloaded.
 */
class LlamaRefIT {
    static final String REF = LlamaIsolatedIT.REPO + ":Q8_0";

    @Test
    void loadsByReference() {
        assumeTrue(LlamaIsolatedIT.gguf().isPresent(), "model not downloaded");
        try (var model = new LlamaModel(REF, -1, 2048)) {
            String reply = model.chat(LlamaIsolatedIT.FORNAX, LlamaIsolatedIT.GREEDY.withMaxTokens(32)).text();
            assertTrue(reply.toLowerCase().contains("fornax"), reply);
        }
    }

    @Test
    void anIsolatedModelByReferenceLoadsTheFileInTheWorker() {
        assumeTrue(LlamaIsolatedIT.gguf().isPresent(), "model not downloaded");
        try (var model = LlamaModel.isolated(REF, null, -1, 2048)) {
            assertTrue(model.isIsolated());
            assertTrue(model.chat(LlamaIsolatedIT.FORNAX, LlamaIsolatedIT.GREEDY.withMaxTokens(32)).text().toLowerCase().contains("fornax"));
        }
    }

    @Test
    void aMissingPathIsntMistakenForAReference() {
        var e = assertThrows(RuntimeException.class, () -> new LlamaModel("models/not-there.gguf", 0, 512));
        assertFalse(e.getMessage().contains("huggingface"), e.getMessage());
    }

    @Test
    void anAliasForAnotherEngineIsRefusedBeforeAnyDownload() {
        var e = assertThrows(IllegalArgumentException.class, () -> new LlamaModel("flux2-klein", 0, 512));
        assertTrue(e.getMessage().contains("image model"), e.getMessage());
        var speech = assertThrows(IllegalArgumentException.class, () -> LlamaModel.isolated("kokoro"));
        assertTrue(speech.getMessage().contains("speech model"), speech.getMessage());
    }
}
