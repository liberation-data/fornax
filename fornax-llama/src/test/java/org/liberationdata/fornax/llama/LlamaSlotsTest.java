package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** What the slots constructor refuses before it loads anything: no model or native library needed. */
class LlamaSlotsTest {
    static final String NONE = "/no/such/model.gguf";

    @Test void slotsNeedAtLeastOneAndEachSlotsContextSize() {
        var e = assertThrows(IllegalArgumentException.class, () -> new LlamaModel(NONE, -1, 2048, 0));
        assertTrue(e.getMessage().contains("slots must be at least 1"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> new LlamaModel(NONE, -1, 0, 2));
        assertTrue(e.getMessage().contains("each slot's context size"), e.getMessage());
    }

    @Test void anIsolatedModelWithSlotsChecksTheSameBeforeItStartsAWorker() {
        var e = assertThrows(IllegalArgumentException.class, () -> LlamaModel.isolated(NONE, -1, 0, 2));
        assertTrue(e.getMessage().contains("each slot's context size"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> LlamaModel.isolated(NONE, -1, 2048, 0));
        assertTrue(e.getMessage().contains("slots must be at least 1"), e.getMessage());
    }

    /** A worker's config with slots for a mode that isn't chat (the parent never sends one) is refused as it loads. */
    @Test void slotsAreForChatOnly() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> new LlamaService(LlamaService.config(NONE, null, LlamaModel.Mode.EMBED, -1, 2048, 2)));
        assertTrue(e.getMessage().contains("slots are for chat"), e.getMessage());
    }
}
