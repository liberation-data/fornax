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

    @Test void slotsAreRefusedUnderProcessIsolation() {
        System.setProperty("fornax.isolation", "process");
        try {
            var e = assertThrows(IllegalArgumentException.class, () -> new LlamaModel(NONE, -1, 2048, 2));
            assertTrue(e.getMessage().contains("runs in-process"), e.getMessage());
        } finally {
            System.clearProperty("fornax.isolation");
        }
    }
}
