package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbedOptionsTest {
    @Test
    void writesEachModelsPrefix() {
        assertEquals("Instruct: find kilns\nQuery:", EmbedOptions.query("find kilns").prefix());
        assertEquals("task: search result | query: ", EmbedOptions.gemmaQuery("search result").prefix());
        assertEquals("title: none | text: ", EmbedOptions.gemmaDocument().prefix());
        assertTrue(EmbedOptions.gemmaQuery("clustering").normalize() && EmbedOptions.gemmaDocument().normalize());
        assertNull(EmbedOptions.DEFAULT.prefix());
    }
}
