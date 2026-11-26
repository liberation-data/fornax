package org.liberationdata.fornax.yue2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Yue2Model by catalogue name and model reference: refusals, which need the native library but no model. */
class Yue2ReferencesIT {
    @Test
    void anotherEnginesSetOrAFileIsRefusedBeforeAnyDownload() {
        String dir = System.getProperty("fornax.library.path");
        assumeTrue(dir != null && new java.io.File(dir, System.mapLibraryName("fornax_yue2")).isFile(), "native library not built");
        var e = assertThrows(IllegalArgumentException.class, () -> new Yue2Model("ace-step"));
        assertTrue(e.getMessage().startsWith("ace-step is a song model (acestep.cpp); Yue2Model loads YuE2 song models"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> new Yue2Model("/nowhere/yue2.gguf"));
        assertTrue(e.getMessage().contains("by name (yue2)"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> new Yue2Model("/m.gguf", "https://huggingface.co/Serveurperso/YuE2-GGUF/blob/main/YuE2-Vae-F32.gguf", true, 0));
        assertTrue(e.getMessage().contains("Serveurperso/YuE2-GGUF:YuE2-Vae-F32.gguf"), e.getMessage());
    }
}
