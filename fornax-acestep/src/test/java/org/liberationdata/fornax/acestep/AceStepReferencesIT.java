package org.liberationdata.fornax.acestep;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** AceStepModel by catalogue name and model reference: refusals need the native library only, loading the set. */
class AceStepReferencesIT {
    static void assumeLibrary() {
        String dir = System.getProperty("fornax.library.path");
        assumeTrue(dir != null && new java.io.File(dir, System.mapLibraryName("fornax_acestep")).isFile(), "native library not built");
    }

    @Test
    void anotherEnginesSetOrAFileIsRefusedBeforeAnyDownload() {
        assumeLibrary();
        var e = assertThrows(IllegalArgumentException.class, () -> new AceStepModel("yue2"));
        assertTrue(e.getMessage().startsWith("yue2 is a song model (yue2.cpp); AceStepModel loads ACE-Step song models"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> new AceStepModel("Serveurperso/ACE-Step-1.5-GGUF:vae-BF16.gguf"));
        assertTrue(e.getMessage().contains("by name (ace-step-turbo, ace-step, ace-step-xl-turbo, ace-step-xl)"), e.getMessage());
        // a set's name where one file goes
        e = assertThrows(IllegalArgumentException.class, () -> new AceStepModel(null, "ace-step", "/d.gguf", "/v.gguf", true));
        assertTrue(e.getMessage().contains("one file per path here"), e.getMessage());
    }

    @Test
    void loadsTheTurboSetByName() {
        assumeLibrary();
        assumeTrue(FornaxHub.standard().status(ModelCatalog.require("ace-step-turbo")).available(), "ace-step-turbo not downloaded: fornax pull ace-step-turbo");
        try (var model = new AceStepModel("ace-step-turbo")) {
            var r = model.generate("{\"caption\": \"solo piano\", \"lyrics\": \"[Instrumental]\", \"duration\": 4, \"seed\": 1, "
                    + "\"inference_steps\": 4, \"guidance_scale\": 1.0, \"shift\": 3.0}");
            assertTrue(r.audio().seconds() > 3, "seconds: " + r.audio().seconds());
        }
    }
}
