package org.liberationdata.kiln.acestep;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The shim exports every function AceStepModel binds, kiln_acestep_set_cancel included (no model needed). */
class AceStepBindingsTest {
    @Test
    void bindsEveryFunction() {
        String dir = System.getProperty("kiln.library.path");
        assumeTrue(dir != null && new java.io.File(dir, System.mapLibraryName("kiln_acestep")).isFile(), "native library not built");
        assertDoesNotThrow(() -> Class.forName(AceStepModel.class.getName()));
    }
}
